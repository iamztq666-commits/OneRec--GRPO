"""
GRPO训练器
- 收集rollout trajectory（仿真用户 × Rec Agent）
- 计算group-normalized advantage
- 优化MLP ranking head

GRPO简介：
  每个state采样G组推荐列表，用reward相对均值做advantage
  无需critic网络，适合LLM-based policy
  这里policy = MLP ranking head（Qwen固定做特征提取）
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import List, Dict, Tuple
from tqdm import tqdm

from config import cfg
from env import RecoWorldEnv, MDPState
from rec_agent import RecAgent, RankingHead
from user_sim import UserSimulator


# ─────────────────────────────────────────────
# Rollout Buffer
# ─────────────────────────────────────────────
class RolloutBuffer:
    def __init__(self):
        self.features: List[torch.Tensor] = []    # (K, INPUT_DIM)
        self.actions: List[List[int]] = []        # selected item indices
        self.rewards: List[float] = []
        self.advantages: List[float] = []
        self.log_probs_old: List[torch.Tensor] = []

    def clear(self):
        self.features.clear()
        self.actions.clear()
        self.rewards.clear()
        self.advantages.clear()
        self.log_probs_old.clear()

    def __len__(self):
        return len(self.rewards)


def compute_log_prob(ranking_head: RankingHead, features: torch.Tensor,
                     actions: List[int]) -> torch.Tensor:
    """
    计算推荐列表的log probability
    将ranking scores转为概率分布（softmax over candidates）
    选中的item的log_prob求和
    """
    ranking_head.train()
    K = features.shape[0]
    # 从features中拆分出需要的维度
    # features: (K, 1536+1536+3)
    user_emb = features[:, :cfg.embed_dim].to(cfg.device)
    item_emb = features[:, cfg.embed_dim:cfg.embed_dim*2].to(cfg.device)
    rest = features[:, cfg.embed_dim*2:].to(cfg.device)
    instr_sim = rest[:, 0:1]
    fatigue = rest[:, 1:2]
    step = rest[:, 2:3]

    scores = ranking_head(user_emb, item_emb, instr_sim, fatigue, step)  # (K,)
    log_probs = F.log_softmax(scores, dim=0)  # (K,)

    # 选中actions的log_prob之和（类似排列的log_prob近似）
    action_log_prob = log_probs[actions].sum()
    return action_log_prob


# ─────────────────────────────────────────────
# GRPO Trainer
# ─────────────────────────────────────────────
class GRPOTrainer:
    def __init__(self, ranking_head: RankingHead, rec_agent: RecAgent,
                 user_sim: UserSimulator, env: RecoWorldEnv):
        self.ranking_head = ranking_head
        self.rec_agent = rec_agent
        self.user_sim = user_sim
        self.env = env

        self.optimizer = AdamW(
            ranking_head.parameters(),
            lr=cfg.grpo_lr,
            weight_decay=1e-4
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.grpo_epochs * cfg.n_rollout_episodes,
            eta_min=cfg.grpo_lr * 0.1
        )
        self.buffer = RolloutBuffer()
        self.train_log: List[Dict] = []

    def collect_rollouts(self, user_ids: List[int]):
        """
        收集rollout数据
        对每个state，采样G组不同推荐（通过在MLP输出上加噪声实现）
        """
        self.buffer.clear()
        total_reward = 0.0
        n_episodes = 0

        for uid in tqdm(user_ids[:cfg.n_rollout_episodes], desc="Collecting rollouts"):
            state = self.env.reset(uid)
            episode_reward = 0.0

            while not state.done:
                # 获取候选特征
                from rec_agent import FAISSRetriever
                if self.rec_agent.retriever is not None:
                    candidates = self.rec_agent.retriever.retrieve(
                        state.mindset, cfg.recall_topk)
                else:
                    candidates = list(np.random.choice(
                        self.env.data.n_items, cfg.recall_topk, replace=False))

                seen = set(state.history_iids[-50:])
                candidates = [c for c in candidates if c not in seen]
                if len(candidates) < cfg.rec_list_size:
                    extras = list(np.random.choice(
                        self.env.data.n_items,
                        cfg.rec_list_size - len(candidates), replace=False))
                    candidates.extend(extras)

                features = self.rec_agent.get_scoring_features(state, candidates)

                # ── GRPO: 采样G组推荐列表 ──
                group_rewards = []
                group_actions = []
                group_log_probs = []

                for g in range(cfg.grpo_group_size):
                    # 加入探索噪声
                    with torch.no_grad():
                        noise_features = features.clone()
                        if g > 0:  # 第0组用greedy，其余加噪声
                            noise = torch.randn_like(
                                noise_features[:, cfg.embed_dim:cfg.embed_dim+1]) * 0.1
                            noise_features[:, cfg.embed_dim:cfg.embed_dim+1] += noise

                        f = noise_features.to(cfg.device)
                        user_e = f[:, :cfg.embed_dim]
                        item_e = f[:, cfg.embed_dim:cfg.embed_dim*2]
                        rest = f[:, cfg.embed_dim*2:]
                        scores = self.ranking_head(
                            user_e, item_e, rest[:, 0:1], rest[:, 1:2], rest[:, 2:3]
                        )

                        if g == 0:
                            # Greedy
                            top_idx = torch.argsort(scores, descending=True)[:cfg.rec_list_size]
                        else:
                            # Stochastic sampling
                            probs = F.softmax(scores / 0.5, dim=0)
                            top_idx = torch.multinomial(probs, cfg.rec_list_size, replacement=False)

                        selected = top_idx.cpu().tolist()

                    rec_list = [candidates[i] for i in selected]
                    user_actions, instruction = self.user_sim.evaluate_recommendations(
                        state, rec_list)
                    result = self.env.step(state, rec_list, user_actions, instruction)
                    group_rewards.append(result.reward)
                    group_actions.append(selected)

                    log_p = compute_log_prob(self.ranking_head, features, selected)
                    group_log_probs.append(log_p.detach())

                # ── GRPO advantage: reward - mean(group rewards) ──
                mean_r = np.mean(group_rewards)
                std_r = np.std(group_rewards) + 1e-8
                for g in range(cfg.grpo_group_size):
                    adv = (group_rewards[g] - mean_r) / std_r
                    self.buffer.features.append(features)
                    self.buffer.actions.append(group_actions[g])
                    self.buffer.rewards.append(group_rewards[g])
                    self.buffer.advantages.append(adv)
                    self.buffer.log_probs_old.append(group_log_probs[g])

                # 用greedy推荐推进状态
                greedy_rec = [candidates[i] for i in group_actions[0]]
                user_actions, instruction = self.user_sim.evaluate_recommendations(
                    state, greedy_rec)
                result = self.env.step(state, greedy_rec, user_actions, instruction)
                episode_reward += result.reward
                state = result.next_state

            total_reward += episode_reward
            n_episodes += 1

        avg_reward = total_reward / max(n_episodes, 1)
        print(f"  Collected {len(self.buffer)} samples, avg episode reward: {avg_reward:.3f}")
        return avg_reward

    def update(self) -> Dict:
        """GRPO policy update（PPO-clip风格）"""
        if len(self.buffer) == 0:
            return {}

        total_loss = 0.0
        n_batches = 0
        indices = np.arange(len(self.buffer))

        for epoch in range(3):  # 每次rollout做3个mini-epoch
            np.random.shuffle(indices)
            for start in range(0, len(indices), cfg.grpo_batch_size):
                batch_idx = indices[start:start + cfg.grpo_batch_size]

                batch_loss = torch.tensor(0.0, requires_grad=True).to(cfg.device)
                for i in batch_idx:
                    features = self.buffer.features[i]
                    actions = self.buffer.actions[i]
                    adv = self.buffer.advantages[i]
                    log_prob_old = self.buffer.log_probs_old[i]

                    # 重新计算log_prob
                    log_prob_new = compute_log_prob(self.ranking_head, features, actions)

                    # PPO-clip ratio
                    ratio = torch.exp(log_prob_new - log_prob_old.to(cfg.device))
                    adv_t = torch.tensor(adv, dtype=torch.float32).to(cfg.device)

                    obj = ratio * adv_t
                    obj_clipped = torch.clamp(ratio, 1 - cfg.grpo_epsilon,
                                              1 + cfg.grpo_epsilon) * adv_t
                    loss_i = -torch.min(obj, obj_clipped)
                    batch_loss = batch_loss + loss_i / len(batch_idx)

                self.optimizer.zero_grad()
                batch_loss.backward()
                nn.utils.clip_grad_norm_(self.ranking_head.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()

                total_loss += batch_loss.item()
                n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        return {"policy_loss": avg_loss}

    def train(self, user_ids: List[int]) -> List[Dict]:
        """完整训练循环"""
        print(f"\nStarting GRPO training: {cfg.grpo_epochs} epochs")
        logs = []

        for epoch in range(cfg.grpo_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{cfg.grpo_epochs}")

            # 打乱用户顺序
            np.random.shuffle(user_ids)

            # 收集rollout
            avg_reward = self.collect_rollouts(user_ids)

            # 更新policy
            update_info = self.update()

            log = {
                "epoch": epoch + 1,
                "avg_reward": avg_reward,
                **update_info
            }
            logs.append(log)
            self.train_log.append(log)
            print(f"  Avg reward: {avg_reward:.4f} | "
                  f"Policy loss: {update_info.get('policy_loss', 0):.4f}")

            # 保存checkpoint
            ckpt_path = f"{cfg.output_dir}/ranking_head_epoch{epoch+1}.pt"
            torch.save(self.ranking_head.state_dict(), ckpt_path)

        # 保存训练日志
        with open(f"{cfg.output_dir}/train_log.json", "w") as f:
            json.dump(logs, f, indent=2)

        return logs
