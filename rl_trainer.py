"""
Session-level GRPO Trainer（对齐 OneRec 思路）

OneRec 核心：session-wise 生成
  传统 step-level GRPO：每步采样 G 个推荐列表，step reward 做 group 归一化
  → 信号局部，无法捕捉长程 session 效果

  本实现（session-level GRPO）：
    1. 每个用户生成 G 条完整 session（greedy × 1 + stochastic × G-1）
    2. 每条 session 累计奖励 R_g 做 group 归一化：A_g = (R_g - mean) / std
    3. session 内所有步共享同一 advantage A_g 做 PPO-clip 更新
    4. 奖励信号全局，信用分配更准确
"""
import json
import copy
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
# Rollout Buffer（存 session 内所有 step）
# ─────────────────────────────────────────────
class RolloutBuffer:
    def __init__(self):
        self.features: List[torch.Tensor] = []      # (K, INPUT_DIM) per step
        self.actions: List[List[int]] = []           # selected indices per step
        self.session_rewards: List[float] = []       # session 累计奖励（用于 logging）
        self.advantages: List[float] = []            # session-level advantage（每步共享）
        self.log_probs_old: List[torch.Tensor] = []  # old log prob per step

    def clear(self):
        self.features.clear()
        self.actions.clear()
        self.session_rewards.clear()
        self.advantages.clear()
        self.log_probs_old.clear()

    def __len__(self):
        return len(self.advantages)


def compute_log_prob(
    ranking_head: RankingHead,
    features: torch.Tensor,
    actions: List[int],
) -> torch.Tensor:
    """计算推荐列表的 log probability（softmax over candidates）"""
    ranking_head.train()
    user_emb = features[:, :cfg.embed_dim].to(cfg.device)
    item_emb = features[:, cfg.embed_dim:cfg.embed_dim * 2].to(cfg.device)
    rest = features[:, cfg.embed_dim * 2:].to(cfg.device)

    scores = ranking_head(user_emb, item_emb, rest[:, 0:1], rest[:, 1:2], rest[:, 2:3])
    log_probs = F.log_softmax(scores, dim=0)
    return log_probs[actions].sum()


# ─────────────────────────────────────────────
# Session-level GRPO Trainer
# ─────────────────────────────────────────────
class GRPOTrainer:
    def __init__(
        self,
        ranking_head: RankingHead,
        rec_agent: RecAgent,
        user_sim: UserSimulator,
        env: RecoWorldEnv,
    ):
        self.ranking_head = ranking_head
        self.rec_agent = rec_agent
        self.user_sim = user_sim
        self.env = env

        self.optimizer = AdamW(
            ranking_head.parameters(), lr=cfg.grpo_lr, weight_decay=1e-4
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.grpo_epochs * cfg.n_rollout_episodes,
            eta_min=cfg.grpo_lr * 0.1,
        )
        self.buffer = RolloutBuffer()
        self.train_log: List[Dict] = []

    # ── 辅助：FAISS 召回 + 去重 ──
    def _recall_candidates(self, state: MDPState) -> List[int]:
        if self.rec_agent.retriever is not None:
            candidates = self.rec_agent.retriever.retrieve(state.mindset, cfg.recall_topk)
        else:
            candidates = list(np.random.choice(
                self.env.data.n_items, cfg.recall_topk, replace=False))

        seen = set(state.history_iids[-50:])
        candidates = [c for c in candidates if c not in seen]
        if len(candidates) < cfg.rec_list_size:
            extras = list(np.random.choice(
                self.env.data.n_items,
                cfg.rec_list_size - len(candidates),
                replace=False,
            ))
            candidates.extend(extras)
        return candidates

    # ── 辅助：从 ranking head 采样推荐列表 ──
    def _select_actions(
        self, features: torch.Tensor, temperature: float
    ) -> List[int]:
        with torch.no_grad():
            f = features.to(cfg.device)
            scores = self.ranking_head(
                f[:, :cfg.embed_dim],
                f[:, cfg.embed_dim:cfg.embed_dim * 2],
                f[:, cfg.embed_dim * 2: cfg.embed_dim * 2 + 1],
                f[:, cfg.embed_dim * 2 + 1: cfg.embed_dim * 2 + 2],
                f[:, cfg.embed_dim * 2 + 2: cfg.embed_dim * 2 + 3],
            )
            if temperature == 0.0:
                return torch.argsort(scores, descending=True)[: cfg.rec_list_size].cpu().tolist()
            probs = F.softmax(scores / temperature, dim=0)
            return torch.multinomial(probs, cfg.rec_list_size, replacement=False).cpu().tolist()

    # ── 核心：生成一条完整 session ──
    def _generate_session(
        self, uid: int, temperature: float
    ) -> Tuple[List[Tuple[torch.Tensor, List[int]]], float]:
        """
        生成用户 uid 的一条完整 session。
        temperature=0 → greedy；>0 → stochastic（探索）
        返回：(trajectory, total_reward)
          trajectory = [(features_cpu, actions), ...]  # 每步一个 entry
        """
        state = self.env.reset(uid)
        trajectory: List[Tuple[torch.Tensor, List[int]]] = []
        total_reward = 0.0

        while not state.done:
            candidates = self._recall_candidates(state)
            features = self.rec_agent.get_scoring_features(state, candidates)
            selected = self._select_actions(features, temperature)
            rec_list = [candidates[i] for i in selected]

            user_actions, instruction = self.user_sim.evaluate_recommendations(state, rec_list)
            result = self.env.step(state, rec_list, user_actions, instruction)

            trajectory.append((features.cpu(), selected))
            total_reward += result.reward
            state = result.next_state

        return trajectory, total_reward

    # ── session-level GRPO rollout 收集 ──
    def collect_rollouts(self, user_ids: List[int]) -> float:
        """
        对每个用户生成 G 条 session（1 greedy + G-1 stochastic）
        用 session 累计奖励做 group 归一化，session 内每步共享同一 advantage
        """
        self.buffer.clear()
        total_best_reward = 0.0
        n_users = 0

        for uid in tqdm(user_ids[: cfg.n_rollout_episodes], desc="Collecting sessions"):
            # ── 生成 G 条 session ──
            sessions: List[Tuple[List, float]] = []
            # session 0：greedy（baseline）
            traj, r = self._generate_session(uid, temperature=0.0)
            sessions.append((traj, r))
            # session 1..G-1：stochastic（探索）
            for _ in range(cfg.grpo_group_size - 1):
                traj, r = self._generate_session(uid, temperature=cfg.grpo_temperature)
                sessions.append((traj, r))

            # ── session-level group 归一化 ──
            rewards = np.array([r for _, r in sessions], dtype=np.float32)
            mean_r = rewards.mean()
            std_r = rewards.std() + 1e-8

            for traj, r in sessions:
                adv = float((r - mean_r) / std_r)  # 同一 session 内所有步共享此 advantage
                for features, actions in traj:
                    with torch.no_grad():
                        log_p = compute_log_prob(self.ranking_head, features, actions)
                    self.buffer.features.append(features)
                    self.buffer.actions.append(actions)
                    self.buffer.session_rewards.append(r)
                    self.buffer.advantages.append(adv)
                    self.buffer.log_probs_old.append(log_p)

            total_best_reward += float(rewards.max())
            n_users += 1

        avg_best = total_best_reward / max(n_users, 1)
        print(f"  Buffer: {len(self.buffer)} steps | avg best-session reward: {avg_best:.3f}")
        return avg_best

    # ── PPO-clip 更新 ──
    def update(self) -> Dict:
        if len(self.buffer) == 0:
            return {}

        total_loss = 0.0
        n_batches = 0
        indices = np.arange(len(self.buffer))

        for _ in range(cfg.grpo_inner_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(indices), cfg.grpo_batch_size):
                batch_idx = indices[start: start + cfg.grpo_batch_size]

                # ── 修正：用 list + stack，而非 requires_grad=True 初始值累加 ──
                losses = []
                for i in batch_idx:
                    features = self.buffer.features[i]
                    actions = self.buffer.actions[i]
                    adv = self.buffer.advantages[i]
                    log_prob_old = self.buffer.log_probs_old[i].to(cfg.device)

                    log_prob_new = compute_log_prob(self.ranking_head, features, actions)
                    ratio = torch.exp(log_prob_new - log_prob_old)
                    adv_t = torch.tensor(adv, dtype=torch.float32, device=cfg.device)

                    obj = ratio * adv_t
                    obj_clipped = torch.clamp(ratio, 1 - cfg.grpo_epsilon,
                                              1 + cfg.grpo_epsilon) * adv_t
                    losses.append(-torch.min(obj, obj_clipped))

                batch_loss = torch.stack(losses).mean()

                self.optimizer.zero_grad()
                batch_loss.backward()
                nn.utils.clip_grad_norm_(self.ranking_head.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()

                total_loss += batch_loss.item()
                n_batches += 1

        return {"policy_loss": total_loss / max(n_batches, 1)}

    # ── 完整训练循环 ──
    def train(self, user_ids: List[int]) -> List[Dict]:
        print(f"\nStarting Session-level GRPO: {cfg.grpo_epochs} epochs, "
              f"G={cfg.grpo_group_size} sessions/user")
        logs = []

        for epoch in range(cfg.grpo_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{cfg.grpo_epochs}")
            np.random.shuffle(user_ids)

            avg_reward = self.collect_rollouts(user_ids)
            update_info = self.update()

            log = {"epoch": epoch + 1, "avg_reward": avg_reward, **update_info}
            logs.append(log)
            self.train_log.append(log)
            print(f"  Avg best-session reward: {avg_reward:.4f} | "
                  f"Policy loss: {update_info.get('policy_loss', 0):.4f}")

            torch.save(
                self.ranking_head.state_dict(),
                f"{cfg.output_dir}/ranking_head_epoch{epoch+1}.pt",
            )

        with open(f"{cfg.output_dir}/train_log.json", "w") as f:
            json.dump(logs, f, indent=2)

        return logs
