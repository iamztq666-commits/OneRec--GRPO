"""
生成式推荐 GRPO 训练

SFT 之后，用 GRPO 优化 Qwen 生成策略：
  1. 对同一用户历史，生成 G=8 条候选 session
  2. BehaviorPredictor 对每条 session 打分（click/skip/leave）
  3. 加多样性、指令跟随奖励
  4. Group 归一化 advantage，PPO-clip 更新 LoRA 权重

作用对象是"生成出的 session SID token 序列"，不是排序头分数。
"""
import _path  # noqa

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from typing import List, Dict, Tuple

from config import cfg
from generative_rec import GenerativeRec
from behavior_predictor import BehaviorPredictor


def compute_session_reward(
    item_ids: List[int],
    user_mindset: np.ndarray,
    item_embeddings: np.ndarray,
    behavior_predictor: BehaviorPredictor,
) -> float:
    """
    对一条生成 session 计算奖励：
      click × 0.5 + stay × 0.1 - leave × 1.0 - diversity_penalty
    """
    if not item_ids:
        return -1.0

    total_reward = 0.0
    left = False

    valid_embs = []
    for iid in item_ids:
        if iid >= len(item_embeddings):
            continue
        item_emb = item_embeddings[iid]
        action = behavior_predictor.predict_action(user_mindset, item_emb)

        if action == "click":
            total_reward += cfg.reward_click
            valid_embs.append(item_emb)
        elif action == "leave":
            total_reward += cfg.reward_leave
            left = True
            break

    if not left:
        total_reward += cfg.reward_session_step

    # 多样性惩罚
    if len(valid_embs) >= 2:
        embs = np.array(valid_embs)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        normed = embs / (norms + 1e-9)
        sim = normed @ normed.T
        upper = sim[np.triu_indices(len(normed), k=1)]
        if np.mean(upper) > cfg.diversity_sim_threshold:
            total_reward += cfg.reward_diversity_penalty

    return total_reward


class GenGRPOTrainer:
    def __init__(
        self,
        gen_rec: GenerativeRec,
        behavior_predictor: BehaviorPredictor,
        item_embeddings: np.ndarray,
        user_histories: Dict[int, List[int]],
        user_profiles: Dict[int, np.ndarray],
    ):
        self.gen_rec = gen_rec
        self.behavior_predictor = behavior_predictor
        self.item_embeddings = item_embeddings
        self.user_histories = user_histories
        self.user_profiles = user_profiles

        self.optimizer = AdamW(
            gen_rec.model.parameters(), lr=cfg.grpo_lr * 0.1, weight_decay=0.01
        )

    def _rollout_user(self, uid: int) -> Tuple[List[Tuple[List[int], float]], float]:
        """
        对单个用户生成 G=8 条 session，计算各自奖励。
        返回: [(item_ids, reward), ...], best_reward
        """
        history = self.user_histories.get(uid, [])[-20:]
        mindset = self.user_profiles.get(uid, np.zeros(cfg.embed_dim))
        sessions = []

        # greedy × 1
        iids = self.gen_rec.generate_session(history, temperature=0.0)
        r = compute_session_reward(iids, mindset, self.item_embeddings, self.behavior_predictor)
        sessions.append((iids, r))

        # stochastic × G-1
        for _ in range(cfg.grpo_group_size - 1):
            iids = self.gen_rec.generate_session(history, temperature=cfg.grpo_temperature)
            r = compute_session_reward(iids, mindset, self.item_embeddings, self.behavior_predictor)
            sessions.append((iids, r))

        best_r = max(r for _, r in sessions)
        return sessions, best_r

    def train(self, user_ids: List[int], epochs: int = 3) -> List[Dict]:
        logs = []
        print(f"\n[GenGRPO] Starting: {epochs} epochs, G={cfg.grpo_group_size}")

        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            np.random.shuffle(user_ids)
            epoch_best_rewards = []
            total_loss = 0.0
            n_updates = 0

            for uid in tqdm(user_ids[:cfg.n_rollout_episodes], desc="GRPO rollout"):
                history = self.user_histories.get(uid, [])[-20:]
                if len(history) < 3:
                    continue

                sessions, best_r = self._rollout_user(uid)
                epoch_best_rewards.append(best_r)

                rewards = np.array([r for _, r in sessions], dtype=np.float32)
                mean_r, std_r = rewards.mean(), rewards.std() + 1e-8

                # PPO-clip 更新
                for iids, r in sessions:
                    if not iids:
                        continue
                    adv = float((r - mean_r) / std_r)
                    adv_t = torch.tensor(adv, dtype=torch.float32,
                                         device=next(self.gen_rec.model.parameters()).device)

                    # 计算新旧 log prob
                    log_prob_new = self.gen_rec.compute_session_logp(history, iids)

                    with torch.no_grad():
                        log_prob_old = log_prob_new.detach()

                    ratio = torch.exp(log_prob_new - log_prob_old)
                    obj = ratio * adv_t
                    obj_clipped = torch.clamp(
                        ratio, 1 - cfg.grpo_epsilon, 1 + cfg.grpo_epsilon) * adv_t
                    loss = -torch.min(obj, obj_clipped)

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.gen_rec.model.parameters(), 1.0)
                    self.optimizer.step()

                    total_loss += loss.item()
                    n_updates += 1

            avg_best = float(np.mean(epoch_best_rewards)) if epoch_best_rewards else 0.0
            avg_loss = total_loss / max(n_updates, 1)
            print(f"  avg best reward: {avg_best:.3f} | loss: {avg_loss:.4f}")

            self.gen_rec.save_lora()
            log = {"epoch": epoch, "avg_best_reward": avg_best, "loss": avg_loss}
            logs.append(log)

        return logs
