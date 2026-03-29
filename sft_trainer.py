"""
SFT Trainer（监督预训练，对应 OneRec Pre-training 阶段）

目的：在 GRPO 之前给 TransformerRankingHead 一个好的初始化
      让模型先学会"用户点击过的 item > 没点击的 item"
      避免 GRPO 从随机初始化开始探索，训练更稳定

数据：KuaiRec 历史交互（watch_ratio >= threshold 为正样本）
损失：Listwise Softmax Cross-Entropy
      对每个用户采样 n_pos 个正样本 + n_neg 个负样本，
      最大化正样本在候选集中的归一化得分

训练流程：
  SFT（1-3 epoch，快）→ GRPO（5-10 epoch，慢）
"""
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import List, Dict
from tqdm import tqdm

from config import cfg
from env import KuaiRecEnvData
from rec_agent import TransformerRankingHead


class SFTTrainer:
    """
    监督预训练 TransformerRankingHead

    每个 user 的训练样本构造：
      候选集 = n_pos 个正样本（历史点击）+ n_neg 个随机负样本
      损失   = Softmax Cross-Entropy（listwise）
               -sum(label_normalized * log_softmax(scores))

    为什么用 listwise 而不是 pointwise BCE：
      与 TransformerRankingHead 的 forward 设计一致（整个候选集联合建模）
      同时也与之后 GRPO 的 log_softmax 概率形式统一
    """

    def __init__(self, ranking_head: TransformerRankingHead, data: KuaiRecEnvData):
        self.ranking_head = ranking_head
        self.data = data
        self.optimizer = AdamW(
            ranking_head.parameters(), lr=cfg.sft_lr, weight_decay=1e-4
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.sft_epochs * cfg.sft_users_per_epoch,
            eta_min=cfg.sft_lr * 0.1,
        )

    def _sample_candidates(self, uid: int):
        """
        为用户采样候选集（正样本 + 负样本）
        返回: (candidates, labels) 或 (None, None) 如果历史不足
        """
        pos_items = self.data.user_histories.get(uid, [])
        if len(pos_items) < 2:
            return None, None

        # 正样本：取历史最近的 n_pos 条
        n_pos = min(cfg.sft_n_pos, len(pos_items))
        pos_sample = list(
            np.random.choice(pos_items[-50:], n_pos, replace=False)
        )
        pos_set = set(pos_sample)

        # 负样本：随机采，排除正样本
        neg_pool = [i for i in np.random.choice(
            self.data.n_items, cfg.sft_n_neg * 3, replace=False
        ) if i not in pos_set]
        neg_sample = neg_pool[: cfg.sft_n_neg]

        candidates = pos_sample + neg_sample
        np.random.shuffle(candidates)

        labels = [1.0 if iid in pos_set else 0.0 for iid in candidates]
        return candidates, labels

    def _build_features(self, uid: int, candidates: List[int]) -> torch.Tensor:
        """
        构建 (K, INPUT_DIM) 特征张量
        SFT 阶段没有 session 上下文，instruction_sim=0, fatigue=0, step=0
        """
        K = len(candidates)

        # user embedding（历史 item 均值）
        user_emb = self.data.user_profiles.get(uid)
        if user_emb is None:
            user_emb = np.zeros(cfg.embed_dim, dtype=np.float32)

        # item embeddings
        cand_embs = np.array([
            self.data.item_embeddings[iid]
            if iid < len(self.data.item_embeddings)
            else np.zeros(cfg.embed_dim, dtype=np.float32)
            for iid in candidates
        ], dtype=np.float32)

        u = torch.tensor(np.tile(user_emb, (K, 1)), dtype=torch.float32)
        it = torch.tensor(cand_embs, dtype=torch.float32)
        sim = torch.zeros(K, 1)     # 无指令上下文
        fat = torch.zeros(K, 1)     # 无疲劳状态
        stp = torch.zeros(K, 1)     # 无 session 步数

        return torch.cat([u, it, sim, fat, stp], dim=-1)  # (K, INPUT_DIM)

    def train(self, user_ids: List[int]) -> List[Dict]:
        print(f"\nStarting SFT: {cfg.sft_epochs} epochs, "
              f"{cfg.sft_users_per_epoch} users/epoch, "
              f"pos={cfg.sft_n_pos} neg={cfg.sft_n_neg}")
        logs = []

        for epoch in range(cfg.sft_epochs):
            print(f"\n{'='*50}")
            print(f"SFT Epoch {epoch+1}/{cfg.sft_epochs}")

            np.random.shuffle(user_ids)
            epoch_users = user_ids[: cfg.sft_users_per_epoch]

            total_loss = 0.0
            n_valid = 0

            self.ranking_head.train()
            for uid in tqdm(epoch_users, desc="SFT"):
                candidates, labels = self._sample_candidates(uid)
                if candidates is None:
                    continue

                features = self._build_features(uid, candidates)
                K = len(candidates)

                u = features[:, :cfg.embed_dim].to(cfg.device)
                it = features[:, cfg.embed_dim: cfg.embed_dim * 2].to(cfg.device)
                rest = features[:, cfg.embed_dim * 2:].to(cfg.device)

                scores = self.ranking_head(
                    u, it, rest[:, 0:1], rest[:, 1:2], rest[:, 2:3]
                )

                # Listwise softmax cross-entropy
                label_t = torch.tensor(labels, dtype=torch.float32, device=cfg.device)
                label_t = label_t / (label_t.sum() + 1e-9)  # 归一化为概率分布

                log_probs = F.log_softmax(scores, dim=0)
                loss = -(label_t * log_probs).sum()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ranking_head.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()

                total_loss += loss.item()
                n_valid += 1

            avg_loss = total_loss / max(n_valid, 1)
            log = {"epoch": epoch + 1, "sft_loss": avg_loss}
            logs.append(log)
            print(f"  SFT avg loss: {avg_loss:.4f} ({n_valid} users)")

            torch.save(
                self.ranking_head.state_dict(),
                f"{cfg.output_dir}/ranking_head_sft_epoch{epoch+1}.pt",
            )

        # 保存 SFT 最终权重（GRPO 的起点）
        torch.save(
            self.ranking_head.state_dict(),
            f"{cfg.output_dir}/ranking_head_sft_final.pt",
        )
        with open(f"{cfg.output_dir}/sft_log.json", "w") as f:
            json.dump(logs, f, indent=2)

        print(f"\nSFT done. Checkpoint: {cfg.output_dir}/ranking_head_sft_final.pt")
        return logs
