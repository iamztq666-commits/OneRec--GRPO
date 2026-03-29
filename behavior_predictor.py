"""
行为预测分类器
输入: 用户 mindset embedding + item embedding
输出: click / skip / leave 概率

训练数据: KuaiRec 历史交互
  watch_ratio >= 0.5  → click
  0.1 <= watch_ratio < 0.5 → skip
  watch_ratio < 0.1   → leave

训练完后集成到 UserSimulator，替代大部分 LLM 调用
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

from config import cfg


LABEL_CLICK = 0
LABEL_SKIP  = 1
LABEL_LEAVE = 2


# ─────────────────────────────────────────────
# 模型
# ─────────────────────────────────────────────
class BehaviorPredictor(nn.Module):
    def __init__(self, embed_dim: int = None, hidden_dim: int = 256):
        super().__init__()
        embed_dim = embed_dim or cfg.embed_dim
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
        )

    def forward(self, user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([user_emb, item_emb], dim=-1)
        return self.net(x)

    def predict_probs(self, user_emb: np.ndarray, item_emb: np.ndarray) -> np.ndarray:
        """返回 [p_click, p_skip, p_leave]"""
        with torch.no_grad():
            u = torch.tensor(user_emb, dtype=torch.float32).unsqueeze(0).to(next(self.parameters()).device)
            i = torch.tensor(item_emb, dtype=torch.float32).unsqueeze(0).to(next(self.parameters()).device)
            logits = self.forward(u, i)
            return F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    def predict_action(self, user_emb: np.ndarray, item_emb: np.ndarray,
                       fatigue: float = 0.0) -> str:
        """返回 'click' / 'skip' / 'leave'，fatigue 越高越容易 leave"""
        probs = self.predict_probs(user_emb, item_emb)
        # fatigue 影响 leave 概率
        probs[LABEL_LEAVE] = probs[LABEL_LEAVE] + fatigue * 0.2
        probs = probs / probs.sum()
        idx = int(np.random.choice(3, p=probs))
        return ["click", "skip", "leave"][idx]


# ─────────────────────────────────────────────
# 训练数据集
# ─────────────────────────────────────────────
class BehaviorDataset(Dataset):
    def __init__(self, user_embs: np.ndarray, item_embs: np.ndarray, labels: np.ndarray):
        self.user_embs = torch.tensor(user_embs, dtype=torch.float32)
        self.item_embs = torch.tensor(item_embs, dtype=torch.float32)
        self.labels    = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.user_embs[idx], self.item_embs[idx], self.labels[idx]


def build_training_data(data, item_embeddings: np.ndarray,
                        max_samples: int = 200_000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从 KuaiRec 交互记录构建训练数据
    user_emb = 该交互之前历史点击 item 的 embedding 均值（模拟 mindset）
    """
    print("[BehaviorPredictor] Building training data...")
    df = data.interactions.copy()
    df = df[df["iid"] < len(item_embeddings)].dropna(subset=["watch_ratio"])
    df = df.sort_values(["uid", "timestamp"]).reset_index(drop=True)

    user_emb_list, item_emb_list, label_list = [], [], []

    for uid, group in df.groupby("uid"):
        iids = group["iid"].tolist()
        wrs  = group["watch_ratio"].tolist()

        history_embs = []
        for step, (iid, wr) in enumerate(zip(iids, wrs)):
            # 用历史均值作为用户 mindset
            if history_embs:
                user_emb = np.mean(history_embs[-20:], axis=0)
            else:
                user_emb = item_embeddings[iid]

            item_emb = item_embeddings[iid]

            # 标签
            if wr >= 0.5:
                label = LABEL_CLICK
                history_embs.append(item_emb)
            elif wr >= 0.1:
                label = LABEL_SKIP
            else:
                label = LABEL_LEAVE

            user_emb_list.append(user_emb.astype(np.float32))
            item_emb_list.append(item_emb.astype(np.float32))
            label_list.append(label)

            if len(label_list) >= max_samples:
                break
        if len(label_list) >= max_samples:
            break

    print(f"  Samples: {len(label_list):,}")
    counts = np.bincount(label_list, minlength=3)
    print(f"  click={counts[0]:,}  skip={counts[1]:,}  leave={counts[2]:,}")

    return (np.array(user_emb_list), np.array(item_emb_list), np.array(label_list))


# ─────────────────────────────────────────────
# 训练
# ─────────────────────────────────────────────
def train_behavior_predictor(data, item_embeddings: np.ndarray,
                              epochs: int = 5, batch_size: int = 2048,
                              lr: float = 1e-3) -> BehaviorPredictor:
    ckpt = f"{cfg.output_dir}/behavior_predictor.pt"
    model = BehaviorPredictor().to(cfg.device)

    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=cfg.device))
        print(f"[BehaviorPredictor] Loaded checkpoint: {ckpt}")
        return model

    user_embs, item_embs, labels = build_training_data(data, item_embeddings, max_samples=50_000)

    # 80/20 split
    n = len(labels)
    idx = np.random.permutation(n)
    train_idx, val_idx = idx[:int(n * 0.8)], idx[int(n * 0.8):]

    train_ds = BehaviorDataset(user_embs[train_idx], item_embs[train_idx], labels[train_idx])
    val_ds   = BehaviorDataset(user_embs[val_idx],   item_embs[val_idx],   labels[val_idx])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size * 2, shuffle=False, num_workers=0)

    # 类别不均衡：用 class weight
    counts = np.bincount(labels, minlength=3).astype(float)
    weights = torch.tensor(1.0 / (counts + 1), dtype=torch.float32).to(cfg.device)
    weights = weights / weights.sum() * 3

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for u, i, y in train_dl:
            u, i, y = u.to(cfg.device), i.to(cfg.device), y.to(cfg.device)
            logits = model(u, i)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)

        # 验证
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for u, i, y in val_dl:
                u, i, y = u.to(cfg.device), i.to(cfg.device), y.to(cfg.device)
                val_correct += (model(u, i).argmax(1) == y).sum().item()
                val_total += len(y)
        val_acc = val_correct / val_total

        print(f"  Epoch {epoch}/{epochs} | loss={total_loss/total:.4f} "
              f"| train_acc={correct/total:.3f} | val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=cfg.device))
    print(f"[BehaviorPredictor] Best val_acc={best_val_acc:.3f}, saved to {ckpt}")
    return model
