"""
用户意图分类器
输入: 用户最近 N 个点击 item 的 embedding 序列
输出: 意图类别（从 KuaiRec item tag 中提取的 Top-K 类别）

作用: 召回阶段作为类别 bias，补充 mindset 向量
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple

from config import cfg


# ─────────────────────────────────────────────
# 模型
# ─────────────────────────────────────────────
class IntentClassifier(nn.Module):
    def __init__(self, embed_dim: int = None, hidden_dim: int = 128, n_classes: int = 20):
        super().__init__()
        embed_dim = embed_dim or cfg.embed_dim
        self.n_classes = n_classes
        # GRU 编码历史序列
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, num_layers=1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        seq: (B, T, embed_dim)
        lengths: (B,) 实际序列长度
        """
        packed = nn.utils.rnn.pack_padded_sequence(
            seq, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        hidden = hidden.squeeze(0)  # (B, hidden_dim)
        return self.classifier(hidden)

    def predict(self, history_embs: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        history_embs: (T, embed_dim) 最近点击序列
        返回: (top_class_idx, probs)
        """
        if len(history_embs) == 0:
            probs = np.ones(self.n_classes) / self.n_classes
            return 0, probs
        with torch.no_grad():
            seq = torch.tensor(history_embs[-20:], dtype=torch.float32).unsqueeze(0)
            seq = seq.to(next(self.parameters()).device)
            length = torch.tensor([seq.shape[1]])
            logits = self.forward(seq, length)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return int(probs.argmax()), probs


# ─────────────────────────────────────────────
# 标签提取：从 item tag 中提取 Top-K 类别
# ─────────────────────────────────────────────
def extract_categories(data, top_k: int = 20) -> Tuple[Dict[int, int], List[str]]:
    """
    从 item 文本中提取高频 tag 作为类别标签
    返回: (iid -> category_id, category_names)
    """
    from collections import Counter
    tag_counter = Counter()
    iid_tags: Dict[int, List[str]] = {}

    for iid, text in data.id2text.items():
        tags = [t.strip() for t in text.split() if len(t.strip()) > 1][:5]
        iid_tags[iid] = tags
        tag_counter.update(tags)

    top_tags = [tag for tag, _ in tag_counter.most_common(top_k)]
    tag2id = {t: i for i, t in enumerate(top_tags)}

    iid2cat: Dict[int, int] = {}
    for iid, tags in iid_tags.items():
        for tag in tags:
            if tag in tag2id:
                iid2cat[iid] = tag2id[tag]
                break
        if iid not in iid2cat:
            iid2cat[iid] = top_k - 1  # 其他类

    print(f"[IntentClassifier] Categories: {top_tags[:10]}...")
    return iid2cat, top_tags


# ─────────────────────────────────────────────
# 训练数据集
# ─────────────────────────────────────────────
class IntentDataset(Dataset):
    def __init__(self, seqs: List[np.ndarray], labels: List[int], max_len: int = 20):
        self.seqs = seqs
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.seqs[idx][-self.max_len:]
        length = len(seq)
        # padding
        if length < self.max_len:
            pad = np.zeros((self.max_len - length, seq.shape[1]), dtype=np.float32)
            seq = np.vstack([seq, pad])
        return (torch.tensor(seq, dtype=torch.float32),
                torch.tensor(length, dtype=torch.long),
                torch.tensor(self.labels[idx], dtype=torch.long))


def build_intent_data(data, item_embeddings: np.ndarray,
                      iid2cat: Dict[int, int],
                      max_samples: int = 100_000):
    """
    构建意图分类训练数据：
    历史序列 → 下一个点击 item 的类别
    """
    seqs, labels = [], []
    for uid, hist in data.user_histories.items():
        hist = [iid for iid in hist if iid < len(item_embeddings)]
        if len(hist) < 3:
            continue
        for t in range(2, len(hist)):
            history_embs = np.array([item_embeddings[iid] for iid in hist[:t]])
            next_cat = iid2cat.get(hist[t], len(iid2cat) - 1)
            seqs.append(history_embs.astype(np.float32))
            labels.append(next_cat)
            if len(labels) >= max_samples:
                break
        if len(labels) >= max_samples:
            break
    print(f"[IntentClassifier] Training samples: {len(labels):,}")
    return seqs, labels


# ─────────────────────────────────────────────
# 训练
# ─────────────────────────────────────────────
def train_intent_classifier(data, item_embeddings: np.ndarray,
                             n_classes: int = 20, epochs: int = 5,
                             batch_size: int = 512, lr: float = 1e-3) -> Tuple["IntentClassifier", List[str]]:
    ckpt = f"{cfg.output_dir}/intent_classifier.pt"
    iid2cat, category_names = extract_categories(data, top_k=n_classes)
    model = IntentClassifier(n_classes=n_classes).to(cfg.device)

    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=cfg.device))
        print(f"[IntentClassifier] Loaded checkpoint: {ckpt}")
        return model, category_names

    seqs, labels = build_intent_data(data, item_embeddings, iid2cat, max_samples=5_000)

    n = len(labels)
    idx = np.random.permutation(n)
    split = int(n * 0.8)
    train_idx, val_idx = idx[:split], idx[split:]

    train_ds = IntentDataset([seqs[i] for i in train_idx], [labels[i] for i in train_idx])
    val_ds   = IntentDataset([seqs[i] for i in val_idx],   [labels[i] for i in val_idx])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for seq, length, y in train_dl:
            seq, length, y = seq.to(cfg.device), length, y.to(cfg.device)
            logits = model(seq, length)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for seq, length, y in val_dl:
                seq, length, y = seq.to(cfg.device), length, y.to(cfg.device)
                val_correct += (model(seq, length).argmax(1) == y).sum().item()
                val_total += len(y)
        val_acc = val_correct / val_total

        print(f"  Epoch {epoch}/{epochs} | loss={total_loss/total:.4f} "
              f"| train_acc={correct/total:.3f} | val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=cfg.device))
    print(f"[IntentClassifier] Best val_acc={best_val_acc:.3f}")
    return model, category_names
