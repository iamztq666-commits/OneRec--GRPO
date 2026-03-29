"""
Semantic ID 构造（分层 k-means 量化）

每个 item 映射成 3 级 token 串：
  item_123 -> [sid_L1_34, sid_L2_12, sid_L3_56]

训练时包装成 special tokens：
  <s1_34><s2_12><s3_56>

双向映射表：
  item_id -> sid_tuple
  sid_tuple -> item_id
"""
import _path  # noqa

import os
import pickle
import numpy as np
from typing import Dict, Tuple, List

from config import cfg


N_L1 = 64   # 一级聚类数
N_L2 = 32   # 二级聚类数（在 L1 内部）
N_L3 = 16   # 三级聚类数（在 L2 内部）

SID_CACHE = f"{cfg.cache_dir}/semantic_ids.pkl"


def build_semantic_ids(item_embeddings: np.ndarray,
                       force_rebuild: bool = False) -> Tuple[Dict, Dict, List[str]]:
    """
    构造分层语义 ID。

    返回:
      iid2sid: {item_id: (l1, l2, l3)}
      sid2iid: {(l1, l2, l3): item_id}
      vocab:   所有新增 special token 列表，用于扩展 tokenizer
    """
    if os.path.exists(SID_CACHE) and not force_rebuild:
        with open(SID_CACHE, "rb") as f:
            data = pickle.load(f)
        print(f"[SemanticID] Loaded from cache: {len(data['iid2sid'])} items")
        return data["iid2sid"], data["sid2iid"], data["vocab"]

    print(f"[SemanticID] Building hierarchical k-means ({N_L1}×{N_L2}×{N_L3})...")
    from sklearn.cluster import MiniBatchKMeans

    n_items = len(item_embeddings)
    embs = item_embeddings.astype(np.float32)

    # 归一化
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs_normed = embs / (norms + 1e-9)

    # L1 聚类
    print(f"  L1 clustering ({N_L1} clusters)...")
    km1 = MiniBatchKMeans(n_clusters=N_L1, random_state=42, batch_size=1024, n_init=3)
    l1_labels = km1.fit_predict(embs_normed)

    # L2 聚类（在每个 L1 内部）
    print(f"  L2 clustering ({N_L2} clusters per L1)...")
    l2_labels = np.zeros(n_items, dtype=int)
    for l1 in range(N_L1):
        mask = l1_labels == l1
        if mask.sum() < 2:
            continue
        k2 = min(N_L2, mask.sum())
        km2 = MiniBatchKMeans(n_clusters=k2, random_state=42, batch_size=512, n_init=3)
        l2_labels[mask] = km2.fit_predict(embs_normed[mask])

    # L3 聚类（在每个 L1+L2 内部）
    print(f"  L3 clustering ({N_L3} clusters per L2)...")
    l3_labels = np.zeros(n_items, dtype=int)
    for l1 in range(N_L1):
        for l2 in range(N_L2):
            mask = (l1_labels == l1) & (l2_labels == l2)
            if mask.sum() < 2:
                continue
            k3 = min(N_L3, mask.sum())
            km3 = MiniBatchKMeans(n_clusters=k3, random_state=42, batch_size=256, n_init=3)
            l3_labels[mask] = km3.fit_predict(embs_normed[mask])

    # 构建映射表
    iid2sid: Dict[int, Tuple[int, int, int]] = {}
    sid2iid: Dict[Tuple[int, int, int], int] = {}

    for iid in range(n_items):
        sid = (int(l1_labels[iid]), int(l2_labels[iid]), int(l3_labels[iid]))
        iid2sid[iid] = sid
        # 碰撞处理：若 sid 已存在，在 l3 上偏移
        offset = 0
        while sid in sid2iid and sid2iid[sid] != iid:
            offset += 1
            sid = (int(l1_labels[iid]), int(l2_labels[iid]),
                   int(l3_labels[iid]) + offset * N_L3)
        iid2sid[iid] = sid
        sid2iid[sid] = iid

    # 构建 special token 词表
    all_l1 = sorted(set(v[0] for v in iid2sid.values()))
    all_l2 = sorted(set(v[1] for v in iid2sid.values()))
    all_l3 = sorted(set(v[2] for v in iid2sid.values()))
    vocab = (
        [f"<s1_{i}>" for i in all_l1] +
        [f"<s2_{i}>" for i in all_l2] +
        [f"<s3_{i}>" for i in all_l3] +
        ["<|sid_begin|>", "<|sid_end|>",
         "<|session_begin|>", "<|session_end|>",
         "<|hist_begin|>", "<|hist_end|>"]
    )

    with open(SID_CACHE, "wb") as f:
        pickle.dump({"iid2sid": iid2sid, "sid2iid": sid2iid, "vocab": vocab}, f)

    print(f"[SemanticID] Done: {n_items} items, {len(vocab)} special tokens")
    return iid2sid, sid2iid, vocab


def sid_to_tokens(sid: Tuple[int, int, int]) -> str:
    """(l1, l2, l3) -> '<|sid_begin|><s1_34><s2_12><s3_56><|sid_end|>'"""
    return f"<|sid_begin|><s1_{sid[0]}><s2_{sid[1]}><s3_{sid[2]}><|sid_end|>"


def tokens_to_sid(token_str: str) -> Tuple[int, int, int]:
    """从 token 字符串解析回 (l1, l2, l3)"""
    import re
    l1 = int(re.search(r"<s1_(\d+)>", token_str).group(1))
    l2 = int(re.search(r"<s2_(\d+)>", token_str).group(1))
    l3 = int(re.search(r"<s3_(\d+)>", token_str).group(1))
    return (l1, l2, l3)
