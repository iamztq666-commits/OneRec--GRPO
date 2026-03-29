"""
Session 数据准备

从 KuaiRec 真实日志切割 session，构建训练对：
  输入：用户历史 N 条交互的 SID token 序列
  输出：目标 session 的 SID token 序列

Session 切割规则：
  同一用户相邻交互时间间隔 > SESSION_GAP_MIN 分钟 → 新 session
  Session 内至少有 MIN_SESSION_LEN 条交互
  高质量 session：avg watch_ratio >= QUALITY_THRESHOLD

训练样本格式（文本）：
  <|hist_begin|>
  <|sid_begin|><s1_3><s2_12><s3_5><|sid_end|>
  <|sid_begin|><s1_7><s2_4><s3_11><|sid_end|>
  ...
  <|hist_end|>
  请生成下一次推荐 session：
  <|session_begin|>
  <|sid_begin|><s1_2><s2_9><s3_3><|sid_end|>
  ...
  <|session_end|>
"""
import _path  # noqa

import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from config import cfg
from semantic_ids import sid_to_tokens, build_semantic_ids

SESSION_GAP_MIN = 30        # 分钟间隔判定新 session
MIN_SESSION_LEN = 3         # session 最少 item 数
MAX_SESSION_LEN = 10        # session 最多 item 数（截断）
QUALITY_THRESHOLD = 0.4     # 高质量 session 的平均 watch_ratio
MAX_HIST_LEN = 20           # 历史序列最大长度
SESSION_DATA_CACHE = f"{cfg.cache_dir}/session_data.pkl"


@dataclass
class SessionSample:
    user_id: int
    session_id: str                  # f"{user_id}_{session_idx}"
    history_iids: List[int]          # 历史 item ids
    target_iids: List[int]           # 目标 session item ids
    target_watch_ratios: List[float] # 目标 session 每个 item 的 watch_ratio
    avg_quality: float               # 目标 session 平均 watch_ratio


def cut_sessions(df: pd.DataFrame) -> Dict[int, List[List[dict]]]:
    """
    按时间间隔切割用户 session。
    返回: {uid: [[{iid, wr, ts}, ...], ...]}  # 每个用户的 session 列表
    """
    print("[SessionData] Cutting sessions by time gap...")
    df = df.sort_values(["uid", "timestamp"]).reset_index(drop=True)
    user_sessions: Dict[int, List[List[dict]]] = {}

    for uid, group in df.groupby("uid"):
        rows = group[["iid", "watch_ratio", "timestamp"]].to_dict("records")
        sessions = []
        current = [rows[0]]

        for r in rows[1:]:
            gap_min = (r["timestamp"] - current[-1]["timestamp"]) / 60.0
            if gap_min > SESSION_GAP_MIN:
                sessions.append(current)
                current = [r]
            else:
                current.append(r)
        sessions.append(current)

        # 过滤太短的 session
        sessions = [s for s in sessions if len(s) >= MIN_SESSION_LEN]
        if sessions:
            user_sessions[uid] = sessions

    total = sum(len(v) for v in user_sessions.values())
    print(f"  Users with sessions: {len(user_sessions):,}, Total sessions: {total:,}")
    return user_sessions


def build_train_pairs(
    user_sessions: Dict[int, List[List[dict]]],
    iid2sid: Dict,
    min_history: int = 5,
) -> List[SessionSample]:
    """
    构建 (历史, 目标session) 训练对。
    对每个用户，用前 i 个 session 作为历史，第 i+1 个 session 作为目标。
    只保留高质量目标 session（avg_watch_ratio >= QUALITY_THRESHOLD）。
    """
    print("[SessionData] Building training pairs...")
    samples = []

    for uid, sessions in user_sessions.items():
        # 累积历史
        history_iids = []

        for sess_idx, session in enumerate(sessions):
            target_iids = [r["iid"] for r in session[:MAX_SESSION_LEN]
                           if r["iid"] in iid2sid]
            target_wrs = [r["watch_ratio"] for r in session[:MAX_SESSION_LEN]
                          if r["iid"] in iid2sid]

            if len(target_iids) < MIN_SESSION_LEN:
                history_iids.extend([r["iid"] for r in session])
                continue

            avg_wr = float(np.mean(target_wrs))

            if len(history_iids) >= min_history and avg_wr >= QUALITY_THRESHOLD:
                hist = [iid for iid in history_iids[-MAX_HIST_LEN:]
                        if iid in iid2sid]
                if hist:
                    samples.append(SessionSample(
                        user_id=uid,
                        session_id=f"{uid}_{sess_idx}",
                        history_iids=hist,
                        target_iids=target_iids,
                        target_watch_ratios=target_wrs,
                        avg_quality=avg_wr,
                    ))

            # 更新历史
            history_iids.extend([r["iid"] for r in session])

    # 按质量排序，优先高质量 session
    samples.sort(key=lambda x: x.avg_quality, reverse=True)
    print(f"  Training pairs: {len(samples):,} "
          f"(avg quality: {np.mean([s.avg_quality for s in samples]):.3f})")
    return samples


def sample_to_text(sample: SessionSample, iid2sid: Dict) -> Tuple[str, str]:
    """
    把 SessionSample 转成 (input_text, target_text) 字符串对。
    """
    # 历史序列
    hist_tokens = []
    for iid in sample.history_iids:
        if iid in iid2sid:
            hist_tokens.append(sid_to_tokens(iid2sid[iid]))

    input_text = (
        "<|hist_begin|>\n" +
        "\n".join(hist_tokens) +
        "\n<|hist_end|>\n请生成下一次推荐 session："
    )

    # 目标 session
    target_tokens = []
    for iid in sample.target_iids:
        if iid in iid2sid:
            target_tokens.append(sid_to_tokens(iid2sid[iid]))

    target_text = (
        "<|session_begin|>\n" +
        "\n".join(target_tokens) +
        "\n<|session_end|>"
    )

    return input_text, target_text


def prepare_session_data(data, item_embeddings: np.ndarray,
                          force_rebuild: bool = False) -> Tuple[List[SessionSample], Dict, Dict, List[str]]:
    """
    完整数据准备流程：
    1. 构建 semantic IDs
    2. 切割真实 session
    3. 构建训练对
    返回: (samples, iid2sid, sid2iid, vocab)
    """
    # 1. Semantic IDs
    iid2sid, sid2iid, vocab = build_semantic_ids(item_embeddings, force_rebuild)

    # 2. 切割 session
    if os.path.exists(SESSION_DATA_CACHE) and not force_rebuild:
        with open(SESSION_DATA_CACHE, "rb") as f:
            samples = pickle.load(f)
        print(f"[SessionData] Loaded {len(samples):,} pairs from cache")
        return samples, iid2sid, sid2iid, vocab

    user_sessions = cut_sessions(data.interactions)
    samples = build_train_pairs(user_sessions, iid2sid)

    with open(SESSION_DATA_CACHE, "wb") as f:
        pickle.dump(samples, f)

    return samples, iid2sid, sid2iid, vocab


def save_as_jsonl(samples: List[SessionSample], iid2sid: Dict,
                   path: str, max_samples: int = 50_000):
    """保存为 JSONL 格式，供 SFT 训练用"""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    count = 0
    with open(path, "w") as f:
        for s in samples[:max_samples]:
            inp, tgt = sample_to_text(s, iid2sid)
            f.write(json.dumps({
                "input": inp,
                "target": tgt,
                "session_id": s.session_id,
                "avg_quality": s.avg_quality,
            }, ensure_ascii=False) + "\n")
            count += 1
    print(f"[SessionData] Saved {count:,} samples to {path}")
