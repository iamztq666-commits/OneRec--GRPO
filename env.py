"""
RecoWorld MDP环境
State  : (user_id, history_iids, mindset_vec, session_step, last_instruction)
Action : Top-K推荐列表 (ranked item indices)
Reward : watch_ratio + 留存 + 指令跟随 + 多样性
"""
import pickle
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from config import cfg


# ─────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────
@dataclass
class MDPState:
    user_id: int
    history_iids: List[int]          # 历史交互item id列表
    mindset: np.ndarray              # 用户当前兴趣向量 (embed_dim,)
    fatigue: float                   # 疲劳度 [0, 1]
    session_step: int                # 当前session步数
    last_instruction: str            # 上一轮用户反思指令 (空串=无)
    done: bool = False

@dataclass
class StepResult:
    next_state: MDPState
    reward: float
    done: bool
    info: Dict                       # 详细奖励分量、用户行为等


# ─────────────────────────────────────────────
# KuaiRec 数据加载
# ─────────────────────────────────────────────
class KuaiRecEnvData:
    def __init__(self):
        self.interactions: pd.DataFrame = None
        self.item_meta: pd.DataFrame = None
        self.item_embeddings: np.ndarray = None  # (n_items, embed_dim)
        self.user2id: Dict = {}
        self.item2id: Dict = {}
        self.id2item: Dict = {}
        self.id2text: Dict = {}      # iid -> text description
        self.n_users: int = 0
        self.n_items: int = 0
        self.user_histories: Dict[int, List[int]] = {}   # uid -> sorted iid list
        self.user_profiles: Dict[int, np.ndarray] = {}  # uid -> mean embedding

    def load(self) -> "KuaiRecEnvData":
        print("Loading KuaiRec 2.0...")
        inter = pd.read_csv(f"{cfg.data_dir}/big_matrix.csv")
        item_meta = pd.read_csv(f"{cfg.data_dir}/item_categories.csv")

        try:
            daily = pd.read_csv(f"{cfg.data_dir}/item_daily_features.csv")
            tags = daily[["video_id", "video_tag_name"]].drop_duplicates("video_id").rename(columns={"video_tag_name": "video_tag_list"})
            item_meta = item_meta.merge(tags, on="video_id", how="left")
        except FileNotFoundError:
            item_meta["video_tag_list"] = ""

        # 抽样item池
        top_items = inter["video_id"].value_counts().head(cfg.item_pool_size).index
        inter = inter[inter["video_id"].isin(top_items)]
        item_meta = item_meta[item_meta["video_id"].isin(top_items)]

        users = inter["user_id"].unique()
        items = inter["video_id"].unique()
        self.user2id = {u: i for i, u in enumerate(users)}
        self.item2id = {v: i for i, v in enumerate(items)}
        self.id2item = {i: v for v, i in self.item2id.items()}

        inter["uid"] = inter["user_id"].map(self.user2id)
        inter["iid"] = inter["video_id"].map(self.item2id)
        inter["label"] = (inter["watch_ratio"] >= cfg.watch_ratio_threshold).astype(int)
        inter = inter.sort_values("timestamp").reset_index(drop=True)

        self.interactions = inter
        self.item_meta = item_meta
        self.n_users = len(users)
        self.n_items = len(items)

        # 构建item文本
        for _, row in item_meta.iterrows():
            vid = row["video_id"]
            if vid in self.item2id:
                iid = self.item2id[vid]
                tags_str = str(row.get("video_tag_list", "")).replace(",", " ")
                feat_str = str(row.get("feat", "")).replace(",", " ")
                self.id2text[iid] = f"{tags_str} {feat_str}".strip() or f"video_{vid}"

        # 构建用户历史序列
        pos = inter[inter["label"] == 1]
        for uid, grp in pos.groupby("uid"):
            self.user_histories[uid] = grp.sort_values("timestamp")["iid"].tolist()

        print(f"  Users:{self.n_users:,} Items:{self.n_items:,} Interactions:{len(inter):,}")
        return self

    def get_item_text(self, iid: int) -> str:
        return self.id2text.get(iid, f"video_{iid}")

    def load_embeddings(self, path: str):
        """加载预计算的item embeddings"""
        data = np.load(path)
        self.item_embeddings = data  # (n_items, embed_dim)
        # 计算用户profile = 历史item embedding均值
        for uid, hist in self.user_histories.items():
            if hist and self.item_embeddings is not None:
                embs = [self.item_embeddings[iid] for iid in hist[-20:]
                        if iid < len(self.item_embeddings)]
                if embs:
                    self.user_profiles[uid] = np.mean(embs, axis=0)
        print(f"  Loaded embeddings, user profiles: {len(self.user_profiles):,}")


# ─────────────────────────────────────────────
# MDP 环境
# ─────────────────────────────────────────────
class RecoWorldEnv:
    def __init__(self, data: KuaiRecEnvData):
        self.data = data
        self._rng = np.random.default_rng(42)
        # 预计算watch_ratio查找表: (uid, iid) -> watch_ratio
        self._wr_table: Dict[Tuple[int,int], float] = {}
        self._build_wr_table()

    def _build_wr_table(self):
        df = self.data.interactions[["uid", "iid", "watch_ratio"]].drop_duplicates(
            subset=["uid", "iid"], keep="last"
        )
        self._wr_table = dict(zip(
            zip(df["uid"].values.astype(int), df["iid"].values.astype(int)),
            df["watch_ratio"].values.astype(float)
        ))

    def reset(self, uid: int) -> MDPState:
        """初始化一个session"""
        hist = self.data.user_histories.get(uid, [])
        # 用训练集前80%作为初始历史
        cutoff = int(len(hist) * 0.8)
        init_hist = hist[:cutoff][-cfg.max_history_len:]

        profile = self.data.user_profiles.get(uid)
        mindset = profile.copy() if profile is not None else np.zeros(cfg.embed_dim)

        return MDPState(
            user_id=uid,
            history_iids=init_hist,
            mindset=mindset,
            fatigue=0.0,
            session_step=0,
            last_instruction="",
            done=False,
        )

    def step(self, state: MDPState, rec_list: List[int],
             user_actions: List[str], instruction: str) -> StepResult:
        """
        执行一步MDP
        rec_list     : 推荐的iid列表 (长度=rec_list_size)
        user_actions : 每个item对应的行为 ["click","skip","leave",...]
        instruction  : 本轮用户发出的反思指令 (可为空)
        """
        uid = state.user_id
        total_reward = 0.0
        info = {"click": 0, "skip": 0, "leave": False,
                "watch_ratios": [], "instruction_followed": False}

        # ── 即时奖励 ──
        new_history = state.history_iids.copy()
        for iid, action in zip(rec_list, user_actions):
            wr = self._wr_table.get((uid, iid), 0.0)
            info["watch_ratios"].append(wr)

            if action == "click":
                total_reward += cfg.reward_click + wr * 0.5
                info["click"] += 1
                new_history.append(iid)
            elif action == "skip":
                total_reward += cfg.reward_skip
                info["skip"] += 1
            elif action == "leave":
                total_reward += cfg.reward_leave
                info["leave"] = True
                break

        # ── 留存奖励 ──
        total_reward += cfg.reward_session_step

        # ── 多样性惩罚 ──
        diversity_penalty = self._compute_diversity_penalty(rec_list)
        total_reward += diversity_penalty
        info["diversity_penalty"] = diversity_penalty

        # ── 指令跟随奖励 ──
        inst_reward = 0.0
        if state.last_instruction and self.data.item_embeddings is not None:
            inst_reward = self._compute_instruction_reward(
                state.last_instruction, rec_list)
            total_reward += inst_reward
            info["instruction_followed"] = inst_reward > 0.5
        info["instruction_reward"] = inst_reward

        # ── 更新状态 ──
        new_fatigue = min(1.0, state.fatigue * cfg.fatigue_decay + 0.1 * info["click"])
        new_mindset = self._update_mindset(state.mindset, rec_list, user_actions)
        done = info["leave"] or state.session_step + 1 >= cfg.max_session_steps

        next_state = MDPState(
            user_id=uid,
            history_iids=new_history[-cfg.max_history_len:],
            mindset=new_mindset,
            fatigue=new_fatigue,
            session_step=state.session_step + 1,
            last_instruction=instruction,
            done=done,
        )
        return StepResult(next_state=next_state, reward=total_reward,
                          done=done, info=info)

    def _compute_diversity_penalty(self, rec_list: List[int]) -> float:
        if self.data.item_embeddings is None or len(rec_list) < 2:
            return 0.0
        embs = np.array([self.data.item_embeddings[iid]
                         for iid in rec_list if iid < len(self.data.item_embeddings)])
        if len(embs) < 2:
            return 0.0
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        normed = embs / (norms + 1e-9)
        sim_matrix = normed @ normed.T
        upper = sim_matrix[np.triu_indices(len(normed), k=1)]
        if np.mean(upper) > cfg.diversity_sim_threshold:
            return cfg.reward_diversity_penalty
        return 0.0

    def _compute_instruction_reward(self, instruction: str,
                                    rec_list: List[int]) -> float:
        """
        简化版指令跟随度：
        如果指令包含类别关键词，检查推荐列表中相关item占比
        完整版应用embedding相似度，这里用关键词匹配作快速近似
        """
        if not instruction:
            return 0.0
        instruction_lower = instruction.lower()
        hit = 0
        for iid in rec_list:
            text = self.data.id2text.get(iid, "").lower()
            # 简单关键词重叠
            instr_words = set(instruction_lower.split())
            text_words = set(text.split())
            if len(instr_words & text_words) >= 2:
                hit += 1
        ratio = hit / max(len(rec_list), 1)
        return cfg.reward_instruction_follow * ratio

    def _update_mindset(self, mindset: np.ndarray,
                        rec_list: List[int], actions: List[str]) -> np.ndarray:
        """点击的item embedding加权平均更新mindset"""
        if self.data.item_embeddings is None:
            return mindset
        clicked = [iid for iid, a in zip(rec_list, actions)
                   if a == "click" and iid < len(self.data.item_embeddings)]
        if not clicked:
            return mindset * 0.95  # 无点击，兴趣向量衰减
        click_embs = np.mean([self.data.item_embeddings[iid] for iid in clicked], axis=0)
        return mindset * 0.7 + click_embs * 0.3

    def get_item_text(self, iid: int) -> str:
        return self.data.id2text.get(iid, f"video_{iid}")

    def sample_users(self, n: int) -> List[int]:
        """采样有足够历史的用户"""
        valid = [uid for uid, hist in self.data.user_histories.items()
                 if len(hist) >= 10]
        return list(self._rng.choice(valid, size=min(n, len(valid)), replace=False))
