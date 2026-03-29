"""
Rec Agent
架构：
  1. FAISS向量召回 Top-50（固定，不参与训练）
  2. Qwen embedding生成context向量（固定）
  3. MLP ranking head：输入(user_ctx, item_emb, instruction_emb)，输出score
  4. MLP是GRPO优化的目标

指令跟随：将last_instruction编码为embedding，与item embedding计算相似度，
         作为ranking head的额外特征。
"""
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
from typing import List, Optional
from openai import OpenAI

from config import cfg
from env import MDPState, KuaiRecEnvData

client = OpenAI(
    api_key=cfg.dashscope_api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


# ─────────────────────────────────────────────
# Embedding工具
# ─────────────────────────────────────────────
_embed_cache: dict = {}
_embed_cache_path = f"{cfg.cache_dir}/embed_cache.pkl"

def _load_embed_cache():
    global _embed_cache
    if os.path.exists(_embed_cache_path):
        with open(_embed_cache_path, "rb") as f:
            _embed_cache = pickle.load(f)

def _save_embed_cache():
    with open(_embed_cache_path, "wb") as f:
        pickle.dump(_embed_cache, f)

def encode_text(text: str) -> np.ndarray:
    """单条文本embedding，带缓存"""
    if text in _embed_cache:
        return _embed_cache[text]
    resp = client.embeddings.create(model=cfg.embed_model, input=[text[:512]])
    emb = np.array(resp.data[0].embedding, dtype=np.float32)
    _embed_cache[text] = emb
    _save_embed_cache()
    return emb

def encode_texts_batch(texts: List[str], batch_size: int = 25) -> np.ndarray:
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        uncached = [(j, t) for j, t in enumerate(batch) if t not in _embed_cache]
        if uncached:
            resp = client.embeddings.create(
                model=cfg.embed_model,
                input=[t[:512] for _, t in uncached]
            )
            for (j, t), d in zip(uncached, resp.data):
                emb = np.array(d.embedding, dtype=np.float32)
                _embed_cache[t] = emb
            _save_embed_cache()
        results.extend([_embed_cache[t] for t in batch])
    return np.array(results, dtype=np.float32)

_load_embed_cache()


# ─────────────────────────────────────────────
# MLP Ranking Head（GRPO优化目标）
# ─────────────────────────────────────────────
class RankingHead(nn.Module):
    """
    输入: [user_emb(1536) | item_emb(1536) | instruction_sim(1) | fatigue(1) | step(1)]
    输出: score scalar
    """
    INPUT_DIM = cfg.embed_dim * 2 + 3   # 1536+1536+3 = 3075

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.INPUT_DIM, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, user_emb, item_emb, instruction_sim, fatigue, step):
        """
        user_emb      : (B, 1536)
        item_emb      : (B, 1536)
        instruction_sim: (B, 1)
        fatigue       : (B, 1)
        step          : (B, 1)
        """
        feat = torch.cat([user_emb, item_emb, instruction_sim, fatigue, step], dim=-1)
        return self.net(feat).squeeze(-1)  # (B,)

    def score_candidates(
        self,
        user_emb: np.ndarray,      # (embed_dim,)
        item_embs: np.ndarray,     # (K, embed_dim)
        instruction_emb: Optional[np.ndarray],  # (embed_dim,) or None
        fatigue: float,
        step: int,
        device: str,
    ) -> np.ndarray:
        """批量对候选打分，返回scores (K,)"""
        K = len(item_embs)
        u = torch.tensor(np.tile(user_emb, (K, 1)), dtype=torch.float32).to(device)
        it = torch.tensor(item_embs, dtype=torch.float32).to(device)

        if instruction_emb is not None:
            instr = torch.tensor(instruction_emb, dtype=torch.float32).to(device)
            instr_norm = F.normalize(instr.unsqueeze(0), dim=-1)
            item_norm = F.normalize(it, dim=-1)
            sim = (item_norm @ instr_norm.T).squeeze(-1).unsqueeze(-1)  # (K,1)
        else:
            sim = torch.zeros(K, 1).to(device)

        fat = torch.full((K, 1), fatigue, dtype=torch.float32).to(device)
        stp = torch.full((K, 1), step / cfg.max_session_steps, dtype=torch.float32).to(device)

        self.eval()
        with torch.no_grad():
            scores = self(u, it, sim, fat, stp).cpu().numpy()
        return scores


# ─────────────────────────────────────────────
# Transformer Ranking Head（Listwise，对齐 OneRec）
# ─────────────────────────────────────────────
class TransformerRankingHead(nn.Module):
    """
    Listwise Transformer 排序头

    与 MLP（pointwise）的区别：
      MLP：每个候选 item 独立打分，互相看不见
      Transformer：所有候选 item 一起过 self-attention，
                   能捕捉 item-item 交互（多样性、互补性）
                   更接近 OneRec session-wise 的整体建模思路

    输入格式与 RankingHead 完全兼容（drop-in 替换）：
      user_emb:        (K, 1536) - 同一用户平铺 K 次
      item_emb:        (K, 1536)
      instruction_sim: (K, 1)
      fatigue:         (K, 1)
      step:            (K, 1)
    输出: scores (K,)

    参数量：约 5M（vs MLP 的 1.6M）
    """
    D_MODEL = 256
    N_HEAD = 8
    N_LAYERS = 3
    DIM_FF = 512

    def __init__(self):
        super().__init__()
        d = self.D_MODEL

        # 输入投影
        self.user_proj = nn.Sequential(
            nn.Linear(cfg.embed_dim, d),
            nn.LayerNorm(d),
        )
        self.item_proj = nn.Sequential(
            nn.Linear(cfg.embed_dim, d),
            nn.LayerNorm(d),
        )
        # context（instruction_sim + fatigue + step）融入 item token
        self.ctx_proj = nn.Linear(3, d)

        # Transformer encoder — Pre-LN，训练更稳定
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=self.N_HEAD,
            dim_feedforward=self.DIM_FF,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.N_LAYERS)

        # 打分头
        self.score_head = nn.Linear(d, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, user_emb, item_emb, instruction_sim, fatigue, step):
        """
        1. user embedding → CLS token（给整个候选序列提供用户上下文）
        2. item embeddings + context → K 个 item token
        3. [CLS, item_1, ..., item_K] → Transformer → 每个 item 输出打分
        """
        # user token：同一用户取第一行，投影为 (1, D)
        user_token = self.user_proj(user_emb[0:1])          # (1, D)

        # item tokens：投影 + context 融合
        item_tokens = self.item_proj(item_emb)               # (K, D)
        ctx = torch.cat([instruction_sim, fatigue, step], dim=-1)  # (K, 3)
        item_tokens = item_tokens + self.ctx_proj(ctx)       # (K, D)

        # 拼序列：[CLS, item_1, ..., item_K] → (1, K+1, D)
        seq = torch.cat([user_token, item_tokens], dim=0).unsqueeze(0)

        # Transformer 联合编码（item-item self-attention + user context）
        out = self.transformer(seq)                          # (1, K+1, D)

        # 取 item 部分（跳过 CLS）打分
        item_out = out[0, 1:, :]                             # (K, D)
        scores = self.score_head(item_out).squeeze(-1)       # (K,)
        return scores

    def score_candidates(
        self,
        user_emb: np.ndarray,
        item_embs: np.ndarray,
        instruction_emb: Optional[np.ndarray],
        fatigue: float,
        step: int,
        device: str,
    ) -> np.ndarray:
        """与 RankingHead 接口完全一致"""
        K = len(item_embs)
        u = torch.tensor(np.tile(user_emb, (K, 1)), dtype=torch.float32).to(device)
        it = torch.tensor(item_embs, dtype=torch.float32).to(device)

        if instruction_emb is not None:
            instr = torch.tensor(instruction_emb, dtype=torch.float32).to(device)
            instr_norm = F.normalize(instr.unsqueeze(0), dim=-1)
            item_norm = F.normalize(it, dim=-1)
            sim = (item_norm @ instr_norm.T).squeeze(-1).unsqueeze(-1)
        else:
            sim = torch.zeros(K, 1).to(device)

        fat = torch.full((K, 1), fatigue, dtype=torch.float32).to(device)
        stp = torch.full((K, 1), step / cfg.max_session_steps, dtype=torch.float32).to(device)

        self.eval()
        with torch.no_grad():
            scores = self(u, it, sim, fat, stp).cpu().numpy()
        return scores


# ─────────────────────────────────────────────
# FAISS 召回索引
# ─────────────────────────────────────────────
class FAISSRetriever:
    def __init__(self, item_embeddings: np.ndarray, item_ids: List[int]):
        self.item_ids = item_ids
        norms = np.linalg.norm(item_embeddings, axis=1, keepdims=True)
        normed = item_embeddings / (norms + 1e-9)
        dim = normed.shape[1]
        index = faiss.IndexFlatIP(dim)
        if faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        index.add(normed.astype(np.float32))
        self.index = index
        self.embeddings = normed

    def retrieve(self, query_emb: np.ndarray, topk: int) -> List[int]:
        q = query_emb / (np.linalg.norm(query_emb) + 1e-9)
        q = q.astype(np.float32).reshape(1, -1)
        _, idx = self.index.search(q, topk)
        return [self.item_ids[i] for i in idx[0] if i < len(self.item_ids)]


# ─────────────────────────────────────────────
# Rec Agent
# ─────────────────────────────────────────────
class RecAgent:
    def __init__(self, data: KuaiRecEnvData, ranking_head: RankingHead):
        self.data = data
        self.ranking_head = ranking_head.to(cfg.device)
        self.retriever: Optional[FAISSRetriever] = None
        self._instruction_emb_cache: dict = {}

    def build_retriever(self):
        """构建FAISS索引（需要item embeddings已加载）"""
        assert self.data.item_embeddings is not None
        all_iids = list(range(self.data.n_items))
        self.retriever = FAISSRetriever(self.data.item_embeddings, all_iids)
        print(f"FAISS index built: {self.data.n_items} items")

    def recommend(self, state: MDPState, env=None) -> List[int]:
        """
        完整推荐流程：
        1. 用mindset做FAISS召回 Top-50
        2. MLP ranking head重排，取Top-10
        """
        # ── 1. 召回 ──
        if self.retriever is None or self.data.item_embeddings is None:
            # 无索引时随机召回
            candidates = list(np.random.choice(self.data.n_items,
                                               cfg.recall_topk, replace=False))
        else:
            candidates = self.retriever.retrieve(state.mindset, cfg.recall_topk)

        # 过滤已看过的item
        seen = set(state.history_iids[-50:])
        candidates = [iid for iid in candidates if iid not in seen]
        if not candidates:
            candidates = list(np.random.choice(self.data.n_items,
                                               cfg.rec_list_size, replace=False))

        # ── 2. 获取instruction embedding ──
        instr_emb = None
        if state.last_instruction:
            if state.last_instruction not in self._instruction_emb_cache:
                self._instruction_emb_cache[state.last_instruction] = \
                    encode_text(state.last_instruction)
            instr_emb = self._instruction_emb_cache[state.last_instruction]

        # ── 3. MLP重排 ──
        cand_embs = np.array([
            self.data.item_embeddings[iid]
            if iid < len(self.data.item_embeddings)
            else np.zeros(cfg.embed_dim, dtype=np.float32)
            for iid in candidates
        ], dtype=np.float32)

        scores = self.ranking_head.score_candidates(
            user_emb=state.mindset,
            item_embs=cand_embs,
            instruction_emb=instr_emb,
            fatigue=state.fatigue,
            step=state.session_step,
            device=cfg.device,
        )

        top_idx = np.argsort(scores)[::-1][:cfg.rec_list_size]
        return [candidates[i] for i in top_idx]

    def get_scoring_features(
        self, state: MDPState, candidates: List[int]
    ) -> torch.Tensor:
        """
        为GRPO提供特征张量，shape: (K, INPUT_DIM)
        用于计算log_prob
        """
        K = len(candidates)
        instr_emb = None
        if state.last_instruction and state.last_instruction in self._instruction_emb_cache:
            instr_emb = self._instruction_emb_cache[state.last_instruction]

        cand_embs = np.array([
            self.data.item_embeddings[iid]
            if (self.data.item_embeddings is not None and iid < len(self.data.item_embeddings))
            else np.zeros(cfg.embed_dim, dtype=np.float32)
            for iid in candidates
        ], dtype=np.float32)

        u = torch.tensor(np.tile(state.mindset, (K, 1)), dtype=torch.float32)
        it = torch.tensor(cand_embs, dtype=torch.float32)

        if instr_emb is not None:
            instr_t = torch.tensor(instr_emb, dtype=torch.float32)
            instr_norm = F.normalize(instr_t.unsqueeze(0), dim=-1)
            item_norm = F.normalize(it, dim=-1)
            sim = (item_norm @ instr_norm.T)  # (K,1)
        else:
            sim = torch.zeros(K, 1)

        fat = torch.full((K, 1), state.fatigue)
        stp = torch.full((K, 1), state.session_step / cfg.max_session_steps)

        return torch.cat([u, it, sim, fat, stp], dim=-1)  # (K, INPUT_DIM)
