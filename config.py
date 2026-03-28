import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # ── 路径 ──
    data_dir: str = "."
    output_dir: str = "./output"
    cache_dir: str = "./cache"

    # ── DashScope / Qwen ──
    dashscope_api_key: str = field(default_factory=lambda: os.getenv("DASHSCOPE_API_KEY", ""))
    qwen_model: str = "qwen-plus"
    embed_model: str = "text-embedding-v2"
    embed_dim: in
    t = 1536

    # ── 数据 ──
    watch_ratio_threshold: float = 0.5
    max_history_len: int = 20        # 用户历史序列长度
    n_sim_users: int = 200           # 仿真用户数量
    item_pool_size: int = 10000      # 视频池大小（抽样）

    # ── 环境 / MDP ──
    max_session_steps: int = 20      # 每个session最多推荐轮数
    rec_list_size: int = 10          # 每轮推荐数量
    recall_topk: int = 50            # FAISS召回候选数

    # ── 奖励 ──
    reward_click: float = 0.5
    reward_skip: float = 0.0
    reward_leave: float = -1.0
    reward_session_step: float = 0.1     # 每轮留存奖励
    reward_instruction_follow: float = 1.0
    reward_diversity_penalty: float = -0.3
    diversity_sim_threshold: float = 0.8  # 相似度超过此值触发多样性惩罚

    # ── User Simulator ──
    fatigue_decay: float = 0.9           # 疲劳衰减系数（每轮）
    fatigue_threshold: float = 0.7       # 超过此值触发"想换口味"
    leave_prob_base: float = 0.05        # 基础离开概率
    instruction_sim_threshold: float = 0.7  # 指令跟随判定阈值

    # ── Rec Agent (MLP ranking head) ──
    mlp_input_dim: int = 1536 * 2 + 64  # user_emb + item_emb + context
    mlp_hidden_dim: int = 256
    mlp_output_dim: int = 1

    # ── GRPO ──
    grpo_lr: float = 3e-4
    grpo_epochs: int = 5
    grpo_batch_size: int = 32
    grpo_group_size: int = 8            # GRPO每组采样数
    grpo_epsilon: float = 0.2           # clip epsilon
    grpo_gamma: float = 0.99            # 折扣因子
    grpo_lam: float = 0.95              # GAE lambda
    n_rollout_episodes: int = 50        # 每epoch收集的episode数

    # ── 评估 ──
    eval_episodes: int = 100
    ndcg_k: int = 10

    # ── 设备 ──
    device: str = "cpu"


cfg = Config()

for d in [cfg.output_dir, cfg.cache_dir]:
    os.makedirs(d, exist_ok=True)
