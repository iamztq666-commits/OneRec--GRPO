import os
import torch
from dataclasses import dataclass, field


@dataclass
class Config:
    # ── 路径 ──
    data_dir: str = "."
    output_dir: str = "./output"
    cache_dir: str = "./cache"

    # ── DashScope / Qwen API（embedding 仍走 API，已有本地缓存）──
    dashscope_api_key: str = field(default_factory=lambda: os.getenv("DASHSCOPE_API_KEY", ""))
    qwen_model: str = "qwen-plus"           # API fallback 时使用
    embed_model: str = "text-embedding-v2"
    embed_dim: int = 1536

    # ── 本地 LLM（用户仿真器推理，替换 API 调用）──
    use_local_llm: bool = True              # True=本地 Qwen3-8B，False=DashScope API
    local_llm_model: str = "Qwen/Qwen3-8B" # HuggingFace model id 或本地路径
    local_llm_dtype: str = "bfloat16"      # bfloat16(~16GB) 或 int4(~5GB，需 bitsandbytes)

    # ── 数据 ──
    watch_ratio_threshold: float = 0.5
    max_history_len: int = 20
    n_sim_users: int = 200
    item_pool_size: int = 10000

    # ── 环境 / MDP ──
    max_session_steps: int = 20
    rec_list_size: int = 10
    recall_topk: int = 50

    # ── 奖励 ──
    reward_click: float = 0.5
    reward_skip: float = 0.0
    reward_leave: float = -1.0
    reward_session_step: float = 0.1
    reward_instruction_follow: float = 1.0
    reward_diversity_penalty: float = -0.3
    diversity_sim_threshold: float = 0.8

    # ── User Simulator ──
    fatigue_decay: float = 0.9
    fatigue_threshold: float = 0.7
    leave_prob_base: float = 0.05
    instruction_sim_threshold: float = 0.7

    # ── Rec Agent (MLP ranking head) ──
    mlp_input_dim: int = 1536 * 2 + 64
    mlp_hidden_dim: int = 256
    mlp_output_dim: int = 1

    # ── SFT（监督预训练，GRPO 之前的热启动，对应 OneRec Pre-training 阶段）──
    sft_epochs: int = 2               # epoch 数（少量即可，目的是热启动）
    sft_lr: float = 1e-3              # SFT 学习率（比 GRPO 大，收敛快）
    sft_users_per_epoch: int = 2000   # 每 epoch 使用的用户数
    sft_n_pos: int = 5                # 每用户正样本数
    sft_n_neg: int = 20               # 每用户负样本数

    # ── Session-level GRPO（对齐 OneRec session-wise 生成思路）──
    #   OneRec 核心：对每个用户生成 G 条完整 session，
    #   用 session 级累计奖励做 group 归一化优势，而不是 step-level
    grpo_epochs: int = 5
    grpo_lr: float = 3e-4
    grpo_batch_size: int = 32
    grpo_inner_epochs: int = 3         # 每次 rollout 后的内循环更新轮数
    grpo_group_size: int = 8           # 每个用户生成的候选 session 数 G
    grpo_temperature: float = 0.5      # 随机 session 的采样温度
    grpo_epsilon: float = 0.2          # PPO-clip epsilon
    n_rollout_episodes: int = 50       # 每 epoch 采样的用户数

    # ── 评估 ──
    eval_episodes: int = 100
    ndcg_k: int = 10

    # ── 设备（auto = 自动检测 CUDA / MPS / CPU）──
    device: str = "auto"

    def __post_init__(self):
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        print(f"[Config] device = {self.device}")
        if self.device == "cuda":
            print(f"[Config] GPU: {torch.cuda.get_device_name(0)}, "
                  f"Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")


cfg = Config()

for d in [cfg.output_dir, cfg.cache_dir]:
    os.makedirs(d, exist_ok=True)
