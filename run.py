"""
主入口
流程：
  1. 数据加载
  2. Item embedding 预计算（带缓存）
  3. 构建 FAISS 索引
  4. Session-level GRPO 训练（对齐 OneRec）
  5. 消融评估
  6. 打印简历描述

云端 GPU 运行示例：
  python run.py --device cuda --epochs 10 --num-users 500
"""
import argparse
import os
import numpy as np
import torch
from tqdm import tqdm

from config import cfg
from env import KuaiRecEnvData, RecoWorldEnv
from rec_agent import RecAgent, RankingHead, TransformerRankingHead, encode_texts_batch
from user_sim import UserSimulator
from sft_trainer import SFTTrainer
from rl_trainer import GRPOTrainer
from evaluate import run_ablation


def parse_args():
    parser = argparse.ArgumentParser(description="RecoWorld-KuaiRec Training")
    parser.add_argument("--device", type=str, default=None,
                        help="覆盖自动检测的设备，例如 cuda / cuda:1 / cpu")
    parser.add_argument("--epochs", type=int, default=None,
                        help="GRPO 训练轮数")
    parser.add_argument("--num-users", type=int, default=None,
                        help="仿真用户总数")
    parser.add_argument("--group-size", type=int, default=None,
                        help="每用户生成的候选 session 数 G")
    parser.add_argument("--skip-sft", action="store_true",
                        help="跳过 SFT 阶段，直接 GRPO（用于已有 SFT checkpoint 时）")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="数据目录")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录")
    return parser.parse_args()


def apply_args(args):
    """将命令行参数覆盖到 cfg"""
    if args.device:
        cfg.device = args.device
        print(f"[Args] device overridden to: {cfg.device}")
    if args.epochs:
        cfg.grpo_epochs = args.epochs
    if args.num_users:
        cfg.n_sim_users = args.num_users
    if args.group_size:
        cfg.grpo_group_size = args.group_size
    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.output_dir:
        cfg.output_dir = args.output_dir
        os.makedirs(cfg.output_dir, exist_ok=True)


def precompute_item_embeddings(data: KuaiRecEnvData) -> np.ndarray:
    """预计算所有 item embedding，带缓存"""
    cache_path = f"{cfg.cache_dir}/item_embeddings_{data.n_items}.npy"
    if os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        return np.load(cache_path)

    print(f"Computing item embeddings for {data.n_items} items...")
    texts = [data.get_item_text(iid) for iid in range(data.n_items)]
    embeddings = encode_texts_batch(texts, batch_size=25)
    np.save(cache_path, embeddings)
    print(f"Saved embeddings to {cache_path}")
    return embeddings


def main():
    args = parse_args()
    apply_args(args)

    print("=" * 60)
    print("RecoWorld-KuaiRec: SFT → Session-level GRPO (OneRec-aligned)")
    sft_status = "skip" if args.skip_sft else f"{cfg.sft_epochs} epochs"
    print(f"device={cfg.device} | SFT={sft_status} | GRPO={cfg.grpo_epochs} epochs | G={cfg.grpo_group_size}")
    print("=" * 60)

    # ── 1. 数据加载 ──
    print("\n[1/5] Loading data...")
    data = KuaiRecEnvData().load()

    # ── 2. Item embedding ──
    print("\n[2/5] Computing item embeddings...")
    item_embs = precompute_item_embeddings(data)
    data.item_embeddings = item_embs
    for uid, hist in data.user_histories.items():
        embs = [item_embs[iid] for iid in hist[-20:] if iid < len(item_embs)]
        if embs:
            data.user_profiles[uid] = np.mean(embs, axis=0)
    print(f"  Embeddings shape: {item_embs.shape}")
    print(f"  User profiles: {len(data.user_profiles)}")

    # ── 3. 构建环境 & 组件 ──
    print("\n[3/5] Building environment and agents...")
    env = RecoWorldEnv(data)

    ranking_head = TransformerRankingHead().to(cfg.device)
    ckpt = f"{cfg.output_dir}/ranking_head_best.pt"
    if os.path.exists(ckpt):
        ranking_head.load_state_dict(torch.load(ckpt, map_location=cfg.device))
        print(f"  Loaded checkpoint: {ckpt}")

    rec_agent = RecAgent(data, ranking_head)
    rec_agent.build_retriever()

    user_sim = UserSimulator(data)

    all_users = env.sample_users(cfg.n_sim_users)
    split = int(len(all_users) * 0.8)
    train_users, eval_users = all_users[:split], all_users[split:]
    print(f"  Train users: {len(train_users)}, Eval users: {len(eval_users)}")

    # ── 4a. SFT 预训练（热启动 ranking head）──
    sft_ckpt = f"{cfg.output_dir}/ranking_head_sft_final.pt"
    if not args.skip_sft:
        print("\n[4a/5] SFT Pre-training...")
        sft_all_users = list(data.user_histories.keys())  # 全量用户，不限于仿真子集
        sft_trainer = SFTTrainer(ranking_head, data)
        sft_trainer.train(sft_all_users)
    elif os.path.exists(sft_ckpt):
        ranking_head.load_state_dict(torch.load(sft_ckpt, map_location=cfg.device))
        print(f"\n[4a/5] Loaded SFT checkpoint: {sft_ckpt}")
    else:
        print("\n[4a/5] --skip-sft but no checkpoint found, starting GRPO from scratch")

    # ── 4b. Session-level GRPO 训练 ──
    print("\n[4b/5] Session-level GRPO Training...")
    trainer = GRPOTrainer(ranking_head, rec_agent, user_sim, env)
    train_logs = trainer.train(train_users)

    torch.save(ranking_head.state_dict(), f"{cfg.output_dir}/ranking_head_best.pt")
    print(f"  Model saved: {cfg.output_dir}/ranking_head_best.pt")

    # ── 5. 消融评估 ──
    print("\n[5/5] Running ablation experiments...")
    results = run_ablation(env, rec_agent, user_sim, eval_users)

    _print_resume_description(results, train_logs)


def _print_resume_description(results: dict, train_logs: list):
    r_a = results.get("A_offline_baseline", {})
    r_b = results.get("B_sim_user_rule_rec", {})
    r_c = results.get("C_full_recoworld", {})

    ndcg_gain = (r_c.get("avg_ndcg", 0) - r_a.get("avg_ndcg", 0)) / max(r_a.get("avg_ndcg", 1), 1e-9) * 100
    retention_gain = (r_c.get("avg_retention", 0) - r_b.get("avg_retention", 0)) / max(r_b.get("avg_retention", 1), 1e-9) * 100
    ifr = r_c.get("avg_ifr", 0)
    final_reward = train_logs[-1].get("avg_reward", 0) if train_logs else 0

    print("\n" + "=" * 60)
    print("简历描述（参考）：")
    print("-" * 60)
    print(f"""复现 OneRec 核心思路，基于 KuaiRec 构建短视频推荐仿真系统：
设计 Qwen 驱动的用户仿真器，维护用户 mindset（兴趣向量 + 疲劳度），
模拟多轮 session 交互并在疲劳时生成反思指令（如"我想看更多 XXX"）；
推荐 Agent 采用 FAISS 双塔召回 + MLP 排序头，使用 Session-level GRPO 优化，
对齐 OneRec session-wise 生成范式：每用户采样 G={cfg.grpo_group_size} 条完整 session，
以 session 累计奖励做 group 归一化优势，信用分配更全局；
奖励函数融合 watch_ratio、session 留存、指令跟随率和多样性；
三组消融实验验证各模块贡献：
  完整系统 vs 离线 baseline NDCG@10 +{ndcg_gain:.1f}%，
  Session 留存率相比规则推荐 +{retention_gain:.1f}%，
  指令跟随率 {ifr:.1%}，累计奖励 {final_reward:.2f}。""")
    print("=" * 60)


if __name__ == "__main__":
    main()
