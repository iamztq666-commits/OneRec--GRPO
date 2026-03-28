"""
主入口
流程：
  1. 数据加载
  2. Item embedding预计算（带缓存）
  3. 构建FAISS索引
  4. GRPO训练
  5. 消融评估
  6. 打印简历描述
"""
import os
import numpy as np
import torch
from tqdm import tqdm
from openai import OpenAI

from config import cfg
from env import KuaiRecEnvData, RecoWorldEnv
from rec_agent import RecAgent, RankingHead, encode_texts_batch
from user_sim import UserSimulator
from rl_trainer import GRPOTrainer
from evaluate import run_ablation

client = OpenAI(
    api_key=cfg.dashscope_api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


def precompute_item_embeddings(data: KuaiRecEnvData) -> np.ndarray:
    """预计算所有item的embedding，带缓存"""
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
    print("=" * 60)
    print("RecoWorld-KuaiRec: Agentic Recommender with User Simulation")
    print("=" * 60)

    # ── 1. 数据加载 ──
    print("\n[1/5] Loading data...")
    data = KuaiRecEnvData().load()

    # ── 2. Item embedding ──
    print("\n[2/5] Computing item embeddings...")
    item_embs = precompute_item_embeddings(data)
    data.item_embeddings = item_embs
    # 更新用户profile
    for uid, hist in data.user_histories.items():
        embs = [item_embs[iid] for iid in hist[-20:] if iid < len(item_embs)]
        if embs:
            data.user_profiles[uid] = np.mean(embs, axis=0)
    print(f"  Embeddings shape: {item_embs.shape}")
    print(f"  User profiles computed: {len(data.user_profiles)}")

    # ── 3. 构建环境 & 组件 ──
    print("\n[3/5] Building environment and agents...")
    env = RecoWorldEnv(data)

    ranking_head = RankingHead()
    # 加载已有checkpoint（如果存在）
    ckpt = f"{cfg.output_dir}/ranking_head_best.pt"
    if os.path.exists(ckpt):
        ranking_head.load_state_dict(torch.load(ckpt, map_location=cfg.device))
        print(f"  Loaded checkpoint: {ckpt}")

    rec_agent = RecAgent(data, ranking_head)
    rec_agent.build_retriever()

    user_sim = UserSimulator(data)

    # ── 4. 用户采样 ──
    print("\n[4/5] GRPO Training...")
    all_users = env.sample_users(cfg.n_sim_users)
    split = int(len(all_users) * 0.8)
    train_users = all_users[:split]
    eval_users = all_users[split:]
    print(f"  Train users: {len(train_users)}, Eval users: {len(eval_users)}")

    # GRPO训练
    trainer = GRPOTrainer(ranking_head, rec_agent, user_sim, env)
    train_logs = trainer.train(train_users)

    # 保存最终模型
    torch.save(ranking_head.state_dict(), f"{cfg.output_dir}/ranking_head_best.pt")
    print(f"  Model saved: {cfg.output_dir}/ranking_head_best.pt")

    # ── 5. 消融评估 ──
    print("\n[5/5] Running ablation experiments...")
    results = run_ablation(env, rec_agent, user_sim, eval_users)

    # ── 简历描述生成 ──
    _print_resume_description(results, train_logs)


def _print_resume_description(results: dict, train_logs: list):
    """自动生成简历描述"""
    r_a = results.get("A_offline_baseline", {})
    r_b = results.get("B_sim_user_rule_rec", {})
    r_c = results.get("C_full_recoworld", {})

    ndcg_gain = (r_c.get("avg_ndcg", 0) - r_a.get("avg_ndcg", 0)) / max(r_a.get("avg_ndcg", 1), 1e-9) * 100
    retention_gain = (r_c.get("avg_retention", 0) - r_b.get("avg_retention", 0)) / max(r_b.get("avg_retention", 1), 1e-9) * 100
    ifr = r_c.get("avg_ifr", 0)

    final_reward = train_logs[-1].get("avg_reward", 0) if train_logs else 0

    print("\n" + "="*60)
    print("简历描述（参考）：")
    print("-"*60)
    print(f"""复现Meta RecoWorld思路，基于KuaiRec构建短视频Agentic推荐仿真系统：
设计Qwen驱动的用户仿真器，维护用户mindset状态（兴趣向量+疲劳度），
模拟多轮session交互并在疲劳时生成反思指令（如"我想看更多XXX"）；
推荐Agent采用FAISS双塔召回+MLP排序头架构，使用GRPO算法优化排序策略，
奖励函数融合watch_ratio、session留存、指令跟随率和多样性；
三组消融实验验证各模块贡献：
  完整系统 vs 离线baseline NDCG@10 +{ndcg_gain:.1f}%，
  Session留存率相比规则推荐 +{retention_gain:.1f}%，
  指令跟随率 {ifr:.1%}，
  累计奖励 {final_reward:.2f}。""")
    print("="*60)


if __name__ == "__main__":
    main()
