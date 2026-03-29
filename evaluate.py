"""
评估模块
三组消融对比（RecoWorld论文核心卖点）：
  A. 离线NDCG baseline（无仿真用户，静态推荐）
  B. 有仿真用户 + 规则推荐（无RL）
  C. 完整RecoWorld（仿真用户 + GRPO训练的MLP head）

核心指标：
  - Session留存率（avg session length / max_session_steps）
  - 指令跟随率（instruction follow rate）
  - NDCG@10（离线评估）
  - ILD（Intra-List Diversity，多样性）
  - 累计奖励
"""
import json
import numpy as np
from typing import List, Dict, Optional
from tqdm import tqdm

from config import cfg
from env import RecoWorldEnv, MDPState, KuaiRecEnvData
from rec_agent import RecAgent, RankingHead
from user_sim import UserSimulator


# ─────────────────────────────────────────────
# 指标计算
# ─────────────────────────────────────────────
def ndcg_at_k(recommended: List[int], relevant: set, k: int = 10) -> float:
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, iid in enumerate(recommended[:k])
        if iid in relevant
    )
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def intra_list_diversity(rec_list: List[int], item_embeddings: np.ndarray) -> float:
    """ILD: 推荐列表内平均pairwise距离（越高越多样）"""
    if item_embeddings is None or len(rec_list) < 2:
        return 0.0
    embs = np.array([item_embeddings[iid]
                     for iid in rec_list if iid < len(item_embeddings)],
                    dtype=np.float32)
    if len(embs) < 2:
        return 0.0
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    normed = embs / (norms + 1e-9)
    sim = normed @ normed.T
    n = len(normed)
    upper = sim[np.triu_indices(n, k=1)]
    return float(1 - np.mean(upper))


def instruction_follow_rate(trajectory: List[Dict], env: RecoWorldEnv) -> float:
    """计算trajectory中指令跟随率"""
    total, followed = 0, 0
    for t in trajectory:
        instr = t["state"].last_instruction
        if instr:
            total += 1
            if t["info"].get("instruction_followed", False):
                followed += 1
    return followed / max(total, 1)


# ─────────────────────────────────────────────
# 规则推荐（Baseline B对照）
# ─────────────────────────────────────────────
class RuleBasedAgent:
    """无RL的简单推荐：用户mindset向量FAISS召回，无重排"""
    def __init__(self, data: KuaiRecEnvData, rec_agent: RecAgent):
        self.data = data
        self.rec_agent = rec_agent

    def recommend(self, state: MDPState, env=None) -> List[int]:
        """纯FAISS召回，无MLP重排"""
        if self.rec_agent.retriever is None:
            return list(np.random.choice(self.data.n_items,
                                         cfg.rec_list_size, replace=False))
        candidates = self.rec_agent.retriever.retrieve(state.mindset, cfg.recall_topk)
        seen = set(state.history_iids[-50:])
        candidates = [c for c in candidates if c not in seen]
        return candidates[:cfg.rec_list_size]


# ─────────────────────────────────────────────
# 单episode评估
# ─────────────────────────────────────────────
def evaluate_episode(uid: int, env: RecoWorldEnv, agent,
                     user_sim: UserSimulator,
                     use_user_sim: bool = True) -> Dict:
    state = env.reset(uid)
    trajectory = []
    total_reward = 0.0
    session_length = 0

    # 离线相关集合（用户历史后20%作测试集）
    hist = env.data.user_histories.get(uid, [])
    cutoff = int(len(hist) * 0.8)
    test_items = set(hist[cutoff:])

    all_ndcg = []
    all_ild = []

    while not state.done:
        rec_list = agent.recommend(state, env)

        if use_user_sim:
            user_actions, instruction = user_sim.evaluate_recommendations(state, rec_list)
        else:
            # 离线baseline：直接用watch_ratio判断行为
            user_actions = []
            instruction = ""
            for iid in rec_list:
                wr = float(env._wr_matrix[uid, iid]) if env._wr_matrix is not None else 0.0
                if wr >= cfg.watch_ratio_threshold:
                    user_actions.append("click")
                else:
                    user_actions.append("skip")

        result = env.step(state, rec_list, user_actions, instruction)

        ndcg = ndcg_at_k(rec_list, test_items, cfg.ndcg_k)
        ild = intra_list_diversity(rec_list, env.data.item_embeddings)
        all_ndcg.append(ndcg)
        all_ild.append(ild)

        trajectory.append({
            "state": state,
            "rec_list": rec_list,
            "user_actions": user_actions,
            "instruction": instruction,
            "reward": result.reward,
            "next_state": result.next_state,
            "done": result.done,
            "info": result.info,
        })

        total_reward += result.reward
        session_length += 1
        state = result.next_state

    return {
        "uid": uid,
        "total_reward": total_reward,
        "session_length": session_length,
        "retention_rate": session_length / cfg.max_session_steps,
        "avg_ndcg": np.mean(all_ndcg),
        "avg_ild": np.mean(all_ild),
        "instruction_follow_rate": instruction_follow_rate(trajectory, env),
        "trajectory_len": len(trajectory),
    }


# ─────────────────────────────────────────────
# 消融实验主函数
# ─────────────────────────────────────────────
def run_ablation(env: RecoWorldEnv, rec_agent: RecAgent,
                 user_sim: UserSimulator, eval_users: List[int]) -> Dict:
    """
    三组消融：
    A: 离线NDCG（无仿真用户，规则agent）
    B: 仿真用户 + 规则召回（无RL）
    C: 仿真用户 + GRPO训练MLP（完整RecoWorld）
    """
    rule_agent = RuleBasedAgent(env.data, rec_agent)
    results = {}

    # ── A: 离线baseline ──
    print("\n[Ablation A] 离线baseline（无仿真用户）")
    scores_a = []
    for uid in tqdm(eval_users[:cfg.eval_episodes]):
        r = evaluate_episode(uid, env, rule_agent, user_sim, use_user_sim=False)
        scores_a.append(r)
    results["A_offline_baseline"] = _aggregate(scores_a)
    print(f"  NDCG@10: {results['A_offline_baseline']['avg_ndcg']:.4f}")

    # ── B: 仿真用户 + 规则推荐 ──
    print("\n[Ablation B] 仿真用户 + 规则推荐（无RL）")
    scores_b = []
    for uid in tqdm(eval_users[:cfg.eval_episodes]):
        r = evaluate_episode(uid, env, rule_agent, user_sim, use_user_sim=True)
        scores_b.append(r)
    results["B_sim_user_rule_rec"] = _aggregate(scores_b)
    print(f"  Retention: {results['B_sim_user_rule_rec']['avg_retention']:.3f} "
          f"| NDCG@10: {results['B_sim_user_rule_rec']['avg_ndcg']:.4f} "
          f"| IFR: {results['B_sim_user_rule_rec']['avg_ifr']:.3f}")

    # ── C: 完整RecoWorld ──
    print("\n[Ablation C] 完整RecoWorld（仿真用户 + GRPO MLP）")
    scores_c = []
    for uid in tqdm(eval_users[:cfg.eval_episodes]):
        r = evaluate_episode(uid, env, rec_agent, user_sim, use_user_sim=True)
        scores_c.append(r)
    results["C_full_recoworld"] = _aggregate(scores_c)
    print(f"  Retention: {results['C_full_recoworld']['avg_retention']:.3f} "
          f"| NDCG@10: {results['C_full_recoworld']['avg_ndcg']:.4f} "
          f"| IFR: {results['C_full_recoworld']['avg_ifr']:.3f}")

    # ── 汇总 ──
    print("\n" + "="*60)
    print("消融实验汇总")
    print(f"{'实验':35s} {'留存率':>8} {'NDCG@10':>9} {'ILD':>7} {'指令跟随率':>10} {'累计奖励':>9}")
    for name, r in results.items():
        print(f"{name:35s} {r['avg_retention']:>8.3f} {r['avg_ndcg']:>9.4f} "
              f"{r['avg_ild']:>7.3f} {r['avg_ifr']:>10.3f} {r['avg_reward']:>9.3f}")

    with open(f"{cfg.output_dir}/ablation_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {cfg.output_dir}/ablation_results.json")
    return results


def _aggregate(scores: List[Dict]) -> Dict:
    return {
        "avg_retention": np.mean([s["retention_rate"] for s in scores]),
        "avg_ndcg": np.mean([s["avg_ndcg"] for s in scores]),
        "avg_ild": np.mean([s["avg_ild"] for s in scores]),
        "avg_ifr": np.mean([s["instruction_follow_rate"] for s in scores]),
        "avg_reward": np.mean([s["total_reward"] for s in scores]),
        "avg_session_len": np.mean([s["session_length"] for s in scores]),
        "n_episodes": len(scores),
    }
