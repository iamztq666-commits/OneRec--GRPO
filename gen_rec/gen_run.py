"""
生成式推荐主入口（miniOneRec）

流程：
  1. 数据：构建 Semantic IDs + 切割真实 session + 生成训练对
  2. SFT：LoRA 微调 Qwen3-8B，学会生成 session SID 序列
  3. GRPO：用 BehaviorPredictor 打分，优化生成策略

运行：
  python gen_run.py --sft-epochs 3 --grpo-epochs 3
  python gen_run.py --skip-sft --grpo-epochs 3   # 已有 SFT checkpoint
"""
import _path  # noqa

import argparse
import os
import json
import numpy as np
import torch

from config import cfg
from env import KuaiRecEnvData
from session_data import prepare_session_data, save_as_jsonl
from generative_rec import GenerativeRec
from gen_sft import train_sft
from gen_grpo import GenGRPOTrainer
from behavior_predictor import train_behavior_predictor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sft-epochs", type=int, default=3)
    p.add_argument("--grpo-epochs", type=int, default=3)
    p.add_argument("--skip-sft", action="store_true")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.device:
        cfg.device = args.device

    print("=" * 60)
    print("miniOneRec: Qwen3-8B + SemanticIDs + SFT → GRPO")
    print("=" * 60)

    # ── 1. 数据加载 ──
    print("\n[1/4] Loading data...")
    data = KuaiRecEnvData().load()
    item_embs = np.load(f"{cfg.cache_dir}/item_embeddings_{data.n_items}.npy")
    data.item_embeddings = item_embs

    # ── 2. Semantic IDs + Session 切割 ──
    print("\n[2/4] Preparing session data...")
    samples, iid2sid, sid2iid, vocab = prepare_session_data(data, item_embs)

    jsonl_path = f"{cfg.cache_dir}/session_train.jsonl"
    if not os.path.exists(jsonl_path):
        save_as_jsonl(samples, iid2sid, jsonl_path, max_samples=30_000)
    print(f"  Training pairs ready: {jsonl_path}")

    # ── 3. 加载模型 ──
    print("\n[3/4] Loading Qwen3-8B + LoRA...")
    load_lora = args.skip_sft and os.path.exists(
        f"{cfg.output_dir}/generative_rec_lora")
    gen_rec = GenerativeRec(vocab=vocab, iid2sid=iid2sid, sid2iid=sid2iid,
                             load_lora=load_lora)

    # ── 4a. SFT ──
    if not args.skip_sft:
        print(f"\n[4a/4] SFT ({args.sft_epochs} epochs)...")
        train_sft(gen_rec, jsonl_path, epochs=args.sft_epochs)
    else:
        print("\n[4a/4] Skipping SFT.")

    # ── 4b. BehaviorPredictor ──
    print("\n[4b/4] Loading BehaviorPredictor...")
    bp = train_behavior_predictor(data, item_embs)

    # ── 4c. GRPO ──
    print(f"\n[4c/4] GRPO ({args.grpo_epochs} epochs)...")
    # 更新用户 profile
    for uid, hist in data.user_histories.items():
        embs = [item_embs[iid] for iid in hist[-20:] if iid < len(item_embs)]
        if embs:
            data.user_profiles[uid] = np.mean(embs, axis=0)

    all_uids = list(data.user_histories.keys())
    np.random.shuffle(all_uids)
    train_uids = all_uids[:int(len(all_uids) * 0.8)]

    grpo_trainer = GenGRPOTrainer(
        gen_rec=gen_rec,
        behavior_predictor=bp,
        item_embeddings=item_embs,
        user_histories=data.user_histories,
        user_profiles=data.user_profiles,
    )
    logs = grpo_trainer.train(train_uids, epochs=args.grpo_epochs)

    with open(f"{cfg.output_dir}/gen_grpo_log.json", "w") as f:
        json.dump(logs, f, indent=2)

    print("\n" + "=" * 60)
    print("Done. miniOneRec training complete.")
    print(f"  LoRA checkpoint: {cfg.output_dir}/generative_rec_lora")
    print("=" * 60)


if __name__ == "__main__":
    main()
