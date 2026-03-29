"""
生成式推荐 SFT 训练

用真实 session 数据微调 Qwen3-8B（LoRA），
学会"给定历史，生成目标 session 的 SID token 序列"。

验收标准：
  1. 生成的是合法 item tokens（能解析回 item id）
  2. top-k 命中真实 session 的能力（Hit@K）
  3. 生成长度和去重可控
"""
import _path  # noqa

import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
from typing import List

from config import cfg
from generative_rec import GenerativeRec


class SessionDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 1024):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(jsonl_path) as f:
            for line in f:
                self.samples.append(json.loads(line))
        print(f"[SFT] Dataset: {len(self.samples):,} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        full_text = s["input"] + "\n" + s["target"]
        prompt_len = len(self.tokenizer(s["input"] + "\n",
                                        return_tensors="pt").input_ids[0])

        enc = self.tokenizer(
            full_text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
        )
        input_ids = enc.input_ids[0]
        attention_mask = enc.attention_mask[0]

        # label：只在 target 部分计算 loss，prompt 部分 mask 掉
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def evaluate_hit_rate(gen_rec: GenerativeRec, val_samples: list,
                      n_eval: int = 50, k: int = 10) -> float:
    """计算 Hit@K：生成 session 中命中真实 session item 的比例"""
    hits = []
    for s in val_samples[:n_eval]:
        generated = gen_rec.generate_session(s["history_iids"], n_items=k)
        target_set = set(s["target_iids"])
        if not target_set:
            continue
        hit = len(set(generated) & target_set) / len(target_set)
        hits.append(hit)
    return float(np.mean(hits)) if hits else 0.0


def train_sft(gen_rec: GenerativeRec, jsonl_path: str,
              epochs: int = 3, batch_size: int = 2, lr: float = 2e-4):
    """
    SFT 训练主循环。

    batch_size=2 是因为 Qwen3-8B + LoRA 在 24G 显存下的极限。
    用梯度累积模拟更大 batch。
    """
    dataset = SessionDataset(jsonl_path, gen_rec.tokenizer)

    # 80/20 split
    n = len(dataset)
    split = int(n * 0.8)
    train_ds = torch.utils.data.Subset(dataset, range(split))
    val_samples_raw = []
    with open(jsonl_path) as f:
        all_raw = [json.loads(l) for l in f]
    val_raw = all_raw[split:]

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    optimizer = AdamW(gen_rec.model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_dl) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

    accum_steps = 8  # 梯度累积，等效 batch_size=16
    gen_rec.model.train()
    best_hit = 0.0

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_dl, desc=f"SFT Epoch {epoch}/{epochs}")):
            input_ids = batch["input_ids"].to(gen_rec.model.device)
            attention_mask = batch["attention_mask"].to(gen_rec.model.device)
            labels = batch["labels"].to(gen_rec.model.device)

            outputs = gen_rec.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / accum_steps
            loss.backward()
            total_loss += loss.item() * accum_steps

            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(gen_rec.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_loss = total_loss / len(train_dl)

        # 评估
        gen_rec.model.eval()
        hit = evaluate_hit_rate(gen_rec, val_raw)
        gen_rec.model.train()

        print(f"  Epoch {epoch}/{epochs} | loss={avg_loss:.4f} | Hit@10={hit:.3f}")

        if hit > best_hit:
            best_hit = hit
            gen_rec.save_lora()
            print(f"  Saved best model (Hit@10={best_hit:.3f})")

    print(f"[SFT] Done. Best Hit@10={best_hit:.3f}")
    return best_hit
