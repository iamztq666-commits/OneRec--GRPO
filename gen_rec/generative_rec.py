"""
生成式推荐模型（Qwen3-8B + LoRA）

范式：decoder-only，给定历史 SID token 序列，自回归生成目标 session SID 序列。
对应 OneRec/OpenOneRec：Qwen backbone + Itemic Tokens + session-wise generation。

SFT 格式：
  input:  <|hist_begin|> ... SID tokens ... <|hist_end|> 请生成下一次推荐 session：
  target: <|session_begin|> ... SID tokens ... <|session_end|>

GRPO 阶段：
  对同一历史生成 G=8 条 session，用 reward 做 group 归一化优势，更新 LoRA 权重。
"""
import _path  # noqa

import os
import re
import json
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from config import cfg
from semantic_ids import sid_to_tokens, tokens_to_sid

LORA_CKPT = f"{cfg.output_dir}/generative_rec_lora"


# ─────────────────────────────────────────────
# 模型包装
# ─────────────────────────────────────────────
class GenerativeRec:
    """
    Qwen3-8B + LoRA 生成式推荐器。
    推理时给定历史序列，生成目标 session 的 SID token 序列。
    """

    def __init__(self, vocab: List[str], iid2sid: Dict, sid2iid: Dict,
                 load_lora: bool = False):
        self.iid2sid = iid2sid
        self.sid2iid = sid2iid
        self.vocab = vocab

        print(f"[GenRec] Loading {cfg.local_llm_model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.local_llm_model)

        # 添加 special tokens
        self.tokenizer.add_special_tokens({"additional_special_tokens": vocab})

        load_kwargs = {"device_map": "auto"}
        if cfg.local_llm_dtype == "int4":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        else:
            load_kwargs["dtype"] = torch.bfloat16

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.local_llm_model, **load_kwargs
        )
        self.model.resize_token_embeddings(len(self.tokenizer))

        # 加载 LoRA
        if load_lora and os.path.exists(LORA_CKPT):
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, LORA_CKPT)
            print(f"[GenRec] Loaded LoRA from {LORA_CKPT}")
        elif not load_lora:
            self._init_lora()

        self.model.eval()
        print(f"[GenRec] Ready. Vocab size: {len(self.tokenizer)}")

    def _init_lora(self):
        """初始化 LoRA 适配器"""
        from peft import get_peft_model, LoraConfig, TaskType
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_cfg)
        self.model.print_trainable_parameters()

    def history_to_prompt(self, history_iids: List[int]) -> str:
        """历史 item ids -> prompt 文本"""
        hist_tokens = []
        for iid in history_iids[-20:]:
            if iid in self.iid2sid:
                hist_tokens.append(sid_to_tokens(self.iid2sid[iid]))
        return (
            "<|hist_begin|>\n" +
            "\n".join(hist_tokens) +
            "\n<|hist_end|>\n请生成下一次推荐 session："
        )

    def session_to_text(self, iids: List[int]) -> str:
        """session item ids -> target 文本"""
        toks = [sid_to_tokens(self.iid2sid[i]) for i in iids if i in self.iid2sid]
        return "<|session_begin|>\n" + "\n".join(toks) + "\n<|session_end|>"

    @torch.no_grad()
    def generate_session(
        self,
        history_iids: List[int],
        n_items: int = 10,
        temperature: float = 0.0,
        top_p: float = 0.9,
    ) -> Tuple[List[int], float]:
        """
        给定历史，生成一个 session 的 item 列表。
        返回: (item_ids, avg_log_prob)
        """
        prompt = self.history_to_prompt(history_iids)
        prompt += "\n<|session_begin|>"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs.input_ids.shape[1]

        gen_kwargs = dict(
            max_new_tokens=n_items * 8,  # 每个 item 约 5 token
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else 1.0,
            top_p=top_p if temperature > 0 else 1.0,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.convert_tokens_to_ids("<|session_end|>"),
        )
        output = self.model.generate(**inputs, **gen_kwargs)
        generated = output[0][input_len:]
        gen_text = self.tokenizer.decode(generated, skip_special_tokens=False)

        # 解析生成的 SID tokens -> item ids
        item_ids = self._parse_session(gen_text)
        return item_ids

    def _parse_session(self, text: str) -> List[int]:
        """从生成文本中解析 item ids，去重"""
        pattern = r"<\|sid_begin\|>(.*?)<\|sid_end\|>"
        matches = re.findall(pattern, text, re.DOTALL)
        item_ids = []
        seen = set()
        for m in matches:
            try:
                sid = tokens_to_sid("<|sid_begin|>" + m + "<|sid_end|>")
                if sid in self.sid2iid:
                    iid = self.sid2iid[sid]
                    if iid not in seen:
                        item_ids.append(iid)
                        seen.add(iid)
            except Exception:
                continue
        return item_ids

    def compute_session_logp(
        self,
        history_iids: List[int],
        target_iids: List[int],
    ) -> torch.Tensor:
        """
        计算生成目标 session 的 log probability（用于 GRPO）。
        """
        prompt = self.history_to_prompt(history_iids) + "\n"
        target = self.session_to_text(target_iids)
        full_text = prompt + target

        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)
        prompt_inputs = self.tokenizer(prompt, return_tensors="pt")
        prompt_len = prompt_inputs.input_ids.shape[1]

        self.model.train()
        with torch.enable_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # (1, T, V)

        # 只计算 target 部分的 log prob
        shift_logits = logits[0, prompt_len - 1:-1, :]
        shift_labels = inputs.input_ids[0, prompt_len:]

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_logp = log_probs[range(len(shift_labels)), shift_labels]
        return token_logp.sum()

    def save_lora(self):
        os.makedirs(LORA_CKPT, exist_ok=True)
        self.model.save_pretrained(LORA_CKPT)
        self.tokenizer.save_pretrained(LORA_CKPT)
        print(f"[GenRec] LoRA saved to {LORA_CKPT}")
