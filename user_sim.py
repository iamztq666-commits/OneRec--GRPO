"""
User Simulator
- 维护用户mindset：兴趣类别偏好 + 疲劳度 + 情绪
- 逐条评估推荐列表，输出 click/skip/leave
- 疲劳/厌倦时生成反思指令（RecoWorld核心机制）
- 缓存Qwen调用以节省API费用
"""
import json
import hashlib
import pickle
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
import torch

from config import cfg
from env import MDPState, KuaiRecEnvData


# ── LLM调用缓存，避免重复调用 ──
_llm_cache: Dict[str, str] = {}
_cache_path = f"{cfg.cache_dir}/llm_cache.pkl"

def _load_cache():
    global _llm_cache
    if os.path.exists(_cache_path):
        with open(_cache_path, "rb") as f:
            _llm_cache = pickle.load(f)

def _save_cache():
    with open(_cache_path, "wb") as f:
        pickle.dump(_llm_cache, f)


# ── 本地 Qwen3-8B 单例（懒加载，首次调用时初始化）──
_local_model = None
_local_tokenizer = None

def _get_local_model():
    global _local_model, _local_tokenizer
    if _local_model is not None:
        return _local_model, _local_tokenizer

    print(f"[UserSim] Loading local LLM: {cfg.local_llm_model} ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _local_tokenizer = AutoTokenizer.from_pretrained(cfg.local_llm_model)

    load_kwargs = {"device_map": "auto"}
    if cfg.local_llm_dtype == "int4":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16

    _local_model = AutoModelForCausalLM.from_pretrained(
        cfg.local_llm_model, **load_kwargs
    )
    _local_model.eval()
    print(f"[UserSim] Local LLM loaded on {cfg.device}")
    return _local_model, _local_tokenizer


def _call_local_llm(prompt: str, max_tokens: int = 300) -> str:
    """用本地 Qwen3-8B 推理，关闭 thinking 模式加速"""
    model, tokenizer = _get_local_model()
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,   # 关闭 Qwen3 思考链，节省 token
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _call_api_llm(prompt: str, max_tokens: int = 300) -> str:
    """DashScope API fallback"""
    from openai import OpenAI
    client = OpenAI(
        api_key=cfg.dashscope_api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    resp = client.chat.completions.create(
        model=cfg.qwen_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def _cached_llm_call(prompt: str, max_tokens: int = 300) -> str:
    key = hashlib.md5(prompt.encode()).hexdigest()
    if key in _llm_cache:
        return _llm_cache[key]

    if cfg.use_local_llm:
        result = _call_local_llm(prompt, max_tokens)
    else:
        result = _call_api_llm(prompt, max_tokens)

    _llm_cache[key] = result
    _save_cache()
    return result


_load_cache()


# ─────────────────────────────────────────────
# 用户画像（从KuaiRec历史构建）
# ─────────────────────────────────────────────
@dataclass
class UserProfile:
    uid: int
    interest_summary: str       # LLM生成的兴趣描述
    preferred_categories: List[str]
    history_texts: List[str]    # 最近观看的item文本（用于prompt）


def build_user_profile(uid: int, data: KuaiRecEnvData) -> UserProfile:
    """根据历史观看记录用LLM生成用户画像"""
    cache_key = f"profile_{uid}"
    cache_file = f"{cfg.cache_dir}/user_profiles.pkl"
    profiles_cache = {}
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            profiles_cache = pickle.load(f)
    if cache_key in profiles_cache:
        return profiles_cache[cache_key]

    hist = data.user_histories.get(uid, [])[-10:]
    texts = [data.get_item_text(iid) for iid in hist]
    history_str = "\n".join([f"- {t[:80]}" for t in texts[:5]])

    prompt = f"""你是一个短视频用户分析助手。
根据该用户最近观看的视频（标签如下），分析其兴趣偏好。

最近观看：
{history_str}

请输出JSON格式：
{{
  "interest_summary": "用户兴趣的一句话描述（20字以内）",
  "preferred_categories": ["类别1", "类别2", "类别3"]
}}
只输出JSON。"""

    try:
        raw = _cached_llm_call(prompt, max_tokens=150)
        parsed = json.loads(raw)
        profile = UserProfile(
            uid=uid,
            interest_summary=parsed.get("interest_summary", "普通用户"),
            preferred_categories=parsed.get("preferred_categories", ["综合"]),
            history_texts=texts,
        )
    except Exception:
        profile = UserProfile(
            uid=uid,
            interest_summary="普通短视频用户",
            preferred_categories=["综合"],
            history_texts=texts,
        )

    profiles_cache[cache_key] = profile
    with open(cache_file, "wb") as f:
        pickle.dump(profiles_cache, f)
    return profile


# ─────────────────────────────────────────────
# User Simulator
# ─────────────────────────────────────────────
class UserSimulator:
    """
    RecoWorld用户仿真器
    核心机制：
    1. 逐条评估推荐，输出click/skip/leave
    2. 疲劳度累积到阈值 → 触发反思指令
    3. 反思指令：用自然语言表达"想要什么变化"

    若传入 behavior_predictor，则用小分类器替代大部分 LLM 调用；
    LLM 仅在疲劳触发反思指令时使用。
    """

    def __init__(self, data: KuaiRecEnvData, behavior_predictor=None):
        self.data = data
        self.behavior_predictor = behavior_predictor
        self._profiles: Dict[int, UserProfile] = {}
        self._rng = np.random.default_rng(42)

    def get_profile(self, uid: int) -> UserProfile:
        if uid not in self._profiles:
            self._profiles[uid] = build_user_profile(uid, self.data)
        return self._profiles[uid]

    def evaluate_recommendations(
        self, state: MDPState, rec_list: List[int]
    ) -> Tuple[List[str], str]:
        """
        主接口：评估推荐列表
        返回: (actions列表, 反思指令)
        actions中每项为 "click" / "skip" / "leave"
        """
        profile = self.get_profile(state.user_id)
        actions = []
        instruction = ""

        # 构建推荐列表文本
        items_text = []
        for i, iid in enumerate(rec_list):
            text = self.data.get_item_text(iid)[:80]
            items_text.append(f"{i+1}. {text}")

        # 优先用行为预测分类器（速度快 100x）
        if self.behavior_predictor is not None and self.data.item_embeddings is not None:
            actions, instruction = self._predictor_evaluate(state, rec_list)
        else:
            use_llm = self._should_use_llm(state)
            if use_llm:
                actions, instruction = self._llm_evaluate(state, profile, rec_list, items_text)
            else:
                actions, instruction = self._rule_evaluate(state, profile, rec_list)

        return actions, instruction

    def _predictor_evaluate(
        self, state: MDPState, rec_list: List[int]
    ) -> Tuple[List[str], str]:
        """用 BehaviorPredictor 仿真行为，LLM 只在疲劳时生成反思指令"""
        actions = []
        instruction = ""
        user_emb = state.mindset

        for iid in rec_list:
            if iid >= len(self.data.item_embeddings):
                actions.append("skip")
                continue
            item_emb = self.data.item_embeddings[iid]
            action = self.behavior_predictor.predict_action(user_emb, item_emb, state.fatigue)
            actions.append(action)
            if action == "leave":
                break

        while len(actions) < len(rec_list):
            actions.append("skip")

        # 疲劳时用 LLM 生成反思指令（调用频率极低）
        if state.fatigue > cfg.fatigue_threshold:
            profile = self.get_profile(state.user_id)
            cats = profile.preferred_categories
            other_cats = [c for c in ["搞笑", "美食", "科技", "音乐", "生活"] if c not in cats]
            if other_cats:
                target = self._rng.choice(other_cats)
                instruction = f"最近看的内容太单一了，我想看更多{target}相关的视频"
            else:
                instruction = "推荐的内容有点重复，希望多一些新鲜感"

        return actions, instruction

    def _should_use_llm(self, state: MDPState) -> bool:
        """控制LLM调用频率：每3步调一次，或疲劳高时"""
        if state.fatigue > cfg.fatigue_threshold:
            return True
        if state.session_step % 3 == 0:
            return True
        return False

    def _llm_evaluate(
        self, state: MDPState, profile: UserProfile,
        rec_list: List[int], items_text: List[str]
    ) -> Tuple[List[str], str]:
        """用Qwen仿真用户对推荐列表的反应"""
        items_str = "\n".join(items_text)
        fatigue_desc = "很疲劳" if state.fatigue > 0.8 else \
                       "有点累" if state.fatigue > cfg.fatigue_threshold else "精力充沛"
        last_instr = f'上轮指令："{state.last_instruction}"' if state.last_instruction else ""

        prompt = f"""你是一个正在刷短视频的用户，请模拟你的真实行为。

用户画像：{profile.interest_summary}
偏好类别：{', '.join(profile.preferred_categories)}
当前状态：{fatigue_desc}，已刷了{state.session_step}轮
{last_instr}

当前推荐列表：
{items_str}

请模拟用户逐条浏览，对每条视频做出决策（click/skip），如果觉得不想继续了就输出leave。
一旦输出leave，后面的视频不需要评估。
如果很疲劳或对内容不满意，在最后生成一条反思指令（自然语言，表达希望推荐系统做什么改变）。

输出JSON格式：
{{
  "actions": ["click", "skip", "click", "leave"],  // 从第1条开始，遇到leave即停
  "instruction": "我想看更多...",  // 无指令则为空字符串
  "reason": "简短说明（可选）"
}}
只输出JSON。"""

        try:
            raw = _cached_llm_call(prompt, max_tokens=300)
            parsed = json.loads(raw)
            actions = parsed.get("actions", [])
            instruction = parsed.get("instruction", "")
            # 标准化actions
            valid = []
            for a in actions:
                a = str(a).lower().strip()
                if a in ("click", "skip", "leave"):
                    valid.append(a)
                    if a == "leave":
                        break
            # 填充剩余
            while len(valid) < len(rec_list):
                valid.append("skip")
            return valid[:len(rec_list)], instruction
        except Exception:
            return self._rule_evaluate(state, profile, rec_list)

    def _rule_evaluate(
        self, state: MDPState, profile: UserProfile, rec_list: List[int]
    ) -> Tuple[List[str], str]:
        """规则-based快速评估（无LLM，节省API）"""
        actions = []
        instruction = ""

        # 基础点击率受疲劳影响
        base_ctr = max(0.1, 0.4 - state.fatigue * 0.3)
        # 离开概率
        leave_prob = cfg.leave_prob_base + state.fatigue * 0.15

        for iid in rec_list:
            text = self.data.get_item_text(iid).lower()
            # 检查是否匹配偏好类别
            pref_match = any(cat.lower() in text
                             for cat in profile.preferred_categories)
            ctr = base_ctr * (1.5 if pref_match else 0.8)

            # 离开判断
            if self._rng.random() < leave_prob:
                actions.append("leave")
                break

            if self._rng.random() < ctr:
                actions.append("click")
            else:
                actions.append("skip")

        # 补齐
        while len(actions) < len(rec_list):
            actions.append("skip")

        # 疲劳触发反思指令（规则版）
        if state.fatigue > cfg.fatigue_threshold:
            cats = profile.preferred_categories
            other_cats = [c for c in ["搞笑", "美食", "科技", "音乐", "生活"] if c not in cats]
            if other_cats:
                target = self._rng.choice(other_cats)
                instruction = f"最近看的内容太单一了，我想看更多{target}相关的视频"
            else:
                instruction = "推荐的内容有点重复，希望多一些新鲜感"

        return actions, instruction

    def simulate_full_session(
        self, uid: int, env, rec_agent
    ) -> List[Dict]:
        """
        完整session仿真（用于rollout收集）
        返回trajectory: List of {state, action(rec_list), reward, next_state, done}
        """
        state = env.reset(uid)
        trajectory = []

        while not state.done:
            # Agent生成推荐
            rec_list = rec_agent.recommend(state, env)
            # 用户评估
            user_actions, instruction = self.evaluate_recommendations(state, rec_list)
            # 环境step
            result = env.step(state, rec_list, user_actions, instruction)

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
            state = result.next_state

        return trajectory
