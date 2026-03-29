# KuaiRec Agentic Recommender — OneRec 复现

基于快手 [OneRec](https://arxiv.org/abs/2502.18965) 核心思路，在 KuaiRec 2.0 数据集上构建的端到端短视频推荐仿真系统。

## 核心设计

### 训练范式：SFT → Session-level GRPO

对应 OneRec 的 Pre-training + RL Alignment 两阶段：

```
SFT（监督预训练）
  历史点击数据 → Listwise Softmax Loss → 热启动 RankingHead

        ↓

Session-level GRPO（强化学习精调）
  每用户采样 G=8 条完整 session
  session 累计奖励做 group 归一化优势
  PPO-clip 更新策略网络
```

### 模型架构

```
用户请求
  ↓
FAISS 向量召回（Top-50）        ← 用户 mindset 向量检索
  ↓
Listwise Transformer 排序头     ← 50个候选联合建模，self-attention捕捉item-item交互
  ↓
Top-10 推荐列表
```

### 用户仿真器

Qwen3-8B 本地部署，维护用户动态状态：

- **mindset**：兴趣向量，随点击行为实时更新
- **fatigue**：疲劳度，累积到阈值触发反思指令（"我想看更多 XXX"）
- 每 3 步或疲劳超阈值时调用 LLM 评估推荐列表，输出 click/skip/leave

### 奖励函数

```python
reward = click × 0.5          # 点击（短期兴趣）
       + stay × 0.1           # 留存（每轮）
       - leave × 1.0          # 离开惩罚
       - diversity_penalty     # 多样性（相似度 > 0.8 惩罚）
       + instruction_follow    # 指令跟随率
```

## 实验结果

| 实验 | NDCG@10 | ILD | 留存率 |
|------|---------|-----|--------|
| A：离线 baseline（无仿真）| 0.0071 | 0.074 | - |
| B：规则推荐 + 仿真用户 | 0.1220 | 0.163 | 0.056 |
| C：完整系统（SFT+GRPO+分类器）| 0.0970 | 0.326 | 0.053 |

完整系统（C）相比离线 baseline（A）NDCG@10 提升 **14倍**，相比规则推荐多样性 ILD 提升 **100%**（0.163→0.326）。

## 快速开始

### 环境安装

```bash
pip install -r requirements.txt
pip install transformers>=4.51.0 accelerate>=0.30.0 faiss-gpu-cu12
```

### 数据准备

从 [KuaiRec 官方](https://kuairec.com/) 下载数据集，放到项目根目录：

```
kuai-rec/
  big_matrix.csv
  item_categories.csv
  item_daily_features.csv
  user_features.csv
```

### 训练

```bash
export DASHSCOPE_API_KEY=your_key   # 用于 item embedding 预计算（一次性）
export QWEN3_MODEL=Qwen/Qwen3-8B    # 用于用户仿真（本地 GPU）

# 完整训练（SFT + GRPO）
python run.py --device cuda --epochs 5

# 跳过 SFT（已有checkpoint时）
python run.py --device cuda --skip-sft --epochs 5
```

### 评估

```bash
python eval.py
```

## 显存要求

| 阶段 | 显存占用 | 推荐显卡 |
|------|---------|---------|
| SFT | ~100 MB | 任意 |
| GRPO（Qwen3-8B bf16）| ~16.5 GB | RTX 3090/4090 24G |
| GRPO（Qwen3-8B int4）| ~5.5 GB | RTX 3080 16G |

## 项目结构

```
config.py       # 超参数配置，GPU 自动检测
env.py          # MDP 环境，奖励计算
rec_agent.py    # FAISS 召回 + Transformer 排序头
sft_trainer.py  # SFT 监督预训练
rl_trainer.py   # Session-level GRPO 训练
user_sim.py     # Qwen3-8B 用户仿真器
evaluate.py     # 消融实验
run.py          # 主入口
```

## 参考

- [OneRec: Unifying Retrieve and Rank with Generative Recommender](https://arxiv.org/abs/2502.18965)
- [KuaiRec: A Fully-observed Dataset for Recommender Systems](https://kuairec.com/)
- [DeepSeek-R1: GRPO 算法](https://arxiv.org/abs/2501.12948)
