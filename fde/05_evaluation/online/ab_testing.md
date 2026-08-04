# AI 产品的 A/B 测试

> 传统 A/B 测试 + LLM 的不确定性 = 一堆坑。
> FDE 必须知道哪些坑、怎么绕。

---

## A/B 测试在 AI 场景的特殊性

### 传统软件 A/B
```
A 组：按钮是蓝色
B 组：按钮是红色
看哪组点击率高
```
- 确定性：同一用户每次看到的都一样
- 因果清晰：差异来自按钮颜色

### AI 产品 A/B
```
A 组：用 prompt v1
B 组：用 prompt v2
看哪组满意度高
```
- **非确定性**：同一 prompt，同一用户，两次回答可能不一样
- **因果模糊**：差异来自版本？还是模型波动？还是 query 分布变化？

**结论**：AI 的 A/B 测试**比传统 A/B 难得多**，需要更严格的实验设计。

---

## 三个关键陷阱

### 陷阱 1：方差大（噪声 > 信号）
LLM 输出有随机性，导致同一版本的指标本身就波动。
如果 v1 和 v2 的差异 < 噪声，你看的"提升"是幻觉。

**对策**：
- 每个版本用**固定 seed / temperature=0**（部分可控）
- 同一 query 跑多次取平均
- 算统计显著性（不是看平均数）

### 陷阱 2：query 分布漂移
不同时间段，用户问的问题分布不一样（工作日 vs 周末、上午 vs 下午）。
如果你按时间分版本（前一周 v1，后一周 v2），分布差异会污染结果。

**对策**：
- **同时**跑两个版本（随机分流），不要按时间切
- 用 AA 测试验证分流的两组是否本身一致

### 陷阱 3：学习效应
用户对新版本有新鲜感（或抵触），短期内行为变化。
- 早期数据偏差大
- 长期才能反映真实效果

**对策**：
- 实验至少跑 1–2 周
- 排除前 2–3 天的数据（熟悉期）

---

## A/B 测试的正确姿势

### Step 1：明确假设
```
H1: 把 prompt 从 v1 换到 v2，会提升答案准确率（faithfulness）
H0: 没差异
```
**没有假设就开测，等于没测**。

### Step 2：选定**一个**主指标
- 不能同时优化 5 个指标
- 主指标做决策，其他指标做参考

**例子**：
- 主：faithfulness（准确性）
- 辅：用户 👍 率、平均延迟、单次成本

### Step 3：算样本量
不知道样本量 = 不知道什么时候停。

```python
# 简化公式
# n = 16 * σ² / Δ²
# σ = 指标标准差
# Δ = 你想检测的最小差异

import numpy as np
sigma = 0.15  # 假设 faithfulness 标准差 0.15
delta = 0.02  # 想检测 2% 提升
n = 16 * sigma**2 / delta**2
print(f"每组至少需要 {int(n)} 样本")
# 90 样本每组
```

**没算样本量直接跑**：可能跑太少（看不出差异）或跑太多（浪费钱）。

### Step 4：随机分流
- 按用户 ID 分流（同一用户体验一致）
- 不要按请求分流（同一用户看到不同版本，体验混乱）
- 50/50 分（不要 90/10，统计功效不够）

### Step 5：跑实验
- 同时跑两个版本
- 持续 1–2 周
- 监控实验健康度（流量、错误率、极端 case）

### Step 6：分析
```python
from scipy import stats

# 两组 faithfulness 分数
v1_scores = [...]  # 100 个样本
v2_scores = [...]

# t 检验
t, p = stats.ttest_ind(v1_scores, v2_scores)
print(f"p-value: {p}")

if p < 0.05:
    print("差异显著")
else:
    print("无法拒绝 H0（差异不显著）")
```

**关键**：报告 p-value，不只报平均数差异。

### Step 7：决策
```
显著 + 方向对  → 全量上线
显著 + 方向反  → 不上线，分析为什么
不显著          → 要么差异小，要么样本少；考虑：
                  1. 加样本再跑
                  2. 接受"无差异"，选简单的版本
```

---

## 特殊场景：模型路由 / Agent 的 A/B

### 模型路由
A：全部用 GPT-4o
B：简单 query 用 Haiku，复杂用 GPT-4o

**主指标**：成本
**辅指标**：质量（必须不下降）

陷阱：质量"看起来没下降" 但难 case 变差了。
对策：分层分析（简单 case / 难 case 各自的指标）。

### Agent
A：单步 Agent
B：ReAct 多步 Agent

**陷阱**：
- Agent 路径不同，每次执行步骤数都不一样
- 延迟方差极大
- 用户感知的"快慢"和实际指标对不上

对策：
- 主指标用任务成功率（不是延迟）
- 给每个 case 标注"复杂度"，分桶分析

---

## 影子评估（Shadow Evaluation）

为了避免 A/B 影响线上用户，可以用影子模式：

```
真实流量 → v1（用户看到）
       └→ v2（同时跑，但用户看不到，只记录结果）
```

- 用户感知只有 v1
- 离线对比 v1 和 v2 的输出
- 评估完全无风险

**适用**：
- 大改动，不敢直接 A/B
- 想保留真实流量分布

**缺点**：成本翻倍；用户体验反馈滞后。

---

## Multi-armed Bandit（高级）

如果 A/B 太慢，可以用 bandit 算法：

```
多个版本同时在线
系统自动给"表现好"的版本更多流量
劣质版本流量自动衰减
```

- 优势：不需要等实验结束，边学边优化
- 劣势：失去严格统计显著性
- 适用：长期优化、版本众多

---

## 实战：my_com_rag 的 A/B 最小实现

### 流量分流
```python
import hashlib

def assign_group(user_id, experiment="prompt_v2"):
    h = hashlib.md5(f"{user_id}:{experiment}".encode()).hexdigest()
    return "A" if int(h[:8], 16) % 2 == 0 else "B"

# 在请求处理时
group = assign_group(user_id)
if group == "A":
    response = run_rag(query, prompt_version="v1")
else:
    response = run_rag(query, prompt_version="v2")

# 记录
log_to_langfuse(
    user_id=user_id,
    group=group,
    query=query,
    response=response,
    prompt_version=...,
)
```

### 分析
```sql
SELECT
    experiment_group,
    COUNT(*) as n,
    AVG(rating) as avg_rating,
    STDDEV(rating) as std_rating
FROM feedback
WHERE experiment = 'prompt_v2'
  AND timestamp > '2026-07-01'
GROUP BY experiment_group;
```

---

## A/B 测试的常见反模式

❌ **按时间分版本**（前一周 A，后一周 B）→ 分布漂移污染
❌ **看一眼觉得"差异挺大"就上线** → 没算显著性
❌ **同时跑 10 个实验** → 互相干扰
❌ **A/B 跑到一半发现"明显 B 好"，立即停** → 数据不全
❌ **主指标没显著但辅指标显著，强行上线** → cherry-picking
❌ **没做 AA 测试** → 不知道分流本身有没有 bug

---

## AA 测试（前置 sanity check）

在跑真 A/B 之前，先做 AA：把同一版本分两组跑。

```
预期：两组指标接近，无显著差异
实际：如果有显著差异 → 分流算法或指标有问题
```

**AA 通过，才能信 A/B 的结果**。FDE 必做。

---

## 什么时候**不**做 A/B

- 流量太小（每天 < 100 用户）→ 统计功效不够
- 改动很小且明确（修个 typo）→ 直接上
- 紧急修复 → 先上，后 A/B 验证
- 长尾低频场景 → 样本不够，用离线评估

**A/B 不是万能**。FDE 要判断什么时候该用，什么时候用别的方法。

---

## 自测题

1. v1 满意度 4.1，v2 满意度 4.3，能上 v2 吗？
2. 你只有 50 个日活用户，能跑 A/B 吗？
3. 跑了一周 A/B，发现 B 组特别好，要不要立即停 A？
4. A/B 中 v2 主指标没显著，但辅指标显著，怎么决策？
5. 用户体感"v2 更好"，但数据上没差异，可能是什么原因？

---

> 下一步：[adoption_metrics.md](./adoption_metrics.md)
