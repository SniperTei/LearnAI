# Pairwise vs Scoring：两种评估范式的取舍

> 打分（Scoring）和两两对比（Pairwise）是 LLM-as-Judge 的两大范式。
> 选错了，评估结果会"看起来有数据，实际不准"。

---

## 一句话区分

| | Scoring | Pairwise |
|---|---------|----------|
| 评什么 | 给单条回答打绝对分 | 两条回答比谁更好 |
| 输出 | 1–5 分 | A / B / tie |
| 人判断难度 | 难（绝对分主观） | 易（比较本能） |
| LLM 判断稳定性 | 中 | 高 |

---

## 为什么 Pairwise 通常更准

### 人因层面
人对"哪个更好"的判断，**远比对"打几分"** 敏感、一致。

举个例子：
```
Q: 巴黎奥运会什么时候开幕？
A1: 2024 年 7 月 26 日
A2: 2024 年 7 月 26 日，开幕式首次在塞纳河上举行
```
你给 A1 打几分？A2 打几分？——很难。
但"A1 vs A2 谁更好"——几乎瞬间有答案。

### LLM 层面
- LLM 打绝对分时，分数分布经常聚集（"中庸陷阱"——大多 3–4 分）
- Pairwise 强制二选一，避免中庸
- Pairwise 的 reason 更聚焦，更易审

---

## Pairwise 的核心问题：成本

N 个候选要评出排名，需要：

```
全对比:        N × (N-1) / 2 次 LLM 调用
```

- 10 个候选 → 45 次调用 ✅
- 100 个候选 → 4950 次调用 ❌ 太贵

**对策**：
1. **Swiss Tournament / Bradley-Terry 模型**：每个候选只比 K 次，用统计模型推断排名
2. **Reference-based**：先固定一个"基线版本"，所有新版本只和基线比
3. **Chatbot Arena 模式**：随机抽样对比，长期累积 Elo 评分

---

## Scoring 的核心问题：刻度不一致

### 问题表现
```
你今天的 GPT-4o judge: 大多打 4 分
你明天的 Claude Sonnet judge: 大多打 3 分
你的同事人工标注: 大多打 5 分
```
**谁是"4 分"**？无法跨 judge、跨时间比较。

### 对策
1. **同模型、同 prompt、同 rubric 跑**——分数才能比
2. **校准样本**：每次评估加 N 条"已知答案"的样本作锚点
3. **报告相对值**："比上版提升 0.3 分"，不报"绝对分 4.2"

---

## 什么时候用哪个

### 用 Scoring 当...
- 单条样本独立评估（CI、批量评估）
- 需要可追踪的"绝对指标"（dashboard 展示）
- 候选多（>10 个）或样本多
- 需要细粒度多维评估

### 用 Pairwise 当...
- 对比版本（v1 vs v2）
- 排序多个候选（模型选型、prompt 选型）
- 评估主观质量（"哪个更友好"）
- 不在乎绝对分，只在乎"谁赢"

---

## 实战：版本对比的标准流程

场景：你改了一版 prompt（v2），想确认是否比 v1 好。

### 错误做法
```
跑 v1 100 条 → 平均分 4.1
跑 v2 100 条 → 平均分 4.2
→ "v2 更好，上线"
```
**问题**：0.1 分差异可能只是噪声。

### 正确做法（混合使用）

**Step 1：Scoring 拿粗粒度指标**
```
v1 faithfulness: 0.82
v2 faithfulness: 0.88   ← 显著提升
```

**Step 2：Pairwise 看关键 case 谁更好**
对 30 条"重要 case"两两比：
```
v2 赢: 18 条
v1 赢: 8 条
平:   4 条
```

**Step 3：看 v1 赢的 case 是哪些**
- 如果是关键场景 → v2 还不能上
- 如果是边缘 case → v2 可以上

**决策**：
- Scoring 显著提升 + Pairwise 多数胜 → ✅ 上线
- Scoring 持平 + Pairwise 显著胜 → ✅ 上线（Pairwise 更敏感）
- Scoring 提升但 Pairwise 输 → ⚠️ 数据矛盾，深查

---

## Pairwise 的"位置偏差"陷阱

LLM 在 Pairwise 时有强烈的位置偏好——倾向第一个或第二个位置。

### 对策：双向评估
```python
def pairwise(judge, q, a, b):
    # 正向
    r1 = judge.eval(q, A=a, B=b)  # A 在前
    # 反向
    r2 = judge.eval(q, A=b, B=a)  # B 在前

    if r1 == r2_reversed:
        return r1  # 一致，可信
    else:
        return "tie"  # 矛盾，不信任
```

成本翻倍，但准确性显著提升。**关键评估值得**。

---

## Bradley-Terry / Elo 模型（高级）

如果你有大量对比数据，可以拟合一个模型：

```
P(A 胜 B) = σ(rating_A - rating_B)
```

- 每个候选（模型 / prompt / 系统）有 rating
- 用历史对比数据拟合
- 新候选加入，比几次就能定位

**典型应用**：Chatbot Arena 就是这样给模型排名的。

**对 FDE 的启示**：
- 你不需要造"绝对分数"
- 累积对比数据，长期看 rating 变化
- 这是 LLM 评估的"大数据"思路

---

## 实战推荐：FDE 的最小评估栈

```
日常迭代   →  Scoring（便宜、快）
            └─ 用 RAGAS / LLM-as-Judge 跑 golden set
            └─ 关注趋势，不纠结绝对值

版本决策   →  Pairwise（准、敏感）
            └─ 30–50 条关键 case 双向对比
            └─ 看 win-rate

发布前     →  人工审核（兜底）
            └─ 检查 LLM judge 的 reasoning 是否合理
            └─ 抽样 10–20 条人工复核
```

---

## 自测题

1. 同一份数据，GPT-4o 打平均 4.1 分，Claude 打平均 3.5 分，你怎么解读？
2. v1 平均分 4.0，v2 平均分 4.05，差异可信吗？
3. 你要做"5 个 prompt 选 1 个最优"，用 Scoring 还是 Pairwise？
4. Pairwise 时模型都选"第一个"，怎么修？
5. 1000 条数据想跑 Pairwise，太贵，怎么办？

---

> 下一步：[human_alignment.md](./human_alignment.md)
