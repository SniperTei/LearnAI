# Human Alignment：让 LLM 评委对齐人类

> LLM-as-Judge 没校准就用，等于在沙地上盖楼。
> 人工标注是用来"对齐真理"的，不是替代 LLM 评估。

---

## 为什么必须做人工对齐

### 没对齐时会发生什么
- LLM judge 觉得"答得好"，业务方觉得"完全没用"
- 跑了 100 条，分数 4.5，上线后用户骂街
- 改 prompt 后 judge 分数提升，实际质量下降

**根因**：LLM 的"好"和人类的"好"**不是一回事**。

### 人工标注的真实角色
```
少量人工标注 (20-50 条)
        ↓
    校准 LLM judge
        ↓
LLM judge 大规模评估 (1000+ 条)
        ↓
    发现可疑样本
        ↓
    人工再审核
        ↓
    改进 judge prompt
```
**人在环**不是替代 LLM，是给 LLM 当**真理基准**。

---

## 关键指标：一致性（Agreement）

### 简单同意率（Accuracy）
```
人工和 LLM 打分一致的样本数 / 总样本数
```
- 优点：直观
- 缺点：随机一致性高（比如 5 分制，瞎猜也有 20% 同意率）

### Cohen's Kappa（推荐）
排除了随机一致性的概率：

```
κ = (P_observed - P_expected) / (1 - P_expected)
```

| κ | 一致性强度 |
|---|-----------|
| < 0.2 | 几乎不一致 |
| 0.2–0.4 | 弱 |
| 0.4–0.6 | 中等 |
| 0.6–0.8 | 良好 |
| 0.8–1.0 | 优秀 |

**FDE 实用门槛**：κ ≥ 0.6 才能用，≥ 0.7 算良好。

### Pairwise 的一致性
对 Pairwise，用 **Bradley-Terry 一致性**或简单 win-rate：

```
人工判断 A 比 B 好: 70%
LLM 判断 A 比 B 好: 65%
同意率: 高（趋势一致）
```

---

## 标注实验的最小流程

### Step 1：抽样 20–50 条
- 不要随机抽——按场景分层抽（简单 / 中等 / 难 各占 1/3）
- 包含 edge case（拒答、长答案、敏感问题）

### Step 2：双盲人工标注
- 至少 2 个标注员独立标（避免个人偏差）
- 标注员不知道彼此的分数
- 计算 **标注员之间一致性**（inter-annotator agreement）

**人之间 κ < 0.5**：rubric 定义不清 → 改 rubric，别急着评 LLM。

### Step 3：LLM judge 评同一份数据
用你写好的 judge prompt 跑。

### Step 4：对比 LLM vs 人
- 算 κ
- 找出**最大分歧**的样本——这些是 judge 的盲区

### Step 5：迭代 judge prompt
分析分歧原因：
- 是 rubric 不清？→ 改 rubric
- 是 judge 模型能力不够？→ 换更强模型
- 是任务本身主观？→ 接受偏差，多 judge 投票

---

## 常见的人工-LLM 分歧模式

### 模式 1：LLK 偏向"看起来好的"
- LLM 给"流畅但错了"的答案打高分
- 人工看出"事实错误"
- **对策**：rubric 把"事实错误"设为 0 分一票否决

### 模式 2：LLM 评不出"有用性"
- "礼貌 + 完整" → LLM 给 5 分
- 但人工觉得"完全没解决我的问题"
- **对策**：加"任务完成度"维度，需要实际场景判断

### 模式 3：LLM 在长答案上更宽容
- 长答案看起来"用心" → LLM 给高分
- 人工读着累 → 给低分
- **对策**：加"简洁性"维度

### 模式 4：LLM 看不出"语气不当"
- "您这个问题很简单，答案就是..." → LLM 觉得信息对
- 人工觉得被冒犯
- **对策**：加"语气 / 礼貌"维度

---

## 标注员管理

### 谁来标？
- **领域专家**：最准，但贵、慢（适合关键评估）
- **FDE 自己**：起步阶段必经，最懂业务（适合 calibration）
- **众包工人**：便宜，但质量参差（适合大规模非关键任务）
- **业务方**：最懂真实需求，但没时间（适合抽检）

### 标注规范（Annotation Guideline）
必写清：
1. 每个维度的定义（一句话 + 例子）
2. 每个分数的具体含义
3. 边界 case 怎么处理（拒答、含糊、敏感）
4. 不确定的标"unknown"，不要瞎猜

**反例**：
```
准确性 (1-5): 给答案打分
```
**正例**：
```
准确性:
  1 = 完全错误（核心事实错误）
  2 = 大部分错误
  3 = 对错参半
  4 = 大部分对，细节有误
  5 = 完全准确
边界:
  - 答案拒绝回答 → 标 5（不是错），加 flag "refusal"
  - 答案含糊（"可能是 X"）→ 按最终结论评分
```

### 标注质量监控
- 每批插入 5% 的"金标"样本（已知答案）
- 标注员在金标上 < 80% 同意率 → 重新培训或剔除

---

## 一个最小 calibration 实验

```python
import pandas as pd
from sklearn.metrics import cohen_kappa_score

# 假设你已经有数据
df = pd.DataFrame({
    "question": [...],
    "answer": [...],
    "human_score": [...],   # 人工打的 1-5
    "llm_score": [...],     # LLM judge 打的 1-5
})

# 简单同意率
agree = (df["human_score"] == df["llm_score"]).mean()
print(f"Agreement: {agree:.2%}")

# Cohen's Kappa
kappa = cohen_kappa_score(df["human_score"], df["llm_score"])
print(f"Kappa: {kappa:.2f}")

# 找分歧最大的样本
df["diff"] = (df["human_score"] - df["llm_score"]).abs()
disagreements = df.nlargest(10, "diff")
for _, row in disagreements.iterrows():
    print(f"Q: {row['question']}")
    print(f"  Human: {row['human_score']}, LLM: {row['llm_score']}")
    # 分析为什么
```

---

## 何时"放弃"对齐

不是所有维度都能达到 κ ≥ 0.8。出现以下情况，可以接受较低一致性：

1. **任务本身高度主观**（创意写作、推荐）—— 0.5–0.6 即可
2. **业务方接受 LLM judge 作为代理指标**—— 不强求人工一致
3. **评估目的是趋势对比**，不是绝对决策—— 一致性只要稳定就行

**但**：在以下场景，必须严格对齐（κ ≥ 0.7）：
- 涉及合规 / 安全
- 影响业务核心 KPI
- 发布前的回归测试

---

## 关键产出物

每次 calibration 实验后应该有：
1. **一致性报告**（κ 值 + 分歧分布图）
2. **judge prompt 改进版**（基于分歧分析）
3. **失败案例集**（10–20 条 LLM 评错的）→ 用作回归测试集
4. **标注规范 v2**（如果发现 rubric 模糊）

---

## 自测题

1. Cohen's Kappa 0.4，能上线 LLM judge 吗？
2. 两个标注员之间 κ 0.3，问题在 LLM 还是 rubric？
3. LLM judge 在长答案上系统性偏高，怎么办？
4. 没钱请标注员，能自己做对齐吗？最少几条？
5. calibration 做完了，下一周模型换了，需要重新对齐吗？

---

> 下一步：[../online/user_feedback.md](../online/user_feedback.md)
