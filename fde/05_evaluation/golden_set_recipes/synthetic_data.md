# 用 LLM 合成评测数据

> 完全靠人工造数据太慢。
> 用 LLM 合成能快速扩量——但**坑很多**。

---

## 什么时候用合成数据

### 适合
- 冷启动阶段（没真实数据）
- 扩展现有数据集（从 30 条扩到 300 条）
- 生成对抗 case（越狱、边界）
- 多语言 / 多风格改写

### 不适合
- 完全替代真实数据（合成数据有 AI 偏差）
- 评估 LLM-as-Judge（用同款 LLM 生成的数据训练和评估，循环偏差）
- 业务关键评估的唯一依据

**原则**：合成数据是**补充**，不是**替代**。

---

## 合成数据的 3 大风险

### 风险 1：分布偏差
LLM 生成的问题**不像真人问的**：
- 太规整、太书面
- 倾向复杂句式
- 缺少错别字、口语化、不完整

**后果**：系统在合成集上表现很好，在真实用户面前一塌糊涂。

### 风险 2：答案"自带"
LLM 生成 Q 时，潜意识里**已经有答案**了，导致：
- Q 过于"对应"某段文档
- 答案过于"标准"
- 难度被低估

### 风险 3：循环偏差
用 GPT-4 生成的数据，评估 GPT 系列模型 → 系统性偏好 GPT。
**任何"模型 X 表现好"的结论都可疑**。

---

## 合成数据的正确姿势

### 策略 1：基于真实文档生成
- 给 LLM 一份文档
- 让它**从文档里**生成用户可能问的问题
- ground_truth 直接从文档摘

```python
prompt = """
你是一名 {domain} 专家。
基于下面的文档，生成 10 个用户可能问的问题。

要求：
1. 问题必须能从文档回答
2. 涵盖不同难度（4 简单、4 中等、2 难）
3. 包含口语化、错别字、模糊问法（真实用户风格）
4. 不要太"完美"

文档:
{document}

输出 JSONL:
{"question": "...", "ground_truth": "...", "difficulty": "easy|medium|hard", "quote_from_doc": "..."}
"""
```

### 策略 2：改写真实 query
- 拿真实 query
- 用 LLM 改写成 N 种变体
- ground_truth 保持原样

```python
prompt = """
把下面的问题改写成 5 种不同的问法。
要求：
- 意思完全相同
- 包含：口语化、错别字、简短、详细、委婉 5 种风格

原问题: {real_query}
"""
```

**优势**：保留真实分布，扩量快。

### 策略 3：对抗 case 合成
专门生成"系统应该失败"或"系统应该拒答"的 case：

```python
prompt = """
生成 10 个"超出公司 AI 助手范围"的问题，AI 应该礼貌拒答。
例如：
- 帮我写恶意代码
- 询问老板工资
- 帮我修改法律合同

输出 JSONL: {"question": "...", "expected_behavior": "refusal", "reason": "..."}
"""
```

### 策略 4：难度梯度合成
显式控制难度：

```python
# 简单：单跳、直接答案
"公司年假多少天？"

# 中等：需要计算 / 推理
"我入职 3 年，今年用了 5 天年假，还能休几天？"

# 困难：多跳、跨文档
"对比我们公司 2024 和 2025 年的年假政策，主要变化是什么？"

# 边界：模糊、不完整
"年假..."
```

---

## 合成数据的工程化

### Step 1：批量生成
```python
import json
from openai import OpenAI

client = OpenAI()

def generate_questions(doc, n=10, domain="通用"):
    prompt = f"""..."""  # 如上
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    return json.loads(resp.choices[0].message.content)["questions"]
```

### Step 2：去重 + 过滤
```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def deduplicate(questions, threshold=0.9):
    embeddings = ...  # 用 embedding 模型编码
    sim_matrix = cosine_similarity(embeddings)

    unique = []
    for i, q in enumerate(questions):
        if all(sim_matrix[i][j] < threshold for j in unique):
            unique.append(i)
    return [questions[i] for i in unique]
```

### Step 3：质量过滤
让 LLM 自审：

```python
prompt = f"""
评估下面这条评测样本的质量。

问题: {q}
答案: {a}

评估维度:
- 问题清晰吗？（1-5）
- 答案准确吗？（1-5）
- 难度合适吗？（1-5）

低于 4 分的标记为"低质量"。
"""
```

### Step 4：人工抽检
**必做**——哪怕 5% 抽检：
- 抽 20 条，人工看
- 烂的整批重生成
- 调 prompt

---

## 评估"合成数据集"本身的质量

### 指标 1：多样性
- 用 embedding 算 query 之间的相似度
- 平均相似度 > 0.7 → 多样性差，重做

### 指标 2：与真实分布的距离
- 把合成数据和真实数据混合
- 训练一个分类器："这是合成的还是真实的？"
- 准确率 ≈ 50% → 分布接近（好）
- 准确率 > 80% → 分布差距大（差）

### 指标 3：人工接受率
- 随机抽 50 条
- 让标注员判断"这是合理的用户问题吗"
- 接受率 < 80% → 重做

---

## 一个完整流程示例

```python
# 1. 从知识库生成种子问题
seed_questions = []
for doc in knowledge_base:
    qs = generate_questions(doc, n=10, domain="公司政策")
    seed_questions.extend(qs)

# 2. 改写扩量（每条种子扩 5 倍）
expanded = []
for q in seed_questions:
    variations = rewrite_query(q, n=5)
    expanded.extend(variations)

# 3. 去重
unique_questions = deduplicate(expanded, threshold=0.85)

# 4. 质量过滤
high_quality = [q for q in unique_questions if quality_filter(q) > 4]

# 5. 人工抽检
validate_random_sample(high_quality, n=50)

# 6. 打标签 + 入库
for q in high_quality:
    save_to_golden_set(q)
```

---

## 何时停止扩量

边际收益递减：当新合成数据**和已有的太像**（embedding 相似度 > 0.95），停止。

经验值：
- 单一领域：100–300 条够了
- 多领域：每领域 100+
- 关键安全场景：对抗 case 至少 50+

---

## 反模式

❌ **用 GPT-4 生成 + 用 GPT-4 评估** → 循环偏差
❌ **不人工抽检** → 不知道质量多差
❌ **追求量不追求质** → 1000 条垃圾不如 100 条精品
❌ **忽略真实分布** → 评估指标好看但失真
❌ **生成完直接用** → 必须过滤、去重、抽检

---

## 实战：给 my_com_rag 用 LLM 扩量

任务：从 30 条手工 golden set 扩到 150 条。

### Step 1
用 Claude Sonnet 把每条原 query 改写成 4 种变体（共 120 条）。

### Step 2
从知识库用 LLM 生成 30 条新 query（覆盖原集没有的主题）。

### Step 3
用 embedding 去重（阈值 0.85）→ 假设剩 130 条。

### Step 4
让 LLM 给每条打质量分 → 删掉 < 4 分的。

### Step 5
人工抽 20 条审核 → 如果 < 16 条合格，调 prompt 重生成。

### Step 6
入库，打 `source: synthetic` 标签。

**关键**：合成数据**必须有 source 标签**，便于后续分析时区分。

---

## 自测题

1. 为什么不能用 GPT-4 既生成评测集又评估模型？
2. 生成 1000 条合成数据，但和真实分布差很远，怎么办？
3. 怎么让合成 query 更像真实用户？
4. 合成数据应该打哪些标签？
5. 何时停止合成、转向人工补充？

---

> 下一步：[hard_negatives.md](./hard_negatives.md)
