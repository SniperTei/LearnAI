# RAGAS：RAG 评估的事实标准

> RAGAS = Retrieval Augmented Generation Assessment。
> 它把"RAG 好不好"拆成 4 个可量化的指标。

---

## 为什么需要 RAGAS

一个 RAG 系统的输出：

```
用户问题:  Q
召回的文档: [D1, D2, D3]
生成的答案: A
```

你怎么评 A 好不好？

- 直接看 A 通顺不通顺？—— 可能 A 编造了，文档里没有
- 看 A 和标准答案像不像？—— 你要有标准答案，且不一定有唯一答案
- 看 A 引用 D1 了？—— 引用了不代表用对了

RAGAS 的思路：**用 LLM 当评委，分解成 4 个维度，每个维度独立打分**。

---

## 四大核心指标

### 1. Faithfulness（忠实度）— 答案有没有编造

> 给定召回的 context，答案里的每句话能否被 context 支持？

```
Q:        公司年假多少天？
Context:  "入职享 15 天年假，工龄 5 年以上享 20 天"
A:        "公司年假 15 天，3 年后增加到 20 天"  ← 错！3 年是编的
Faithfulness: 0.5（前半句被支持，后半句没有）
```

**为什么重要**：hallucination 是 RAG 最致命的问题。
Faithfulness 直接量化"编造率"。

### 2. Answer Relevancy（答案相关性）— 答案有没有跑题

> 答案是否真正回答了用户的问题？有没有废话？

```
Q:        公司年假多少天？
A:        "年假制度是公司福利的重要组成部分，旨在让员工休息..."
Answer Relevancy: 低（说了很多，但没回答"多少天"）
```

**实现方式**：让 LLM 根据 A 反向生成可能的 Q，看反推的 Q 和原 Q 相似度。

### 3. Context Precision（上下文精确率）— 召回的有没有用

> 召回的 N 个文档里，**真正有用的**占多少？排序好不好？

```
Q:        年假多少天？
召回:     [D1: 关于食堂菜单, D2: 年假政策, D3: 病假政策]
Context Precision: 1/3（只有 D2 相关，且没排在最前）
```

**为什么重要**：context 里噪音多 → 模型更容易跑偏 + 浪费 token。

### 4. Context Recall（上下文召回率）— 该召回的召回了没

> 回答这个问题需要的所有信息，context 里都有吗？

```
Q:        年假和病假各多少天？
标准答案:  年假 15 天，病假 10 天
召回:     [只有年假政策]
Context Recall: 0.5（缺病假）
```

**注意**：这个指标**需要标准答案**，是 RAGAS 里最"贵"的指标。

---

## 四个指标的解读

```
                  Faithfulness
                       ↑
                       │
        没编造 ────────┼──────── 编造多
                       │
                       ↓

Context Recall ─── [ RAG 系统 ] ─── Context Precision
   (召回全)              │              (召回准)
                        ↓
                 Answer Relevancy
                       ↑
                       │
        切题  ─────────┼──────── 跑题
```

**典型诊断**：

| 现象 | 可能问题 |
|------|---------|
| Faithfulness 低 | Prompt 没限制"只基于 context"，或模型能力不够 |
| Answer Relevancy 低 | Prompt 没强调"直接回答"，或召回不相关 |
| Context Precision 低 | 检索召回差，需要 reranking |
| Context Recall 低 | 知识库不全，或分块不当丢信息 |

---

## RAGAS 实战（Python 代码）

### 安装

```bash
pip install ragas datasets
```

### 最小示例

```python
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# 准备评估数据
eval_data = {
    "question": [
        "公司年假多少天？",
        "如何申请报销？",
    ],
    "answer": [
        "入职即享 15 天年假。",                              # 系统生成的答案
        "提交发票给财务即可。",                              # 系统生成的答案
    ],
    "contexts": [
        ["员工手册第3章：入职享 15 天年假"],                # 系统召回的文档
        ["财务规定：报销需附发票，5个工作日内提交"],         # 系统召回的文档
    ],
    "ground_truth": [                                       # 标准答案（context_recall 必须）
        "15 天",
        "提交发票给财务，5 个工作日内",
    ],
}

dataset = Dataset.from_dict(eval_data)

# 跑评估
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)

print(result)
# {'faithfulness': 0.95, 'answer_relevancy': 0.88, 'context_precision': 0.83, 'context_recall': 0.90}
```

### 用什么模型当 evaluator

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate

# 默认用 OpenAI，但你可以换
llm = ChatOpenAI(model="gpt-4o-mini")  # 便宜的 judge
embeddings = OpenAIEmbeddings()

result = evaluate(
    dataset,
    metrics=[...],
    llm=llm,
    embeddings=embeddings,
)
```

**推荐**：
- Judge 用 GPT-4o-mini 或 Claude Haiku（便宜）
- 关键评估用 GPT-4o / Claude Sonnet（更准）
- 中文场景多对比几个 judge，看一致性

---

## RAGAS 的坑

### 坑 1：评估本身很贵
- 每条样本 4 个指标，每个指标都调用 1 次 LLM
- 100 条样本 × 4 指标 ≈ 400 次 LLM 调用
- 用 GPT-4o 跑一次可能几十美元

**对策**：
- 平时用 GPT-4o-mini
- 关键版本发布前用 GPT-4o 跑一次
- 子采样：先跑 30 条看趋势，再扩到 100

### 坑 2：LLM-as-Judge 不一定准
- Faithfulness 偶尔会"看不出"明显编造
- Answer Relevancy 在主观题上波动大

**对策**：
- 用人工标注校准（见 [human_alignment.md](./human_alignment.md)）
- 多次评估取平均（temperature > 0 时）

### 坑 3：Context Recall 需要 ground truth
- 很多场景没有标准答案
- 强行造 ground truth 反而误导

**对策**：
- 没有 ground truth 时，跳过这个指标
- 改用 LLM-as-Judge 评估"答案完整性"

### 坑 4：指标好 ≠ 用户满意
- Faithfulness 0.95 但答案可能很啰嗦
- Context Precision 1.0 但排序可能依然不好

**对策**：
- RAGAS 是必要不充分条件
- 必须配在线反馈（见 [../online/user_feedback.md](../online/user_feedback.md)）

---

## 给 my_com_rag 加 RAGAS 的步骤

1. **造数据**：从你的知识库里，挑 50–100 个 Q-A 对作为 golden set（参考 [../golden_set_recipes/how_to_build.md](../golden_set_recipes/how_to_build.md)）
2. **跑系统**：用这套 Q 让系统生成 answer 和 contexts
3. **跑 RAGAS**：得到 4 个指标的基线
4. **改一处**：比如加 reranking，再跑一次
5. **对比**：看 Context Precision 提升了多少
6. **写报告**：量化改进 → 决策是否上线

---

## RAGAS 之外的其他选择

| 工具 | 特点 |
|------|------|
| **DeepEval** | Pythonic API，集成 pytest，适合 CI |
| **TruLens** | 强调"track 反馈"，更面向生产 |
| **LangSmith Eval** | LangChain 生态原生，无缝集成 |
| **Phoenix** | Arize 出品，可观测 + 评估一体 |

**选型建议**：
- 单纯评估 → RAGAS（最通用）
- 要 CI 集成 → DeepEval
- 已用 LangChain → LangSmith
- 要可观测一体化 → Phoenix

不必纠结，**先用起来 RAGAS**，下次有具体痛点再换。

---

## 自测题

1. RAGAS 4 指标里，哪个评估"检索质量"？哪个评估"生成质量"？
2. Faithfulness 0.6 意味着什么？下一步该改什么？
3. Context Recall 必须有什么才能算？
4. 为什么 RAGAS 不能替代用户反馈？
5. 你的 my_com_rag 上 RAGAS，估计跑 100 条要多少钱？

---

> 下一步：[llm_as_judge.md](./llm_as_judge.md) — RAGAS 的底层方法论
