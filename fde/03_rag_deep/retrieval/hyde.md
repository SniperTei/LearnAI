# HyDE：假设答案检索

> HyDE = Hypothetical Document Embeddings
> 思路反转：**先让 LLM 答题，再用它的答案去检索**。

---

## 直觉

普通 RAG：
```
query → embedding → 检索文档 → 给 LLM → 答案
```

**问题**：query 通常很短（"年假多少天"），文档通常很长。
短 query 的 embedding 和长文档的 embedding 在向量空间里**不太对位**——
即使文档里有答案，相似度也不一定高。

### HyDE 的反思路
```
query → LLM 编一个假设答案 → 用假设答案 embedding → 检索文档
```

**为什么有效**：假设答案和真实文档**长度相近、风格相近**，
向量空间里更对位。

---

## 流程图

```
用户: "公司年假多少天？"

LLM 生成假设答案（可能编造）:
    "我们公司年假政策是入职享 15 天，工龄 5 年以上 20 天..."
    (数字可能是错的，但风格像真实文档)

假设答案 → embedding → 检索

召回的文档（现在更准了）:
    "员工手册 v3.2: 入职享 15 天年假..."

最终：LLM 基于真实文档回答
```

**关键**：假设答案**不需要准确**——它只是"诱饵"。
它越像真实文档，召回越准。

---

## 最小实现

```python
from openai import OpenAI

client = OpenAI()

def hyde_retrieve(query: str, retriever, n_docs: int = 5, n_hyp: int = 1):
    """HyDE 检索"""
    # Step 1: 生成假设答案
    hypothetical_docs = generate_hypothetical(query, n=n_hyp)

    # Step 2: 用假设答案的 embedding 检索
    # 可以用任一假设，或平均多个假设
    combined = "\n".join(hypothetical_docs)
    return retriever.search_by_text(combined, k=n_docs)


def generate_hypothetical(query: str, n: int = 1) -> list[str]:
    """让 LLM 生成 N 个假设答案。"""
    prompt = f"""请基于下面的问题，写一段可能的答案（{n} 个不同版本）。
不需要准确，但风格要像公司文档。

问题: {query}

输出（每段一行）:
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",  # 便宜的模型即可
        messages=[{"role": "user", "content": prompt}],
        n=n,
        temperature=0.7,  # 高温多样
    )
    return [c.message.content for c in resp.choices]
```

---

## 什么时候 HyDE 有用

### 适合
- **短 query**（"年假"、"报销"）→ 假设答案提供更多上下文
- **query 和文档风格差异大**（用户口语化，文档正式）
- **跨语言**（query 中文，文档英文，假设答案可统一风格）

### 不适合
- **事实精确查询**（"iPhone 15 价格"）→ 假设答案可能误导
- **多跳推理**（"对比 A 和 B"）→ 假设答案无法覆盖
- **对延迟敏感**（多 1 次 LLM 调用，延迟翻倍）
- **冷启动模型**（LLM 完全不熟悉领域时，编的答案太离谱）

---

## HyDE 的代价

### 成本
- 每次查询多 1 次 LLM 调用
- 用便宜模型（Haiku / 4o-mini）成本可控

### 延迟
- 多 200ms–1s
- 对实时场景影响大

### 风险
- LLM 编的假设答案可能完全错（hallucination）
- 如果假设答案太离谱，反而召回更差

**对策**：
- 用 cheap 模型 + 高 temperature（要风格多样，不要"正确"）
- 把假设答案和原 query 都用上（双路检索融合）

---

## 进阶：Multi-HyDE

生成多个不同版本的假设，分别检索，融合结果：

```python
def multi_hyde(query, retriever, n_hyp=3):
    # 1. 生成 N 个假设答案
    hyps = generate_hypothetical(query, n=n_hyp)

    # 2. 每个假设检索
    all_results = []
    for hyp in hyps:
        results = retriever.search_by_text(hyp, k=20)
        all_results.append(results)

    # 3. 用原 query 也检索（兜底）
    all_results.append(retriever.search_by_text(query, k=20))

    # 4. RRF 融合
    return rrf_fusion(all_results, k=5)
```

**优势**：多个假设降低单次 LLM 出错的概率。

---

## HyDE vs Query Rewriting

| | HyDE | Query Rewriting |
|---|------|-----------------|
| 改什么 | 生成"假设答案" | 改写 query 本身 |
| 输出 | 长文本（像文档） | 短文本（像 query） |
| 用途 | 检索 | 检索 + 澄清 |
| 成本 | 中 | 低 |
| 适合 | 短 query、风格差异 | 模糊 query、口语化 |

**可同时用**：先用 query rewrite 澄清意图，再 HyDE 检索。

---

## 实测 HyDE 效果

论文报告：
- 长 query：提升 0–3%（不明显）
- 短 query：提升 5–15%
- 跨语言：提升 10–20%

**典型场景**：用户用关键词搜索时（"年假"、"报销"），HyDE 收益最大。

---

## 反模式

❌ **用强模型生成 HyDE**（GPT-4）→ 浪费钱，弱模型就够
❌ **只信假设答案**，不召回真实文档 → 编造
❌ **盲目 HyDE 所有 query** → 长 query 反而可能变差
❌ **不评估就上 HyDE** → 不知道有没有用
❌ **忽略延迟成本** → 用户体验差

---

## 实战：给 my_com_rag 试 HyDE

### Step 1：找适合的 case
从你的 golden set 里，挑出**短 query**（< 10 字）的子集。

### Step 2：基线对比
```python
short_queries = [q for q in golden_set if len(q.question) < 10]

# 基线（不带 HyDE）
score_base = ragas_eval(short_queries)

# HyDE
score_hyde = ragas_eval_with_hyde(short_queries)

print(f"Context Recall: {score_base.cr:.3f} → {score_hyde.cr:.3f}")
```

### Step 3：决策
- 短 query 子集显著提升 → 全量加 HyDE（或仅对短 query 用）
- 不显著 → 跳过 HyDE，做别的优化

---

## 自测题

1. HyDE 的核心思路是什么？为什么有效？
2. 长 query 用 HyDE 有意义吗？
3. HyDE 多了一次 LLM 调用，怎么控制成本？
4. 假设答案编错了，会影响最终答案吗？
5. HyDE 和 Rerank 顺序是什么？

---

> 下一步：[multi_query.md](./multi_query.md)
