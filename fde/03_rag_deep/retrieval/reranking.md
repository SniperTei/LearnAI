# Reranking：RAG 性价比最高的优化

> 如果 RAG 只能做一件事来提升效果，**加 reranking**。
> 这是 FDE 必备、必懂、必用的武器。

---

## Reranking 是什么

两阶段检索：

```
Stage 1: 召回（Recall）    ─ 快、便宜、量大
    向量检索 top-100

Stage 2: 重排（Rerank）    ─ 慢、贵、精准
    Cross-encoder 把 100 个重新打分，取 top-10
```

**目的**：用快速检索保证召回率，用精准模型保证精度。

---

## 为什么需要 Reranking

### 向量检索的根本问题
embedding 模型是 **Bi-encoder**：
```
query → embedding_q
doc   → embedding_d
相似度 = cosine(embedding_q, embedding_d)
```

**问题**：query 和 doc 各自独立编码，**没有交互**。
- 无法捕捉 query 和 doc 之间的细微关系
- 容易召回"主题相近但具体不对"的文档

### Reranking 用 Cross-encoder
```
[query, doc] 一起送进模型 → 一个分数
```

- query 和 doc **在 attention 层完全交互**
- 精度比 bi-encoder 高一个量级
- 但慢一个量级（每对都要前向）

### 类比
- 召回：图书馆管理员按主题找 100 本相关书（快、粗）
- Reranking：你自己逐本翻一遍，挑出最相关的 10 本（慢、精）

**两阶段是检索系统的标准做法**，RAG 只是其中一例。

---

## Reranking 的实测效果

通常数据：
- Context Precision：+10% 到 +25%
- Faithfulness：+5% 到 +15%
- 用户体感：明显

**ROI 极高**——成本中等，效果显著。

---

## Reranking 模型选型

### 开源（自部署）
| 模型 | 特点 |
|------|------|
| **BGE-reranker-large** | 中文 SOTA，推荐 |
| **BGE-reranker-v2-m3** | 多语言，新版更强 |
| Jina reranker-v2 | 英文强，多语言 |
| Cohere reranker（API） | 闭源但便宜 |

**强烈推荐**：BGE-reranker-v2-m3，中文场景的默认选择。

### 闭源（API）
| 服务 | 特点 |
|------|------|
| Cohere Rerank 3 | 闭源里最强 |
| Voyage Rerank | Anthropic 推荐 |
| Jina Rerank API | 便宜 |

### 大模型当 Reranker
直接让 GPT-4 / Claude 给文档打分（0-10）"和 query 有多相关"。
- 优点：不用训练，零代码
- 缺点：慢、贵
- 适用：PoC、冷启动

---

## 最小实现：用 BGE-reranker

```python
from FlagEmbedding import FlagReranker

# 加载模型（首次会下载，约 1GB）
reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)

def rerank(query: str, documents: list[str], top_k: int = 5) -> list[tuple[int, float]]:
    """返回 [(原索引, 分数), ...]"""
    pairs = [[query, doc] for doc in documents]
    scores = reranker.compute_score(pairs, normalize=True)

    # 单个 doc 时 compute_score 返回 float，需要统一
    if isinstance(scores, float):
        scores = [scores]

    ranked = sorted(enumerate(scores), key=lambda x: -x[1])
    return ranked[:top_k]

# 用法
query = "年假多少天"
candidates = retriever.search(query, k=20)  # 召回 20 个

# 重排
ranked = rerank(query, [c.content for c in candidates], top_k=5)

# 取 top-5 给 LLM
top_chunks = [candidates[i] for i, _ in ranked]
```

---

## 完整 RAG 流程（含 Hybrid + Rerank）

```python
def rag_retrieve(query: str, k: int = 5) -> list[Chunk]:
    # Stage 1: Hybrid 召回（宽口径，多召回）
    candidates = hybrid_retriever.search(query, k=50)  # 50 个

    # Stage 2: Rerank 精排
    ranked = reranker.rank(
        query=query,
        documents=[c.content for c in candidates],
        top_k=k,
    )

    # Stage 3: 返回 top-k 给 LLM
    return [candidates[i] for i, _ in ranked]
```

**关键**：召回要宽（top-50），rerank 后取 top-5。
如果召回 top-5 直接 rerank → 已经丢失了潜在的相关文档。

---

## Reranking 调参

### 召回数 K1
- 推荐：50–100
- 太小（< 20）：相关文档可能没召回
- 太大（> 200）：rerank 成本高

### 最终返回数 K2
- 推荐：3–5
- 太多（> 10）：context 长、噪音多、Lost in Middle
- 太少（1–2）：信息不全

### batch size
rerank 模型可以批处理，但单次 batch 太大会 OOM：
```python
# 分批跑
for i in range(0, len(pairs), batch_size):
    batch_scores = reranker.compute_score(pairs[i:i+batch_size])
```

---

## Reranker 的成本

### 自部署（BGE-reranker）
- CPU：单 query 50 docs ≈ 200ms–1s
- GPU：单 query 50 docs ≈ 20–50ms
- 内存：1–2GB

### API（Cohere 等）
- $2–$5 / 1k searches（每次 search 50 个候选）
- 中等流量项目可接受

### 大模型当 reranker（不推荐生产）
- 50 个候选 × 1 次 LLM 调用 = 慢且贵
- 仅 PoC 用

---

## 什么时候可以不上 Rerank

- **数据量极小**（< 100 chunks）：直接 LLM 全看
- **召回已经极准**（罕见）
- **延迟极敏感**（< 200ms SLA）

99% 的生产 RAG 都该上。

---

## 进阶：Late Interaction（ColBERT）

ColBERT 是介于 bi-encoder 和 cross-encoder 的方案：
- 每个 token 都有 embedding（不是单一向量）
- 相似度用 max-pool 算

精度接近 cross-encoder，速度接近 bi-encoder。
BGE-m3 就包含 ColBERT 功能。

**适用**：超大规模、对性能要求高的场景。

---

## 反模式

❌ **召回 top-5 直接 rerank** → 召回少 = rerank 无意义
❌ **用大模型当 reranker 上生产** → 太慢
❌ **不分场景用英文 reranker** → 中文场景效果差
❌ **Rerank 后返回 top-20** → context 太长
❌ **没评估就加 rerank** → 不知道是否真有用

---

## 实战：给 my_com_rag 加 Rerank

### Step 1：装 BGE-reranker
```bash
pip install FlagEmbedding torch
```

### Step 2：加到检索流程
```python
# 之前：vector_search → top-5 → LLM
# 之后：hybrid_search → top-50 → rerank → top-5 → LLM
```

### Step 3：基线对比
```python
# 不带 rerank
score_no_rerank = ragas_eval(golden_set)

# 带 rerank
score_with_rerank = ragas_eval(golden_set)

# 对比
print(f"Context Precision: {score_no_rerank.cp:.3f} → {score_with_rerank.cp:.3f}")
print(f"Faithfulness: {score_no_rerank.faith:.3f} → {score_with_rerank.faith:.3f}")
```

### Step 4：延迟 / 成本评估
```python
import time
start = time.time()
results = rag_retrieve(query)
print(f"延迟: {(time.time()-start)*1000:.0f}ms")
```

### Step 5：决策
- 提升显著 + 延迟可接受 → 上线
- 提升显著但延迟超 SLA → 优化（GPU、缓存）

**产出**：
- 一份对比报告（rerank 前后的 RAGAS + 延迟 + 成本）
- 这是 FDE 简历上的"硬指标"

---

## 自测题

1. Bi-encoder 和 Cross-encoder 的核心区别是什么？
2. 为什么召回 top-5 直接 rerank 没意义？
3. Reranker 上 GPU 和 CPU，性能差多少？
4. 你的 RAG 加 rerank 后，Faithfulness 没提升，可能什么原因？
5. Reranking 和 Hybrid Search 都该用吗？顺序是？

---

> 下一步：[hyde.md](./hyde.md)
