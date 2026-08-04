# 混合检索（Hybrid Search）

> 单一向量检索有盲区，单一关键词检索有盲区。
> **混合**才是生产级 RAG 的标配。

---

## 两种检索的本质差异

### 向量检索（Dense Retrieval）
- 用 embedding 找"语义相近"
- 优势：理解同义、改写、跨语言
- 弱点：**对精确关键词、专有名词、数字不敏感**

```
query: "iPhone 15 Pro Max 256GB 价格"
向量召回: 可能召回所有 iPhone 相关文档（不准确）
```

### 关键词检索（Sparse Retrieval / BM25）
- 按词频匹配
- 优势：精确匹配关键词、专有名词、代码
- 弱点：**不理解同义、改写**

```
query: "如何请假"
BM25 召回: 文档里必须有"如何"或"请假"
          如果文档写的是"申请休假"→ 召回不到
```

### 各自的盲区
- 向量：专有名词、精确数字、代码标识符
- BM25：同义表达、改写、跨语言

**结论**：两者互补。生产 RAG **必须**混合。

---

## 混合检索的 3 种方式

### 方式 1：分数融合（Score Fusion）
```python
# 各自检索 top-K
dense_results = vector_search(query, k=20)   # [(doc_id, score), ...]
sparse_results = bm25_search(query, k=20)

# 归一化分数到 [0, 1]
def normalize(results):
    if not results: return []
    max_score = max(s for _, s in results)
    return [(d, s/max_score) for d, s in results]

dense_results = normalize(dense_results)
sparse_results = normalize(sparse_results)

# 加权融合
alpha = 0.5  # 各占一半
fused = {}
for doc_id, score in dense_results:
    fused[doc_id] = fused.get(doc_id, 0) + alpha * score
for doc_id, score in sparse_results:
    fused[doc_id] = fused.get(doc_id, 0) + (1-alpha) * score

# 排序取 top-K
final = sorted(fused.items(), key=lambda x: -x[1])[:10]
```

### 方式 2：RRF（Reciprocal Rank Fusion）⭐ 推荐
不看分数，只看排名——更鲁棒：

```python
def rrf(dense_results, sparse_results, k=60):
    """
    k 是平滑常数，常用 60。
    公式: score(doc) = sum(1 / (k + rank(doc)))
    """
    fused = {}
    for rank, (doc_id, _) in enumerate(dense_results, 1):
        fused[doc_id] = fused.get(doc_id, 0) + 1 / (k + rank)
    for rank, (doc_id, _) in enumerate(sparse_results, 1):
        fused[doc_id] = fused.get(doc_id, 0) + 1 / (k + rank)
    return sorted(fused.items(), key=lambda x: -x[1])
```

**为什么 RRF 更好**：
- 不用调权重（alpha）
- 对分数尺度不敏感
- 业界标准（Elasticsearch 默认就用）

### 方式 3：训练的融合模型
用 LightGBM / LambdaMART 学习"最佳融合"。
- 优势：效果最好
- 代价：需要训练数据 + 维护成本

**FDE 默认选 RRF**。

---

## BM25 实现

### 用 rank-bm25
```python
from rank_bm25 import BM25Okapi

# 1. 准备文档（中文要分词）
import jieba
tokenized_docs = [list(jieba.cut(doc)) for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

# 2. 查询
query_tokens = list(jieba.cut(query))
scores = bm25.get_scores(query_tokens)

# 3. 排序
ranked = sorted(enumerate(scores), key=lambda x: -x[1])[:20]
sparse_results = [(doc_ids[i], score) for i, score in ranked]
```

### 用 Elasticsearch / OpenSearch
生产推荐——支持亿级、分布式、原生 BM25 + 向量混合：
```json
{
  "query": {
    "hybrid": {
      "queries": [
        { "match": { "content": "query text" } },
        { "knn": { "embedding": { "vector": [...], "k": 10 } } }
      ]
    }
  }
}
```

### 用 PG + pgvector + 全文检索
```sql
-- 同时用 ts_vector (全文检索) 和 pgvector
SELECT id, content,
    ts_rank(tsv, plainto_tsquery('chinese', $1)) AS bm25_score,
    embedding <=> $2 AS vector_distance
FROM documents
ORDER BY
    -- 简化的融合（实际用 RRF 更好）
    (1 - (embedding <=> $2)) + ts_rank(tsv, plainto_tsquery('chinese', $1)) DESC
LIMIT 10;
```

---

## 完整 Hybrid 检索代码

```python
from rank_bm25 import BM25Okapi
import jieba
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class HybridRetriever:
    def __init__(self, documents: list[str], embed_fn):
        self.documents = documents
        self.embed_fn = embed_fn

        # 1. 建 BM25
        tokenized = [list(jieba.cut(d)) for d in documents]
        self.bm25 = BM25Okapi(tokenized)

        # 2. 建向量索引（这里用内存）
        self.doc_embs = np.array([embed_fn(d) for d in documents])

    def search(self, query: str, k: int = 10, alpha: float = 0.5) -> list[tuple[int, float]]:
        # 向量检索
        q_emb = self.embed_fn(query)
        dense_sims = cosine_similarity([q_emb], self.doc_embs)[0]
        dense_ranked = np.argsort(-dense_sims)[:k*3]  # 多召回点用于融合
        dense_results = [(i, dense_sims[i]) for i in dense_ranked]

        # BM25 检索
        tokens = list(jieba.cut(query))
        bm25_scores = self.bm25.get_scores(tokens)
        bm25_ranked = np.argsort(-bm25_scores)[:k*3]
        sparse_results = [(i, bm25_scores[i]) for i in bm25_ranked]

        # RRF 融合
        return self._rrf(dense_results, sparse_results, k=k)

    def _rrf(self, dense, sparse, k=10, c=60):
        fused = {}
        for rank, (idx, _) in enumerate(dense, 1):
            fused[idx] = fused.get(idx, 0) + 1 / (c + rank)
        for rank, (idx, _) in enumerate(sparse, 1):
            fused[idx] = fused.get(idx, 0) + 1 / (c + rank)
        return sorted(fused.items(), key=lambda x: -x[1])[:k]


# 用法
retriever = HybridRetriever(docs, embed_fn)
top_k = retriever.search("iPhone 15 Pro Max 价格", k=5)
```

---

## 什么时候最受益

| 场景 | Hybrid 提升幅度 |
|------|----------------|
| 含专有名词、产品名、代码 | **高** |
| 精确数字、日期查询 | **高** |
| 模糊、口语化 query | 中 |
| 跨语言查询 | 中 |
| 通用问答 | 低 |

**经验**：你的 RAG 上 Hybrid，RAGAS Context Precision 通常提升 5–15%。

---

## 调参

### RRF 的 `c` 参数
- c=60：业界默认
- c 越大 → 排名差距越小（更平均）
- c 越小 → 头部排名权重更大

一般不用调。

### 加权 alpha（如果用 score fusion）
```python
alpha = 0.5  # 各半
alpha = 0.7  # 偏向向量（语义为主）
alpha = 0.3  # 偏向关键词
```
**怎么选**：在 golden set 上扫 0.3 / 0.5 / 0.7，看哪个最优。

### 召回数 K
- 单路召回 K' = 2K 到 3K
- 融合后取 top-K
- K 太小：融合没意义；K 太大：成本高

---

## 中文分词的关键性

BM25 效果**强依赖分词**。

```python
# 不好
list("年假政策") → ['年', '假', '政', '策']  # 字符切分

# 好
list(jieba.cut("年假政策")) → ['年假', '政策']  # 词级切分
```

**对策**：
- 用 jieba（默认）
- 用 HanLP / pkuseg（更准但慢）
- **加自定义词典**（业务词汇！）
  ```python
  jieba.load_userdict("company_terms.txt")
  # company_terms.txt:
  #   阿里云 10 n
  #   年假政策 5 n
  ```

**FDE 必做**：把你的业务专有名词加到分词词典，BM25 效果立竿见影。

---

## 反模式

❌ **只调向量**——专有名词场景必崩
❌ **只调 BM25**——同义改写场景必崩
❌ **用 score fusion 不归一化**——分数尺度不一致
❌ **不分词直接 BM25**（中文按字符）→ 退化
❌ **不调 alpha 就上线**→ 默认 0.5 不一定最优
❌ **召回数 K 不够大**→ 融合没东西可融

---

## 实战：给 my_com_rag 加 Hybrid

你现在的 `tfidf_matrix.pkl` 其实是一种稀疏检索（TF-IDF 比 BM25 弱）。
升级路径：

### Phase 1：替换 TF-IDF → BM25
```python
# 旧：TF-IDF + cosine
# 新：BM25（分词 + rank_bm25）
```
**预期**：召回精度小幅提升。

### Phase 2：加 RRF 融合
- 向量检索 top-30
- BM25 top-30
- RRF 融合 → top-10
**预期**：Context Precision 提升 5–10%。

### Phase 3：业务词典
- 收集 50–100 个公司专有名词
- 加到 jieba 词典
**预期**：BM25 进一步提升。

### Phase 4（可选）：迁移到 Elasticsearch / pgvector
- 当数据量 > 100 万
- 用现成的 hybrid search 支持

---

## 自测题

1. query = "Q3 销售报告"，纯向量检索可能召回什么无关内容？
2. RRF 相比 score fusion，主要优势是什么？
3. 中文 BM25 为什么必须分词？
4. 你的业务有 50 个专有名词，怎么让 BM25 认识它们？
5. 向量 + BM25 + Reranking，三者顺序是什么？

---

> 下一步：[reranking.md](./reranking.md)（强烈推荐——RAG 性价比最高的优化）
