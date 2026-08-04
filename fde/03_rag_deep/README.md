# 03 RAG Deep

> 你已经会做 RAG 了。这里的目标是**懂每个环节的 trade-off**，
> 这样在客户现场才能拍板"为什么用 A 不用 B"。

## 学完应该能回答

- 你的 chunk 大小是 512，为什么不是 256 或 1024？
- 什么时候用语义检索，什么时候用 BM25，什么时候混合？
- reranking 到底为什么有效？加它会带来什么代价？
- 不同 embedding 模型在中文场景表现差异有多大？
- 客户问"我把 100 万份合同塞进去，能跑吗"——你怎么回答？

## 待写笔记

### 顶层
- [ ] `chunking_strategies.md` — 固定 / 递归 / 语义 / 文档感知分块对比
- [ ] `embedding_theory.md` — embedding 到底编码了什么？维度、模型对比
- [ ] `vector_db_comparison.md` — FAISS / pgvector / Qdrant / Milvus 选型
- [ ] `failure_modes.md` — RAG 为什么会失败：5 大典型模式

### `chunking/`
- [ ] `fixed_vs_recursive.md` — 实测代码
- [ ] `semantic_chunking.md` — 用 LLM 分块的代价
- [ ] `document_aware.md` — Markdown / PDF / HTML 分块注意点

### `retrieval/`
- [ ] `hybrid_search.md` — BM25 + 向量融合
- [ ] `reranking.md` — bge-reranker / Cohere Rerank 实测
- [ ] `hyde.md` — Hypothetical Document Embeddings
- [ ] `multi_query.md` — 多查询融合策略

## 实战任务（全部在 my_com_rag 上做）

- [ ] 把 chunk size 从 512 改成 256/1024，跑 RAGAS 对比
- [ ] 加 bge-reranker，看 faithfulness 提升
- [ ] 把 FAISS 换成 pgvector，记录迁移成本
- [ ] 故意造 10 个"边界 case"，看 RAG 失败长什么样

## 参考资源

- Pinecone 的 RAG 系列博客
- Eugen Yan 博客：RAG evaluation
- "Searching for Best RAG Architecture" 综述
- BGE / E5 / Jina embedding 模型卡
