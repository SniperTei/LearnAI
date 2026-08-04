# 向量库对比与选型

> FAISS、pgvector、Qdrant、Milvus、Chroma、Pinecone...
> 你现在用 FAISS——什么时候该换？

---

## 先想清楚：你需要向量库吗

不是所有 RAG 都需要"专门的向量库"。三类方案：

### 1. 纯内存（numpy / FAISS in-memory）
- 加载所有向量到内存
- 暴力搜索（brute force）
- 适合：< 10 万向量，PoC 阶段
- 你 `my_com_rag` 现在大概率是这个

### 2. 嵌入式（embedded in DB）
- pgvector（PostgreSQL 插件）
- SQLite + sqlite-vss
- 适合：和业务数据共库、中型项目

### 3. 专用向量数据库（purpose-built）
- Qdrant、Milvus、Weaviate、Pinecone（托管）
- 适合：百万-亿级、需要高并发、横向扩展

**FDE 决策原则**：从最简单的开始，**有明确瓶颈再升级**。

---

## 主流方案对比

| 方案 | 类型 | 部署 | 元数据过滤 | 适合规模 | 你的现状？ |
|------|------|------|-----------|---------|-----------|
| FAISS | 库 | 内存/文件 | ❌（需自己管） | < 100 万 | ✅ 用中 |
| Chroma | 嵌入式 DB | 单机 | ✅ | < 100 万 | — |
| pgvector | PG 插件 | 任意 | ✅✅（SQL） | < 1000 万 | — |
| Qdrant | 专用 | 单机/集群 | ✅✅ | 千万-亿 | — |
| Milvus | 专用 | 分布式 | ✅ | 亿+ | — |
| Pinecone | 托管 SaaS | 云 | ✅ | 任意 | — |
| Weaviate | 专用 | 单机/集群 | ✅ | 千万 | — |

### FAISS
**优点**：
- Meta 出品，性能极致
- 算法齐全（Flat / IVF / HNSW / PQ）
- 库形式，灵活

**缺点**：
- 不是数据库——增删改查要自己写
- 元数据过滤要自己拼
- 持久化要自己做（保存 .bin + .pkl，你 my_com_rag 就是）
- 没有并发管理

**何时该换**：
- 数据量上来后内存爆掉
- 需要频繁增删
- 需要按 metadata 过滤（如"只查 2024 年的合同"）

### pgvector
**优点**：
- **关系数据 + 向量一体**——这是 FDE 的杀手锏
- 用熟悉的 SQL 操作
- 元数据过滤、JOIN、事务都有
- 运维门槛低（会 PG 就行）

**缺点**：
- 性能不如专用库（百万级以内够用）
- 索引选择少（HNSW / IVFFlat）

**典型 FDE 场景**：
```sql
-- 业务过滤 + 向量检索一气呵成
SELECT id, content, embedding <=> $1 AS distance
FROM documents
WHERE tenant_id = $2          -- 多租户过滤
  AND created_at > '2024-01-01'
  AND department = 'legal'
ORDER BY embedding <=> $1
LIMIT 10;
```

**强烈推荐**：FDE 中型项目（< 1000 万向量）首选。

### Qdrant
**优点**：
- Rust 写，性能好、内存省
- payload 过滤强大
- API 友好（gRPC + REST）
- 单机版免运维
- 开源 + 托管都有

**缺点**：
- 比 pgvector 多一个组件要运维
- 中文社区不如 Milvus 大

**适合**：千万级以上、纯向量检索、性能敏感

### Milvus
**优点**：
- 国产，中文文档完善
- 分布式架构，能扩到亿级
- 支持多种索引、混合检索

**缺点**：
- 部署复杂（虽然 Milvus Lite 改善了）
- 资源占用大
- 学习曲线

**适合**：超大规模、企业级、国内项目

### Pinecone
**优点**：
- 完全托管，零运维
- 性能稳定
- 企业级 SLA

**缺点**：
- 贵
- 数据出境（国内合规问题）
- 供应商锁定

**适合**：海外项目、不想运维、预算充足

---

## 决策树

```
你的向量数 < 10 万？
├── 是 → FAISS in-memory（不换）
└── 否
    └── 你已经在用 PostgreSQL 吗？
        ├── 是 → pgvector（强烈推荐）
        └── 否
            └── 数据量级？
                ├── < 1000 万 → Qdrant（单机版）
                ├── 1000 万 - 1 亿 → Qdrant / Milvus
                └── > 1 亿 → Milvus 集群 / Pinecone

补充考虑：
- 国内合规 / 私有化 → Milvus / Qdrant 自部署
- 海外 / SaaS → Pinecone / Weaviate Cloud
- 多租户 → pgvector（用 SQL row-level security）
```

---

## 选型的真实考量

### 维度 1：运维成本
- FAISS：零运维（但你自己写持久化）
- pgvector：和现有 PG 一起运维
- Qdrant/Milvus：新增一个组件
- Pinecone：零运维（但要付费）

### 维度 2：业务集成
**关键问题**：你的向量检索需要和业务数据 JOIN 吗？
- 需要（如"查这个用户自己的文档"）→ pgvector
- 不需要（独立的知识库）→ Qdrant/Milvus

### 维度 3：性能要求
- 毫秒级延迟，单机就够 → pgvector / Qdrant 单机
- 高并发、海量 → Milvus / Qdrant 集群

### 维度 4：成本
- 开发自部署：FAISS < pgvector < Qdrant ≤ Milvus
- 托管：Pinecone > 其他云方案

### 维度 5：生态
- 你已经用 LangChain？所有方案都支持
- 你用 LlamaIndex？同上
- 你用 Postgres 生态？pgvector

---

## 何时从 FAISS 迁移

你 `my_com_rag` 现在用 FAISS（看到 `faiss_index.bin` + `tfidf_matrix.pkl`）。
迁移信号：

### 信号 1：内存压力
向量数 > 内存 70% → 系统不稳定

### 信号 2：增删改频繁
FAISS 增删要么重建索引，要么维护 ID 映射——都麻烦。
如果经常更新知识库 → 该换了。

### 信号 3：需要元数据过滤
```python
# FAISS 没有原生 metadata 过滤
# 你必须：
1. 向量检索 top-K * 5
2. 在 Python 里按 metadata 过滤
3. 取前 K 个
# 后果：召回不准、性能差
```

### 信号 4：多租户
不同租户的数据要隔离 → FAISS 几乎做不到优雅。

### 信号 5：并发
多个请求同时查询 → FAISS 的并发管理自己写。

**触发任一信号 → 优先考虑 pgvector**。

---

## pgvector 迁移示例

```sql
-- 1. 启用扩展
CREATE EXTENSION vector;

-- 2. 建表
CREATE TABLE documents (
    id BIGSERIAL PRIMARY KEY,
    content TEXT,
    embedding VECTOR(1536),
    tenant_id BIGINT,
    source TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. 建 HNSW 索引
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- 4. 检索（带元数据过滤）
SELECT id, content, 1 - (embedding <=> $1) AS similarity
FROM documents
WHERE tenant_id = $2
  AND source = 'hr_handbook'
ORDER BY embedding <=> $1
LIMIT 10;
```

```python
# Python 端
import psycopg
from pgvector.psycopg import register_vector

conn = psycopg.connect(...)
register_vector(conn)

query_emb = get_embedding("年假多少天")
results = conn.execute("""
    SELECT id, content FROM documents
    WHERE tenant_id = %s
    ORDER BY embedding <=> %s
    LIMIT 10
""", (tenant_id, query_emb)).fetchall()
```

**迁移工作量**：1–2 天（含测试）。
**收益**：
- 元数据过滤变 SQL（一行）
- 事务、备份、复制都白送
- 多租户天然支持

---

## 不要做的事

❌ **一开始就上 Milvus 集群**——过度工程化
❌ **把所有数据塞向量库**——业务数据放 PG，向量和业务 JOIN 才自然
❌ **不评估就换**——迁移成本不小，先确认有瓶颈
❌ **混用多家向量库**——除非有特殊原因
❌ **忽视索引选择**（HNSW vs IVF）——参数影响性能 10x
❌ **不监控检索延迟**——P95 慢了用户直接感知

---

## 实战：给 my_com_rag 评估是否迁移

### Step 1：测现状
- 当前向量数：?
- 内存占用：?
- 单查询延迟 P50/P95：?
- 增删改频率：?

### Step 2：判断瓶颈
- 上面 5 个信号命中几个？
- 命中 0 → 不动，做别的优化
- 命中 1–2 → 可以做 pgvector 迁移
- 命中 3+ → 必须迁移

### Step 3：迁移试点
- 用 pgvector 做一个并行版本
- 跑同一份 golden set
- 对比性能 + 准确率

### Step 4：决策
- 性能/准确率持平或更好 → 切换
- 否则 → 留在 FAISS，等真有瓶颈再说

**产出**：一份"向量库选型报告"——FDE 简历素材。

---

## 自测题

1. FAISS 现在能满足你的需求吗？什么信号出现时该换？
2. pgvector 相比 Qdrant，最大优势是什么？
3. 你需要"按部门 + 时间过滤"地查向量，FAISS 怎么做？pgvector 怎么做？
4. 多租户 SaaS，你会选哪个向量库？
5. 千万级数据，预算紧，国内私有化，选什么？

---

> 下一步：[failure_modes.md](./failure_modes.md) — RAG 为什么会失败
