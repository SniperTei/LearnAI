# Multi-Query 检索

> 一次 query 不够，**让 LLM 帮你想 N 种问法**，并行检索融合。

---

## 直觉

用户问的问题，**可能不是最佳检索 query**。

```
用户: "怎么请年假"
文档里写的: "申请年假流程"、"休假申请"、"假期审批"
```

用户用一个表达，文档里可能有 N 种表达。
单一 query 召回会漏。

**Multi-Query 的解法**：让 LLM 把 query 改写成 N 个变体，分别检索，融合结果。

---

## 流程

```
原 query: "怎么请年假"
    ↓
LLM 改写:
    1. "如何申请年假"
    2. "年假申请流程"
    3. "休假审批步骤"
    4. "假期申请需要哪些材料"
    ↓
并行检索:
    每个 query 都 top-20
    ↓
融合（RRF / 去重 / Rerank）
    ↓
最终 top-5
```

---

## 最小实现

```python
from openai import OpenAI

client = OpenAI()

def multi_query_retrieve(query: str, retriever, n_queries: int = 4, k: int = 5):
    # Step 1: 生成 N 个变体
    queries = generate_query_variants(query, n=n_queries)
    queries = [query] + queries  # 加上原 query 兜底

    # Step 2: 并行检索
    all_results = []
    for q in queries:
        results = retriever.search(q, k=20)
        all_results.append(results)

    # Step 3: RRF 融合
    fused = rrf_fusion(all_results, k=k * 3)

    # Step 4: Rerank（推荐）
    final = rerank(query, fused[:30], top_k=k)
    return final


def generate_query_variants(query: str, n: int = 3) -> list[str]:
    prompt = f"""请把下面的问题改写成 {n} 个不同的问法。
要求：
- 意思相同
- 表达多样（口语化、书面、关键词等）
- 用文档可能使用的术语

原问题: {query}

输出（每行一个，不要编号）:
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    lines = resp.choices[0].message.content.strip().split("\n")
    return [l.strip() for l in lines if l.strip()][:n]


def rrf_fusion(results_list, k=60):
    fused = {}
    for results in results_list:
        for rank, (doc_id, _) in enumerate(results, 1):
            fused[doc_id] = fused.get(doc_id, 0) + 1 / (k + rank)
    return sorted(fused.items(), key=lambda x: -x[1])
```

---

## LangChain 等价

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
)
# 内部自动生成多 query + 融合
docs = retriever.invoke("怎么请年假")
```

---

## 何时有效

### 适合
- **query 和文档术语差异大**（用户用词不规范）
- **多主题查询**（"对比 A 和 B" → 拆成"A 是什么" + "B 是什么"）
- **跨表达场景**（口语 vs 书面）
- **召回率低**的 query 类型

### 不适合
- **精确查询**（"iPhone 15 价格"）→ 改写反而引入噪音
- **延迟敏感**（多 N 次检索）
- **强主题 query**（用户词很标准）

---

## 调参

### n_queries（变体数）
- 推荐：3–5
- 太少：覆盖不够
- 太多：成本翻倍，边际收益小

### 每路召回数
- 推荐：20–50
- 太少：融合没意义
- 太多：成本高

### 是否用原 query
**强烈建议加**：把原 query 作为其中一路，避免 LLM 改写偏了反而漏召回。

### 是否加 Rerank
**强烈建议加**：Multi-Query 后候选更多，需要 rerank 来精排。

---

## Multi-Query vs RAG-Fusion

| | Multi-Query | RAG-Fusion |
|---|-------------|------------|
| 改写 | 同义问法 | 同义 + 视角 |
| 融合 | RRF | RRF |
| 复杂度 | 简单 | 中 |
| 效果 | 稳定提升 | 复杂 query 更好 |

RAG-Fusion 是 Multi-Query 的"增强版"，思路一样。

---

## 进阶变体

### 1. 分解式 Query（Decomposition）
复杂问题拆成子问题：

```
原: "对比公司两款主力产品"
拆:
    1. "产品 A 的特点是什么"
    2. "产品 B 的特点是什么"
    3. "A 和 B 的差异在哪"
```

每个子问题独立检索 + 答案，最后综合。

### 2. Step-back Prompting
先问更"宽泛"的问题，再问具体：

```
原: "1990 年湖人队教练的儿子的大学"
Step-back: "1990 年湖人队教练是谁"
然后: "X 的儿子上的什么大学"
```

RAG 用 step-back query 检索，召回更全的背景。

### 3. 自适应 Query（Adaptive）
让 LLM 自己决定：
- 简单 query → 直接检索
- 复杂 query → 拆分
- 模糊 query → 反问 / 多 query

需要 Agent 模式（见 [../../04_agents/](../../04_agents/)）。

---

## Multi-Query + HyDE + Rerank 组合

完整现代 RAG 检索栈：

```python
def modern_retrieve(query, retriever):
    # 1. Multi-Query：生成 N 个变体
    queries = [query] + generate_query_variants(query, n=3)

    # 2. 对每个 query 做 HyDE 检索
    all_results = []
    for q in queries:
        hyde_results = hyde_retrieve(q, retriever, k=20)
        direct_results = retriever.search(q, k=20)
        all_results.append(hyde_results)
        all_results.append(direct_results)

    # 3. Hybrid（向量 + BM25）已经在前一步用了

    # 4. RRF 融合所有路
    fused = rrf_fusion(all_results, k=100)

    # 5. Rerank 取 top-5
    return rerank(query, fused[:50], top_k=5)
```

**这是工业级 RAG 检索的标准范式**——也是 Claude / OpenAI 等大厂内部用的方法。

**代价**：
- 延迟：1–2s（多步）
- 成本：单 query 0.01–0.05 美元
- **不能每次都跑全套**——要有缓存 + 路由

---

## 反模式

❌ **永远 Multi-Query**→ 简单 query 浪费钱
❌ **不加原 query 兜底**→ LLM 改写偏了就全错
❌ **不加 Rerank**→ 候选多了反而噪音多
❌ **不评估就上**→ 可能反而变差

---

## 实战：给 my_com_rag 加 Multi-Query

### Step 1：识别受益场景
看 golden set：
- 哪些 query 是"多表达"的（用户口语化 vs 文档正式）
- 哪些是"复杂多跳"

### Step 2：加 Multi-Query（仅对受益场景）
```python
def smart_retrieve(query):
    if is_simple_query(query):
        return simple_retrieve(query)
    else:
        return multi_query_retrieve(query)
```

**自适应** > 一刀切。

### Step 3：评估
对比基线，看 Multi-Query 是否真的提升。

### Step 4：加缓存
同样 query 不要重复跑 LLM 改写。

---

## 自测题

1. Multi-Query 的核心思路是什么？
2. n_queries 设 1 vs 设 10，区别？
3. 为什么 Multi-Query 后必须 Rerank？
4. HyDE + Multi-Query 同时用，主要缺点？
5. 什么样的 query 不适合 Multi-Query？

---

## retrieval/ + 03_rag_deep 完整小结

你已经看完所有 retrieval 高级技巧：

| 技术 | 解决什么问题 | ROI |
|------|-------------|-----|
| [hybrid_search.md](./hybrid_search.md) | 单一检索的盲区 | ⭐⭐⭐⭐ |
| [reranking.md](./reranking.md) | 召回精度不足 | ⭐⭐⭐⭐⭐（最高） |
| [hyde.md](./hyde.md) | 短 query 召回差 | ⭐⭐⭐ |
| [multi_query.md](./multi_query.md) | 表达多样性 | ⭐⭐⭐ |

**推荐落地顺序**：
1. **先加 Reranking**（必做，性价比最高）
2. **再加 Hybrid Search**（必做）
3. 视场景加 **Multi-Query / HyDE**（短 query 多时）

---

## 03_rag_deep 完整闭环

```
[分块] (chunking/)    ─ 决定 chunk 形态
    ↓
[Embedding]           ─ chunk / query → 向量
    ↓
[向量库]              ─ 存、查
    ↓
[检索] (retrieval/)   ─ Hybrid + Rerank + Multi-Query + HyDE
    ↓
[失败诊断]            ─ 出错时定位根因
    ↓
[评估闭环] (../05_evaluation/)  ─ 量化每一步的提升
```

**你现在具备了 FDE 视角下完整的 RAG 能力栈**。

> 回到 [Month 1 任务](../00_foundation/learning_path.md)，
> 把这套**实际用在 my_com_rag 上**，做出第一份 RAGAS 对比报告。
