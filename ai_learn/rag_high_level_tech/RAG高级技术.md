# RAG技术树 - 完整知识体系

## 技术树概览

本文档整理了RAG（检索增强生成）技术的完整知识体系，从基础到高级，帮助开发者系统学习和掌握RAG技术。

---

## 第一层：基础技术

### 1.1 文本处理

#### 文档加载
- **PDF解析**: PyPDF2, pdfplumber, Unstructured
- **Word文档**: python-docx
- **网页爬取**: BeautifulSoup, Scrapy
- **API数据**: requests, aiohttp
- **多格式支持**: LangChain loaders, LlamaIndex readers

#### 文本切分
- **固定大小切分**: 按字符数或token数切分
- **递归切分**: RecursiveCharacterTextSplitter
- **语义切分**: SemanticChunker
- **文档结构切分**: 按章节、段落切分

#### 元数据提取
- 文档标题、作者、创建时间
- 章节、页码、路径
- 自定义元数据标签

### 1.2 Embedding技术

#### 向量化模型
- **通用模型**
  - OpenAI: text-embedding-ada-002, text-embedding-3-small/large
  - Sentence-BERT: all-MiniLM-L6-v2, all-mpnet-base-v2
- **中文优化模型**
  - BGE: bge-base-zh, bge-large-zh
  - m3e-base, m3e-large
- **多语言模型**
  - multilingual-e5
  - BGE-multilingual-gemma2

#### 相似度计算
- **余弦相似度** (Cosine Similarity)
- **点积** (Dot Product)
- **欧氏距离** (Euclidean Distance)

### 1.3 向量数据库

#### 开源方案
- **Chroma**: 轻量级，本地开发
- **FAISS**: Facebook高效检索库
- **Milvus**: 分布式，高性能
- **Qdrant**: 过滤功能强大
- **Weaviate**: GraphQL接口，模块化

#### 托管服务
- **Pinecone**: 全托管，易扩展
- **Zilliz Cloud**: Milvus云服务
- **MongoDB Atlas Vector Search**

---

## 第二层：核心RAG

### 2.1 基础RAG流程

#### 索引构建
```
文档 → 加载 → 切分 → Embedding → 存储
```

#### 检索流程
```
查询 → Embedding → 向量搜索 → Top-K文档
```

#### 生成阶段
```
查询 + 检索文档 → 提示词模板 → LLM → 答案
```

### 2.2 提示词工程

#### Context Stuffing
- 将检索上下文填入提示词
- 控制上下文窗口大小
- 优化上下文排序

#### Prompt模板
```python
template = """
基于以下上下文回答问题：

{context}

问题：{question}

答案：
"""
```

### 2.3 评估指标

#### 检索质量
- **召回率** (Recall): 相关文档被检索到的比例
- **精确率** (Precision): 检索结果中相关文档的比例
- **MRR** (Mean Reciprocal Rank): 第一个相关结果的平均排名
- **NDCG** (Normalized DCG): 考虑排序位置的评分

#### 生成质量
- **忠实度** (Faithfulness): 答案是否基于上下文
- **相关性** (Relevance): 答案是否回答了问题
- **准确性** (Accuracy): 事实正确性

---

## 第三层：检索增强

### 3.1 混合检索 (Hybrid Search)

#### 原理
结合关键词检索和语义检索，兼顾精确匹配和语义理解

#### 实现方式
```python
# BM25 + 语义检索
bm25_results = bm25_search(query, top_k=50)
vector_results = vector_search(query_embedding, top_k=50)

# RRF融合
final_results = reciprocal_rank_fusion(bm25_results, vector_results)
```

#### 优势
- 关键词精确匹配
- 语义相似度匹配
- 互补优势，提高召回率

### 3.2 重排序 (Reranking)

#### 两阶段检索
```
阶段1 - 粗排: 检索 top 100-1000 文档
阶段2 - 精排: Rerank模型重新排序，取 top 5-10
```

#### Rerank模型
- **Cohere Rerank API**
- **BGE-Reranker**: bge-reranker-base, bge-reranker-large
- **Cross-Encoder**: ms-marco-* 系列
- **ColBERT**: late interaction模型

#### 实现示例
```python
# 检索阶段
retrieved_docs = vector_store.search(query, top_k=100)

# 重排序
reranked_docs = reranker_model.rank(query, retrieved_docs, top_k=10)
```

### 3.3 查询优化

#### 查询改写 (Query Rewriting)
- 同义词替换
- 错别字纠正
- 意图澄清
- LLM辅助改写

#### 查询扩展 (Query Expansion)
```python
# 多路查询
original_query = "什么是RAG"
expanded_queries = [
    "RAG检索增强生成原理",
    "RAG技术架构",
    "如何实现RAG系统"
]

# 合并检索结果
all_results = []
for q in expanded_queries:
    all_results.extend(search(q))
```

#### HyDE (Hypothetical Document Embeddings)
```
用户查询 → LLM生成假设答案 → 假设答案Embedding → 检索
```

**原理**: 用假设答案的语义表示去检索，而非原始查询

### 3.4 高级切分策略

#### 语义切分
- 根据语义边界切分
- 保持段落完整性
- 使用句子Embedding计算相似度

#### 递归切分
- 多级粒度切分
- 父子文档关系
- 检索时返回不同粒度

#### 固定大小 + 重叠
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,  # 重叠保持上下文连贯
    length_function=len
)
```

---

## 第四层：高级架构

### 4.1 多模态RAG

#### 图文检索
- **CLIP**: 图像-文本联合Embedding
- **多模态向量数据库**: 支持图像和文本联合检索
- 应用：产品搜索、图文问答

#### 视频RAG
- 视频帧提取 → 图像Embedding
- 字幕提取 → 文本Embedding
- 时间戳检索

#### 表格RAG
- 表格结构化解析
- 行/列级别Embedding
- Unstructured, LlamaParse工具

### 4.2 Agentic RAG (代理式RAG)

#### 核心思想
RAG系统具备自主决策能力，动态调整检索策略

#### 能力特征
```python
# 自我反思
if not satisfied(answer):
    query = refine_query(original_query, answer)
    retrieve_again()

# 动态检索次数
while need_more_info():
    retrieve()
    check_completeness()

# 路由决策
if query_type == "factual":
    use_vector_search()
elif query_type == "analytical":
    use_graph_search()
```

#### 框架实现
- **LangChain Agent**: ReAct, Plan-and-Execute
- **AutoGPT**: 自主任务规划
- **CrewAI**: 多代理协作

### 4.3 GraphRAG

#### 知识图谱增强
```
实体抽取 → 关系抽取 → 图谱构建 → 图检索
```

#### 技术栈
- **图数据库**: Neo4j, NebulaGraph
- **图算法**: PageRank, Community Detection
- **图Embedding**: Node2Vec, GraphSAGE

#### 优势
- 结构化知识表示
- 多跳推理能力
- 实体关系显式建模

#### 应用场景
- 复杂推理问答
- 因果关系分析
- 实体关系查询

### 4.4 自适应RAG

#### 路由机制
```python
# 查询分类
query_type = classify_query(query)

if query_type == "simple":
    direct_answer()
elif query_type == "requires_search":
    rag_pipeline()
elif query_type == "complex":
    multi_step_reasoning()
```

#### 动态Top-K
```python
# 根据查询复杂度调整检索数量
complexity = estimate_complexity(query)
top_k = complexity * 5
results = search(query, top_k=top_k)
```

### 4.5 组合检索策略

#### 父子文档 (Parent-Child Documents)
```
小文档（子）: 用于精准检索
大文档（父）: 提供完整上下文

流程: 检索子文档 → 返回父文档
```

#### 摘要索引
```
原始文档 + 摘要
检索摘要 → 返回原始文档
适合: 长文档检索
```

#### 多索引策略
- 摘要索引 + 详细索引
- 不同Embedding模型索引
- 不同切分策略索引

### 4.6 RAFT (Retrieval-Augmented Fine Tuning)

#### 核心思想
RAFT通过**微调LLM**本身，让模型学会在检索到的文档中区分相关和干扰信息，从而提升RAG系统的准确性和鲁棒性。

#### 三种训练场景对比

**场景1: CoT (Chain-of-Thought)**
```python
# 仅问题，无检索文档
输入: "什么是RAG？"
输出: 模型纯靠内部知识回答（可能产生幻觉）
```

**场景2: 传统RAG（未微调）**
```python
# 问题 + 检索文档（含干扰项）
输入:
  文档D1-D10 (其中D1-D5相关，D6-D10是干扰文档)
  问题: "什么是RAG？"
输出: 模型可能无法正确区分相关/无关文档
```

**场景3: RAFT（微调后）**
```python
# 问题 + 检索文档（含干扰项）
输入: 同上
输出:
  ✅ 学会从干扰文档中提取正确信息
  ✅ 学会引用正确文档
  ✅ 拒绝干扰信息
  ✅ 生成有引用的答案
```

#### RAFT训练流程

**1. 数据准备**
```python
# 为每个训练样本构建
training_sample = {
    "question": "什么是RAG？",
    "documents": [
        {"id": "D1", "content": "RAG是检索增强生成...", "relevant": True},
        {"id": "D2", "content": "RAG结合了检索和生成...", "relevant": True},
        {"id": "D3", "content": "Python是一种编程语言...", "relevant": False},
        {"id": "D4", "content": "深度学习是AI的分支...", "relevant": False},
    ],
    "answer": "根据文档[D1]和[D2]，RAG（检索增强生成）是一种...",
    "citations": ["D1", "D2"]
}
```

**2. 训练格式构建**
```python
def format_raft_training(sample):
    # 格式化文档
    docs_text = "\n\n".join([
        f"[{doc['id']}] {doc['content']}"
        for doc in sample['documents']
    ])

    # 构建训练输入
    training_input = f"""
你是一个有帮助的助手。请根据以下文档回答问题。
如果文档中没有答案，请说明。
请引用你使用的文档（使用文档ID，如[D1]、[D2]）。

文档：
{docs_text}

问题：{sample['question']}
"""

    # 训练目标输出
    training_output = sample['answer']

    return training_input, training_output
```

**3. 微调训练**
```python
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

# 准备数据集
train_dataset = [format_raft_training(s) for s in training_samples]

# 加载基础模型
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# 训练配置
training_args = TrainingArguments(
    output_dir="./raft-model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    warmup_steps=100,
)

# 微调
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
```

#### RAFT vs 传统RAG

| 方面 | 传统RAG | RAFT |
|------|---------|------|
| 模型 | 通用LLM（如GPT-4） | 专门微调的LLM |
| 文档区分 | ❌ 容易混淆相关/无关文档 | ✅ 学会精确区分 |
| 引用能力 | ⚠️ 需要复杂的prompt工程 | ✅ 训练时学会引用 |
| 抗干扰能力 | ❌ 弱，易被干扰信息误导 | ✅ 强，学会忽略干扰 |
| 数据需求 | 无需标注数据 | 需要QA对+文档标注 |
| 部署成本 | 低（API调用） | 中等（需要微调和部署） |
| 定制化 | 通用能力 | 可针对特定领域优化 |

#### 数据构建策略

**Oracle答案生成**
```python
# 使用强模型（如GPT-4）生成训练数据
def generate_raft_sample(question, documents, oracle_model="gpt-4"):
    # 1. 识别相关文档
    relevant_docs = [doc for doc in documents if is_relevant(doc, question)]

    # 2. 让Oracle模型生成答案
    prompt = f"""
基于以下相关文档回答问题，并引用文档：

{format_documents(relevant_docs)}

问题：{question}
"""
    oracle_answer = oracle_model.generate(prompt)

    # 3. 添加干扰文档
    distractor_docs = get_distractor_docs(documents, relevant_docs, k=5)
    all_docs = relevant_docs + distractor_docs

    return {
        "question": question,
        "documents": all_docs,
        "answer": oracle_answer,
        "citations": extract_citations(oracle_answer)
    }
```

**干扰文档选择**
```python
# 策略1: 随机选择
distractors = random.sample(all_docs, k=5)

# 策略2: 困难负样本（语义相似但无关）
distractors = get_semantically_similar_but_irrelevant(query, all_docs, k=5)

# 策略3: 同领域不同主题
distractors = get_same_domain_different_topic(query, all_docs, k=5)
```

#### 实现技巧

**提示词模板**
```python
RAFT_PROMPT_TEMPLATE = """
你是一个专业的问答助手。请仔细阅读提供的文档，并回答用户的问题。

要求：
1. 仅基于提供的文档回答问题
2. 如果文档中没有相关信息，明确说明"提供的文档中没有包含该问题的答案"
3. 回答时引用使用的文档，格式为[文档ID]
4. 忽略与问题无关的文档

文档：
{documents}

问题：{question}

答案：
"""
```

**训练技巧**
```python
# 1. 使用LoRA/QLoRA减少显存需求
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, lora_config)

# 2. 课程学习（从简单到困难）
# Epoch 1: 干扰文档少
# Epoch 2-3: 增加干扰文档数量

# 3. 数据增强
# - 同一问题，不同干扰文档组合
# - 改写问题，保持文档不变
```

#### 评估指标

**检索质量**
```python
# 文档引用准确率
def citation_accuracy(predicted, gold_standard):
    """预测的引用是否与标准答案一致"""
    return len(set(predicted) & set(gold_standard)) / len(gold_standard)

# 干扰文档排除率
def distractor_rejection_rate(predictions, distractor_ids):
    """是否成功忽略干扰文档"""
    cited = set(extract_citations(predictions))
    distractors = set(distractor_ids)
    return 1 - len(cited & distractors) / len(distractors)
```

**生成质量**
```python
# 忠实度（是否基于检索文档）
def faithfulness(answer, retrieved_docs):
    """答案是否仅基于检索文档"""
    pass

# 答案准确性
def accuracy(answer, gold_answer):
    """答案事实准确性"""
    pass
```

#### 适用场景

✅ **适合使用RAFT的场景**
- 需要高准确性的专业领域（法律、医疗、金融）
- 干扰信息较多的文档集合
- 需要精确引用的场景（学术论文、法规问答）
- 有充足标注数据的情况
- 需要部署本地模型（隐私/成本考虑）

❌ **不适合使用RAFT的场景**
- 快速原型验证
- 缺乏标注数据
- 通用知识问答（使用通用RAG即可）
- 预算有限（微调成本高）
- 频繁更新的文档（需要重新微调）

#### 相关资源

**论文**
- "RAFT: Adapting Language Models to Retrieve What's Needed" (2024)
- 作者：Akari Asai, Zejiang Shen, et al. (UC Berkeley, CMU)

**开源实现**
- 官方代码仓库：[GitHub -RAFT]
- HuggingFace RAFT模型示例

**工具**
- transformers: 微调框架
- PEFT (LoRA/QLoRA): 参数高效微调
- DeepSpeed: 大规模训练加速

---

## 第五层：生产优化

### 5.1 性能优化

#### 缓存策略
```python
# 查询缓存
@lru_cache(maxsize=1000)
def search(query):
    return vector_search(query)

# 文档缓存
# 检索结果缓存，避免重复计算
# Embedding缓存
```

#### 批处理
```python
# 批量Embedding
embeddings = embedding_model.embed_batch(texts, batch_size=32)

# 批量检索
queries = ["query1", "query2", ...]
results = vector_search.batch_search(queries)
```

#### 异步检索
```python
import asyncio

async def parallel_retrieval(query):
    results = await asyncio.gather(
        vector_search(query),
        bm25_search(query),
        graph_search(query)
    )
    return merge_results(results)
```

#### 索引优化
- **HNSW索引**: 高性能近似搜索
- **IVF索引**: 倒排文件索引
- **分区策略**: 按类别/时间分区

### 5.2 可观测性 (Observability)

#### 追踪 (Tracing)
- **LangSmith**: 全流程追踪
- **WandB**: 实验跟踪和可视化
- **Arize Phoenix**: LLM可观测性平台

#### 监控指标
```python
# 性能指标
- 检索延迟
- 端到端延迟
- QPS (每秒查询数)
- Token消耗

# 质量指标
- 检索准确率
- 答案相关性
- 用户满意度
```

#### 日志记录
- 查询日志
- 检索上下文
- 生成的答案
- 用户反馈

### 5.3 评估框架

#### Ragas
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

result = evaluate(
    dataset=test_dataset,
    metrics=[faithfulness, answer_relevancy, context_recall]
)
```

#### TruLens (TruEra)
- LLM应用评估
- RAG特定指标
- 可视化仪表板

#### Giskard
- 自动化测试
- 弱点检测
- 回归测试

#### DeepEval
- 单元测试框架
- CI/CD集成
- 自定义指标

### 5.4 安全性

#### 权限控制 (Access Control)
```python
# 文档级权限
def search_with_acl(query, user_id):
    results = vector_search(query)
    # 过滤用户无权访问的文档
    authorized = [doc for doc in results if doc.accessible_to(user_id)]
    return authorized
```

#### 数据脱敏
- 敏感信息检测
- PII（个人身份信息）过滤
- 动态脱敏

#### 提示词注入防护
- 输入验证
- 输出过滤
- 对抗性测试

### 5.5 成本优化

#### 模型选择策略
```python
# 分层模型使用
if task_complexity == "low":
    use_small_model()  # 更便宜、更快
elif task_complexity == "high":
    use_large_model()  # 更准确
```

#### Token优化
- 精简提示词
- 选择性上下文
- 答案压缩

#### 缓存策略
- Embedding缓存
- 检索结果缓存
- 常见问题缓存

---

## 常用工具栈总结

### 框架与库
- **LangChain**: 全功能RAG框架
- **LlamaIndex**: 数据框架，索引优化
- **Haystack**: Deepset开源框架
- **Vertex AI Search**: Google托管服务

### Embedding模型
```python
# OpenAI
openai.Embedding.create(input=text, model="text-embedding-3-small")

# 开源模型
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
embedding = model.encode(text)
```

### 向量数据库
```python
# Chroma
import chromadb
client = chromadb.Client()
collection = client.create_collection("my_collection")

# Pinecone
import pinecone
pinecone.init(api_key="...")
index = pinecone.Index("my-index")
```

### Rerank模型
```python
# Cohere Rerank
import cohere
co = cohere.Client(api_key)
results = co.rerank(query=query, documents=docs, top_n=10)

# BGE Reranker
from FlagEmbedding import FlagReranker
reranker = FlagReranker('BAAI/bge-reranker-large')
scores = reranker.compute_score([[query, doc] for doc in docs])
```

### 评估工具
- **Ragas**: 全面评估指标
- **TruLens**: 可观测性和评估
- **DeepEval**: 单元测试风格
- **Promptfoo**: 提示词测试

---

## 学习路径建议

### 初学者 (1-2个月)
1. ✅ 掌握第一层：基础技术
2. ✅ 实现基础RAG流程
3. ✅ 理解核心概念
4. ✅ 完成简单项目

### 进阶 (2-3个月)
1. ✅ 第三层：检索增强技术
2. ✅ 混合检索、重排序
3. ✅ 查询优化方法
4. ✅ 评估和调优

### 高级 (3-6个月)
1. ✅ 第四层：高级架构
2. ✅ Agentic RAG、GraphRAG
3. ✅ 多模态RAG
4. ✅ 生产环境部署

### 专家 (持续)
1. ✅ 第五层：生产优化
2. ✅ 大规模部署
3. ✅ 性能调优
4. ✅ 最新研究跟踪

---

## 参考资源

### 论文
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (2020)
- "Improving the Retrieval Stage" (2023)
- "HyDE: Precise Zero-Shot Dense Retrieval" (2022)
- "GraphRAG: From Local to Global" (2024)

### 开源项目
- LangChain: https://github.com/langchain-ai/langchain
- LlamaIndex: https://github.com/run-llama/llama_index
- BGE Embeddings: https://github.com/FlagOpen/FlagEmbedding
- Ragas: https://github.com/explodinggradients/ragas

### 学习资源
- LangChain文档: https://python.langchain.com/
- LlamaIndex文档: https://docs.llamaindex.ai/
- Pinecone学习中心: https://www.pinecone.io/learn/

---

**文档版本**: v1.0
**更新日期**: 2026-01-27
**作者**: Claude Code Assistant
