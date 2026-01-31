# RAG 高级技术学习项目

RAG (Retrieval-Augmented Generation) 高级技术学习资料和代码实现。

## 项目结构

```
rag_high_level_tech/
├── raft/                            # RAFT 技术相关
│   ├── raft_simple_demo.py               # RAFT 简化实现
│   └── raft_training_data.json           # 训练数据示例
│
├── rag/                             # RAG 实战项目
│   ├── deepseek_faiss_rag.py             # DeepSeek API + Faiss 实现
│   ├── ollama_faiss_rag.py               # Ollama + Faiss 实现（需要 embedding 模型）
│   ├── ollama_simple_rag.py              # Ollama + TF-IDF 实现（无需 embedding 模型）⭐
│   ├── knowledge_base/                   # 知识库文档
│   ├── requirements.txt                  # 依赖列表
│   ├── .env.example                      # 配置示例
│   └── README.md                         # RAG 项目详细说明
│
└── RAG高级技术.md                   # 理论学习文档
```

## 快速开始

### 1. 学习 RAFT 技术

```bash
cd raft
python raft_simple_demo.py
```

### 2. 运行 RAG 实战项目（三个版本）

#### 版本 A：使用 DeepSeek API（需要 API Key）

```bash
cd rag
pip install -r requirements.txt
# 配置 .env 文件，填入 DeepSeek API Key
python deepseek_faiss_rag.py
```

#### 版本 B：使用本地 Ollama + Faiss（需要 embedding 模型）⭐

```bash
cd rag
pip install -r requirements.txt
# 下载 embedding 模型
ollama pull nomic-embed-text
# 运行
python ollama_faiss_rag.py
```

#### 版本 C：使用本地 Ollama + TF-IDF（无需 embedding 模型）⭐⭐ 推荐

```bash
cd rag
pip install scikit-learn
# 确保启动 Ollama
ollama serve
# 运行
python ollama_simple_rag.py
```

## 学习内容

### 1. RAFT (Retrieval-Augmented Fine Tuning)
- 核心理念：通过微调让模型学会识别相关信息
- 完整示例代码：`raft/raft_simple_demo.py`
- 包含训练、推理、评估的完整流程
- **适用场景**：检索质量差、需要精确引用

### 2. Native RAG 实战项目（三种实现）

#### 版本 A：DeepSeek API + Faiss
- 使用 DeepSeek API 进行 Embedding 和聊天
- 需要互联网连接和 API Key
- 检索质量最佳
- 代码：`rag/deepseek_faiss_rag.py`

#### 版本 B：Ollama + Faiss（向量检索）
- 使用本地 Ollama 模型
- 需要 embedding 模型（nomic-embed-text，274MB）
- 完全本地运行，无需联网
- 检索质量好
- 代码：`rag/ollama_faiss_rag.py`

#### 版本 C：Ollama + TF-IDF（关键词检索）⭐ 推荐学习
- 使用本地 Ollama 模型
- **不需要 embedding 模型**
- 使用 TF-IDF 进行文本匹配
- 完全本地运行，立即可用
- 适合学习和快速验证
- 代码：`rag/ollama_simple_rag.py`

### 3. RAG 版本对比

| 特性 | DeepSeek API | Ollama + Faiss | Ollama + TF-IDF |
|------|-------------|----------------|----------------|
| **需要联网** | ✅ 是 | ❌ 否 | ❌ 否 |
| **需要 API Key** | ✅ 是 | ❌ 否 | ❌ 否 |
| **需要 Embedding 模型** | API 提供 | ✅ 需要（274MB） | ❌ 不需要 |
| **检索质量** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **适合场景** | 生产环境 | 本地高质量 | 学习、快速验证 |

## 技术栈

- **RAFT**: 训练数据生成、微调、评估
- **Native RAG**:
  - DeepSeek API / Ollama
  - Faiss 向量检索 / TF-IDF 检索
  - 交互式问答系统
- **文档处理**: TXT, PDF, DOCX
- **向量数据库**: Faiss
- **文本检索**: TF-IDF (scikit-learn)
- **本地模型**: Ollama + deepseek-r1:1.5b

## 学习路径

### 初学者路径
1. 📖 阅读 `RAG高级技术.md` 了解理论
2. 🚀 运行 `rag/ollama_simple_rag.py` 快速体验 Native RAG（推荐，无需额外模型）
3. 📝 运行 `raft/raft_simple_demo.py` 理解 RAFT 概念

### 进阶路径
1. 📦 下载 embedding 模型：`ollama pull nomic-embed-text`
2. 🔥 运行 `rag/ollama_faiss_rag.py` 体验向量检索版本
3. 🎯 对比三种 RAG 实现的效果差异
4. 💡 根据 RAFT 理念优化你的 RAG 系统

### 生产环境路径
1. 🔑 获取 DeepSeek API Key
2. 🏗️ 部署 `rag/deepseek_faiss_rag.py` 到生产环境
3. 📊 根据实际需求选择：Native RAG 或 RAFT

## 常见问题

### Q: 三个 RAG 版本应该选哪个？
**A:**
- **学习/快速验证**: `ollama_simple_rag.py`（TF-IDF，无需额外模型）
- **本地高质量**: `ollama_faiss_rag.py`（需要下载 embedding 模型）
- **生产环境**: `deepseek_faiss_rag.py`（DeepSeek API，效果最好）

### Q: Native RAG 和 RAFT 有什么区别？
**A:**
- **Native RAG**: 直接检索 → 生成，无需训练，适合快速搭建
- **RAFT**: 通过微调让模型学会识别和忽略干扰文档，需要训练数据，效果更好但成本高

### Q: TF-IDF 和向量检索哪个好？
**A:**
- **TF-IDF**: 基于关键词匹配，简单快速，适合小规模数据
- **向量检索**: 基于语义理解，能理解同义词，效果更好，适合生产环境

### Q: Ollama 和 DeepSeek API 怎么选？
**A:**
- **Ollama**: 完全本地，免费，需要硬件资源，适合离线使用
- **DeepSeek API**: 云端服务，需要付费，效果稳定，适合生产环境

### Q: conda 环境需要特别配置吗？
**A:**
- 创建专用环境：`conda create -n ai_learn python=3.13`
- 安装依赖：`pip install -r rag/requirements.txt`
- Ollama 不依赖 conda，可以独立使用

## 作者

Claude Code Assistant
日期: 2026-01-27
