# RAG 高级技术学习项目

RAG (Retrieval-Augmented Generation) 高级技术学习资料和代码实现。

## 项目结构

```
rag_high_level_tech/
├── raft/                    # RAFT 技术相关
│   ├── raft_simple_demo.py       # RAFT 简化实现
│   └── raft_training_data.json   # 训练数据示例
│
├── rag/                     # RAG 实战项目
│   ├── deepseek_faiss_rag.py     # DeepSeek + Faiss 实现
│   ├── knowledge_base/           # 知识库文档
│   ├── requirements.txt          # 依赖列表
│   ├── .env.example              # 配置示例
│   └── README.md                 # RAG 项目说明
│
└── RAG高级技术.md           # 理论学习文档
```

## 快速开始

### 学习 RAFT 技术

```bash
cd raft
python raft_simple_demo.py
```

### 运行 RAG 实战项目

```bash
cd rag
pip install -r requirements.txt
# 配置 .env 文件
python deepseek_faiss_rag.py
```

## 学习内容

### 1. RAFT (Retrieval-Augmented Fine Tuning)
- 核心理念：通过微调让模型学会识别相关信息
- 完整示例代码：`raft/raft_simple_demo.py`
- 包含训练、推理、评估的完整流程

### 2. RAG 实战项目
- DeepSeek + Faiss 完整实现
- 支持多种文档格式
- 交互式问答系统
- 项目代码：`rag/deepseek_faiss_rag.py`

## 技术栈

- **RAFT**: 训练数据生成、微调、评估
- **RAG**: DeepSeek API、Faiss 向量检索
- **文档处理**: TXT, PDF, DOCX
- **向量数据库**: Faiss

## 学习路径

1. 📖 阅读 `RAG高级技术.md` 了解理论
2. 🚀 运行 `raft/raft_simple_demo.py` 理解 RAFT
3. 💻 实践 `rag/deepseek_faiss_rag.py` 搭建完整系统

## 作者

Claude Code Assistant
日期: 2026-01-27
