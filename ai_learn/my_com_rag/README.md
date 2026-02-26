# 企业知识库系统

基于 **RAG (检索增强生成)** 技术的智能企业知识库系统，由智谱AI驱动。

## 功能特性

- 📚 **文档管理**: 支持上传和管理多种格式的文档 (PDF, DOCX, TXT, MD, XLSX)
- 🤖 **智能问答**: 基于知识库内容的AI问答系统
- 🔍 **语义搜索**: 基于向量相似度的智能搜索
- 🎯 **精准回答**: 使用RAG技术确保答案基于真实文档内容
- 🌐 **Web界面**: 简洁易用的Web操作界面

## 技术栈

- **后端**: Python + FastAPI + LangChain
- **AI模型**: 智谱AI (GLM-4 + Embedding-2)
- **向量数据库**: ChromaDB / FAISS
- **前端**: HTML + JavaScript (原生)

## 快速开始

### 1. 环境准备

确保你已经安装了 Python 3.8+

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# macOS/Linux:
source venv/bin/activate
# Windows:
# venv\Scripts\activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

复制 `.env.example` 文件为 `.env` 并配置你的智谱AI API密钥:

```bash
cp .env.example .env
```

编辑 `.env` 文件,填入你的API密钥:

```env
ZHIPUAI_API_KEY=your_actual_api_key_here
```

### 4. 启动服务

```bash
# 方式1: 使用启动脚本
python start.py

# 方式2: 直接运行
python -m uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. 访问界面

打开浏览器访问: `http://localhost:8000` 或直接打开 `frontend/index.html`

## 项目结构

```
my_com_rag/
├── backend/                # 后端代码
│   ├── api/               # API接口
│   │   ├── main.py        # FastAPI主应用
│   │   └── models.py      # 数据模型
│   ├── core/              # 核心模块
│   │   ├── config.py      # 配置管理
│   │   ├── document_processor.py  # 文档处理
│   │   ├── embeddings.py  # 嵌入模型
│   │   ├── vector_store.py        # 向量数据库
│   │   └── rag_chain.py   # RAG链
│   └── docs/              # 文档
├── frontend/              # 前端代码
│   └── index.html         # Web界面
├── data/                  # 数据目录
│   ├── documents/         # 文档存储
│   ├── uploads/           # 上传文件
│   └── vector_db/         # 向量数据库
├── tests/                 # 测试
├── logs/                  # 日志
├── requirements.txt       # 依赖列表
├── .env.example          # 环境变量示例
└── README.md             # 项目说明
```

## API 接口

### 提问接口
```bash
POST /api/ask
Content-Type: application/json

{
  "question": "什么是人工智能?",
  "use_rag": true
}
```

### 搜索接口
```bash
POST /api/search
Content-Type: application/json

{
  "query": "机器学习",
  "k": 4
}
```

### 上传文档
```bash
POST /api/upload
Content-Type: multipart/form-data

file: <document file>
```

### 批量加载目录
```bash
POST /api/load-directory?directory=/path/to/docs
```

### 获取知识库信息
```bash
GET /api/info
```

### 清空知识库
```bash
DELETE /api/clear
```

## 使用说明

1. **上传文档**
   - 在Web界面点击上传区域
   - 选择要上传的文档 (支持 PDF, DOCX, TXT, MD, XLSX)
   - 点击"上传文档"按钮

2. **提问**
   - 在输入框中输入问题
   - 点击"提问"按钮获取AI生成的答案
   - 或点击"仅搜索"查看相关文档片段

3. **查看来源**
   - 每个答案都会显示相关文档来源
   - 可以追溯答案的原始出处

## 配置说明

在 `.env` 文件中可以配置:

```env
# 智谱AI API
ZHIPUAI_API_KEY=your_key

# 向量数据库类型 (chromadb 或 faiss)
VECTOR_DB_TYPE=chromadb

# 文档分块大小
CHUNK_SIZE=500
CHUNK_OVERLAP=50

# AI温度参数
TEMPERATURE=0.7

# 服务端口
PORT=8000
```

## 开发计划

- [ ] 支持更多文档格式 (PPT, 图片OCR)
- [ ] 添加用户认证系统
- [ ] 支持多轮对话
- [ ] 知识图谱可视化
- [ ] 多语言支持
- [ ] 文档版本管理
- [ ] 权限控制

## 常见问题

### Q: 如何获取智谱AI的API密钥?
A: 访问 [智谱AI开放平台](https://open.bigmodel.cn/) 注册并获取API密钥

### Q: 支持哪些文档格式?
A: 目前支持 PDF, DOCX, TXT, Markdown, XLSX

### Q: 向量数据库占用多少空间?
A: 取决于文档数量,一般每1000个文档块约100-500MB

### Q: 如何清空知识库?
A: 在API中调用 `DELETE /api/clear` 或删除 `data/vector_db` 目录

## 许可证

MIT License

## 联系方式

如有问题或建议,欢迎提Issue!
