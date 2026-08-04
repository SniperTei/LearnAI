# 文档感知分块（Document-aware Chunking）

> 你的知识库不是"一堆文字"，是结构化文档。
> 忽略结构 = 浪费信息。

---

## 什么是文档感知分块

按文档的**自然结构**切分：
- Markdown：按 `#` / `##` / `###` 标题
- HTML：按 `<section>` / `<div>` / `<h2>`
- PDF：按章节、页码
- 代码：按函数 / 类
- 法律：按条款 / 章节

每个 chunk 附带**结构元数据**：
```json
{
  "content": "年假政策：入职享 15 天...",
  "metadata": {
    "doc_title": "员工手册",
    "section_h1": "第三章 福利",
    "section_h2": "3.2 年假",
    "page": 12,
    "source": "hr_handbook_v3.docx"
  }
}
```

---

## 为什么有用

### 用途 1：精准过滤
```sql
-- pgvector 检索时按元数据过滤
WHERE section_h1 = '福利' AND doc_title = '员工手册'
```

### 用途 2：更好的引用
答案可以说："根据《员工手册》第 3.2 节..."
而不是模糊的"根据上下文"。

### 用途 3：避免跨章节切断
传统 chunk 可能把"3.1 病假"和"3.2 年假"切到一块，
检索时混在一起。文档感知分块保证边界清晰。

### 用途 4：父子检索
小 chunk（"3.2 年假"的某段）召回后，
返回它的父节点（整个"3.2 年假"章节）给 LLM。

---

## Markdown 分块（最常见）

```python
import re
from dataclasses import dataclass

@dataclass
class MarkdownChunk:
    content: str
    h1: str = ""
    h2: str = ""
    h3: str = ""

def chunk_markdown(text: str, max_size: int = 500) -> list[MarkdownChunk]:
    """按 Markdown 标题切分，保留层级路径。"""
    lines = text.split("\n")
    chunks = []
    current_h1 = current_h2 = current_h3 = ""
    current_content = []

    def flush():
        if current_content:
            content = "\n".join(current_content).strip()
            if content:
                # 如果内容超长，二次切
                if len(content) > max_size:
                    sub_chunks = recursive_chunk(content, max_size, 50)
                    for sub in sub_chunks:
                        chunks.append(MarkdownChunk(
                            content=sub,
                            h1=current_h1, h2=current_h2, h3=current_h3
                        ))
                else:
                    chunks.append(MarkdownChunk(
                        content=content,
                        h1=current_h1, h2=current_h2, h3=current_h3
                    ))

    for line in lines:
        # 检测标题
        m1 = re.match(r'^#\s+(.+)', line)
        m2 = re.match(r'^##\s+(.+)', line)
        m3 = re.match(r'^###\s+(.+)', line)

        if m1:
            flush()
            current_content = []
            current_h1 = m1.group(1).strip()
            current_h2 = current_h3 = ""
        elif m2:
            flush()
            current_content = []
            current_h2 = m2.group(1).strip()
            current_h3 = ""
        elif m3:
            flush()
            current_content = []
            current_h3 = m3.group(1).strip()
        else:
            current_content.append(line)

    flush()
    return chunks
```

### LangChain 等价
```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
)
chunks = splitter.split_text(md_text)
# 每个 chunk 是 Document 对象，metadata 含 h1/h2/h3
```

---

## HTML 分块

```python
from bs4 import BeautifulSoup

def chunk_html(html: str, max_size: int = 500) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    chunks = []

    # 找所有 section / article / 主要 div
    for section in soup.find_all(["section", "article", "div"]):
        # 跳过空 section
        text = section.get_text(strip=True)
        if not text or len(text) < 50:
            continue

        # 提取该 section 的标题（最近的 h1-h3）
        heading = section.find(["h1", "h2", "h3"])
        heading_text = heading.get_text(strip=True) if heading else ""

        # 切
        if len(text) > max_size:
            sub_texts = recursive_chunk(text, max_size, 50)
        else:
            sub_texts = [text]

        for sub in sub_texts:
            chunks.append({
                "content": sub,
                "heading": heading_text,
                "tag": section.name,
            })

    return chunks
```

---

## PDF 分块（最棘手）

PDF 没有"结构"——只有定位的文字。
策略：

### 策略 1：用 PDF 解析库
```python
import pdfplumber

with pdfplumber.open("doc.pdf") as pdf:
    for page_num, page in enumerate(pdf.pages, 1):
        text = page.extract_text()
        # 进一步分块
```

### 策略 2：按字号判断标题
```python
# 字号大的是标题，小的是正文
for char in page.chars:
    if char["size"] > 14:
        # 标题
        ...
```

### 策略 3：用专门的工具
- **unstructured.io**：自动识别文档结构
- **LlamaParse**：付费但效果好
- **Marker**：开源 Markdown 转换
- **Docling**（IBM 开源）：质量不错

**FDE 推荐**：起步用 `pdfplumber`，效果不够升级到 `unstructured` 或 `LlamaParse`。

---

## 表格 / 图片的处理

### 表格
**不要**把表格按普通文本切——会丢失行列关系。

```python
# 用 markdown 表达
"产品 | 价格 | 库存\n--- | --- | ---\nA | 100 | 50\nB | 200 | 30"
```

或存为结构化数据 + 文本描述：
```
"表格：产品价格表。包含 3 列：产品名、价格、库存。共 2 行..."
```

### 图片
- 用多模态 embedding（如 CLIP）
- 或用 LLM 生成图片描述，再 embed 描述

---

## 元数据设计（关键）

每个 chunk 都应该带元数据。**好的元数据设计**让 RAG 性能大幅提升。

### 必备字段
```json
{
  "content": "...",
  "metadata": {
    "doc_id": "hr_handbook_v3",
    "doc_title": "员工手册",
    "version": "3.2",
    "updated_at": "2024-08-15",
    "section_path": "第三章福利/3.2年假",
    "page": 12,
    "chunk_index": 45,
    "language": "zh",
    "source": "hr_department"
  }
}
```

### 业务字段（按需）
- `tenant_id`：多租户隔离
- `department`：按部门过滤
- `confidence`：信息可信度
- `expiry_date`：信息有效期

---

## 元数据过滤的实际威力

```sql
-- 场景 1：多租户
WHERE tenant_id = $1

-- 场景 2：只查最新版
WHERE doc_id IN (
    SELECT doc_id FROM documents
    WHERE version = (SELECT MAX(version) FROM documents WHERE title = title)
)

-- 场景 3：只查 HR 文档
WHERE department = 'hr'

-- 场景 4：时效性
WHERE updated_at > '2024-01-01'
```

**没有元数据**，这些过滤都要在 Python 里做，又慢又不准。

---

## 决策：要不要做文档感知

```
你的文档是结构化的吗（Markdown / HTML / 有清晰标题的 PDF）？
├── 是 → 必做文档感知分块
└── 否
    └── 你的文档有明显的自然段落吗？
        ├── 是 → 递归分块 + 加 metadata
        └── 否 → 用 LlamaParse / unstructured 先转结构化
```

---

## 反模式

❌ **丢掉元数据**（chunk 只存 text）→ 浪费信息
❌ **PDF 直接 extract_text 就切**→ 丢掉标题、表格结构
❌ **所有文档类型用同一套 splitter**→ 浪费
❌ **元数据字段不统一**（有的有 doc_id 有的没有）→ 过滤不准
❌ **不更新元数据**（文档改了，metadata 还指旧版）→ 召回错版本

---

## 实战：给 my_com_rag 加文档感知

### Step 1：盘点文档类型
你的知识库里：
- Markdown / 纯文本 / PDF / Word？
- 各占多少？

### Step 2：按类型选 splitter
- Markdown → MarkdownHeaderTextSplitter
- PDF → pdfplumber + 后处理 / LlamaParse
- Word → unstructured

### Step 3：统一元数据 schema
所有 chunk 共用一份 metadata 字段定义。

### Step 4：迁移 + 评估
- 用新 chunking 重建索引
- 跑 RAGAS 对比

### Step 5：加元数据过滤
让前端可以传过滤条件（如"只查 HR 文档"）。

**产出**：
- 一份"元数据 schema 文档"
- RAGAS 对比报告（应该有提升）

---

## 自测题

1. 一个 Markdown 文档，用文档感知分块相比递归，主要好处是什么？
2. PDF 文档为什么要避免直接用 extract_text？
3. 元数据应该至少包含哪些字段？
4. 多租户场景下，元数据怎么用？
5. 表格为什么要特殊处理？

---

## chunking/ 小结

你已经看完分块 3 篇：

| 文档类型 / 场景 | 推荐策略 |
|----------------|---------|
| 通用、PoC | [fixed_vs_recursive.md](./fixed_vs_recursive.md) - 递归 |
| 长叙事、主题多变 | [semantic_chunking.md](./semantic_chunking.md) |
| 结构化（MD/HTML/PDF） | [document_aware.md](./document_aware.md)（最常用） |

**默认建议**：递归分块 + 文档感知元数据 + 必要时加 Small-to-Big。

> 下一步：进入 [../retrieval/](../retrieval/)
