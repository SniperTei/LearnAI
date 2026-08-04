# 固定分块 vs 递归分块（实战对比）

> 理论 [../chunking_strategies.md](../chunking_strategies.md) 讲过，
> 这里给可直接跑的代码 + 实测对比。

---

## 最小实现：固定分块

```python
def fixed_size_chunk(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """按字符数固定切分，相邻 chunk 有重叠。"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap  # 下一个 chunk 在重叠处开始
    return chunks

# 用法
text = "公司年假政策......（长文本）"
chunks = fixed_size_chunk(text, chunk_size=500, overlap=50)
print(f"切出 {len(chunks)} 个 chunk")
print(f"第一个 chunk 长度: {len(chunks[0])}")
print(f"前两个 chunk 重叠: '{chunks[0][-50:]}' vs '{chunks[1][:50]}'")
```

### 优点
- 100% 可预测（chunk 数 = ceil(len / (size - overlap))）
- 极快
- 调参只有两个（size, overlap）

### 缺点
- 可能在句子中间切断
- 不感知文档结构

### 典型问题
```
text = "员工手册规定，年假 15 天。病假 10 天。"
fixed_chunk(text, size=10, overlap=0) = [
  "员工手册规定，年",
  "假 15 天。病假 ",     ← "年假"被切开
  "10 天。"
]
```

---

## 最小实现：递归分块

```python
SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ", ""]

def recursive_chunk(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separators: list[str] = None,
) -> list[str]:
    """按分隔符优先级递归切分。"""
    if separators is None:
        separators = SEPARATORS

    # 如果文本已经够小，直接返回
    if len(text) <= chunk_size:
        return [text] if text else []

    # 找一个能切的分隔符
    for sep in separators:
        if sep == "":
            continue  # 最后兜底
        if sep in text:
            parts = text.split(sep)
            # 尝试按这个分隔符组合成接近 chunk_size 的 chunk
            return _merge_parts(parts, sep, chunk_size, chunk_overlap, separators)

    # 所有分隔符都没匹配，按字符硬切
    return fixed_size_chunk(text, chunk_size, chunk_overlap)


def _merge_parts(parts, sep, chunk_size, chunk_overlap, separators):
    chunks = []
    current = ""
    for part in parts:
        candidate = (current + sep + part) if current else part

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            # candidate 太大
            if current:
                chunks.append(current)
            # 单个 part 自己就超长 → 用更细的分隔符递归
            if len(part) > chunk_size:
                sub = recursive_chunk(part, chunk_size, chunk_overlap, separators[1:])
                chunks.extend(sub)
                current = ""
            else:
                current = part
    if current:
        chunks.append(current)
    return chunks

# 用法
chunks = recursive_chunk(long_text, chunk_size=500, chunk_overlap=50)
```

### 优点
- 尽量保留语义边界（段落、句子）
- chunk 大小相对均匀
- 自适应文档结构

### 缺点
- 实现稍复杂
- chunk 大小不严格一致

### LangChain 等价
LangChain 的 `RecursiveCharacterTextSplitter` 就是这个逻辑，可以直接用：

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", " ", ""],
)
chunks = splitter.split_text(long_text)
```

---

## 实测对比脚本

```python
import pandas as pd
from typing import Callable

def evaluate_chunking(name: str, chunk_fn: Callable, text: str, golden_set):
    """对一种 chunking 策略跑评估。"""
    chunks = chunk_fn(text)

    # 1. chunk 统计
    chunk_lens = [len(c) for c in chunks]
    stats = {
        "name": name,
        "n_chunks": len(chunks),
        "len_mean": sum(chunk_lens) / len(chunk_lens),
        "len_min": min(chunk_lens),
        "len_max": max(chunk_lens),
        "len_std": (sum((l - sum(chunk_lens)/len(chunk_lens))**2 for l in chunk_lens) / len(chunk_lens)) ** 0.5,
    }

    # 2. 在 golden_set 上跑 RAG（伪代码）
    # rebuild_index(chunks)
    # ragas_score = ragas_eval(golden_set)

    return stats

# 对比
df = pd.DataFrame([
    evaluate_chunking("fixed-256", lambda t: fixed_size_chunk(t, 256, 30), text, golden),
    evaluate_chunking("fixed-512", lambda t: fixed_size_chunk(t, 512, 50), text, golden),
    evaluate_chunking("fixed-1024", lambda t: fixed_size_chunk(t, 1024, 100), text, golden),
    evaluate_chunking("recursive-256", lambda t: recursive_chunk(t, 256, 30), text, golden),
    evaluate_chunking("recursive-512", lambda t: recursive_chunk(t, 512, 50), text, golden),
    evaluate_chunking("recursive-1024", lambda t: recursive_chunk(t, 1024, 100), text, golden),
])
print(df)
```

### 典型结果（你的实际数会不同）

```
name             n_chunks  len_mean  len_min  len_max  len_std
fixed-256             47       256        7      256      0.5
fixed-512             24       510        5      512      1.2
fixed-1024            12      1020        4     1024      2.5
recursive-256         52       230       45      256     45.6  ← 大小不均
recursive-512         26       470       80      512     80.3
recursive-1024        13       950      150     1024    180.5
```

### 通常观察到的结论

- **固定**：标准差极小，但有边界切断
- **递归**：大小有方差，但语义更完整

**质量对比**（实际跑 RAGAS）：
- 短 factoid：两者差不多
- 长文档推理：recursive 通常赢 5–10%
- 法律 / 技术文档（段落敏感）：recursive 显著赢

---

## 进阶：可调参数

### 1. 自定义分隔符
```python
# Markdown
md_separators = ["\n## ", "\n### ", "\n\n", "\n", "。", " "]

# 法律文档
legal_separators = ["\n第.*条", "\n\n", "\n", "。"]

# 代码
code_separators = ["\nclass ", "\ndef ", "\n\n", "\n", ";"]
```

### 2. 长度函数（按 token 而非字符）
```python
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4")

def token_len(text: str) -> int:
    return len(enc.encode(text))

# 在 chunker 里把 len(text) 换成 token_len(text)
```

**强烈建议**：用 token 而非字符——因为 LLM 计费按 token。
中文字符数和 token 数差异大（1 汉字 ≈ 1–2 token）。

### 3. 长度下限
```python
# chunk 不能太短
if len(chunk) < chunk_size * 0.3:
    # 合并到上一个或下一个
```

避免出现 30 字的小碎片。

---

## 一个真实场景

**场景**：你 my_com_rag 里有 100 份员工手册（每份 5–20KB）。
现在的 chunk_size 是 500，没测过。

**建议流程**：

```python
# 1. 跑基线
build_and_eval("current", chunk_size=500, overlap=0, mode="fixed")

# 2. 扫参数
for size in [256, 512, 1024]:
    for overlap in [0, 50, 100]:
        for mode in ["fixed", "recursive"]:
            build_and_eval(f"{mode}-{size}-{overlap}", size, overlap, mode)

# 3. 看结果
# 假设发现 recursive-512-50 在 faithfulness 上最优
# → 切换
```

**预期产出**：一份 Excel/CSV，9-18 行结果，决策有据。

---

## 反模式

❌ **永远用默认 chunk_size=1000**（LangChain 老默认）→ 中文章节可能正好被切
❌ **不看 chunk 实际内容** → 不知道切得对不对
❌ **不评估就改参数** → 凭感觉调整
❌ **overlap 设太大** → 索引膨胀，成本翻倍
❌ **不分文档类型用同一套** → Markdown 和纯文本混切

---

## 自测题

1. 同样 chunk_size=512，为什么 recursive 的 chunk 长度有方差？
2. 你怎么知道当前 chunk_size 适合你？
3. overlap 设 chunk_size 的 50%，会有什么后果？
4. 中文章节"第三章 年假"，按字符 500 切，可能出什么问题？
5. 用字符长度 vs token 长度，区别大吗？

---

> 下一步：[semantic_chunking.md](./semantic_chunking.md)
