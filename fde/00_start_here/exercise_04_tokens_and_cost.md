# 练习 4：Token 与成本

> 目标：建立"token 直觉"和"成本直觉"。
> 这是 FDE 的核心技能——你必须知道一句话值多少钱。

---

## 学习目标

做完你应该能用人话讲清：

1. 什么是 token？它和字符/字节/词是什么关系？
2. 为什么中文和英文的 token 数差异巨大？
3. 一次 API 调用，input token 和 output token 哪个更贵？
4. 给定一段文本，你能**估算**它的 token 数（误差 ±20% 内）
5. 跑 1000 次平均请求，成本估算多少？

---

## 任务

### 任务 4.1：直觉建立

写 `token_compare.py`：
- 准备 5 段文本：
  1. 英文短句（"Hello world"）
  2. 英文长文（一段新闻）
  3. 中文短句（"你好世界"）
  4. 中文长文（一段新闻）
  5. 代码（一段 50 行 Python）

- 对每段：
  - 数字符数
  - 数词数（按空格 / 中文按字）
  - 用 tiktoken 算 token 数（GPT-4 的 tokenizer）
  - 如果用 DeepSeek：用官方 tokenizer（或近似用 tiktoken）

**观察**：
- 中文每字几个 token？
- 英文每词几个 token？
- 代码的 token 数 / 字符数 比例？

### 任务 4.2：自己测 tokenizer 边界

测一些"反直觉"的：
- "1+1=2" 几个 token？
- "iPhone15ProMax" 几个 token？（连写 vs 分开）
- "https://www.example.com/very/long/url/path" 几个？
- emoji 表情："😀😂🤔" 几个？
- 重复词："哈哈哈哈哈哈哈哈哈" 几个？

**意识到**：tokenizer 不是按字符、不是按词，是按"子词"（BPE 算法），需要实测。

### 任务 4.3：成本核算

写 `cost_calc.py`：

1. 选定一个模型（如 `deepseek-chat`）
2. 查它当前价格（DeepSeek 官网 → pricing）
3. 输入：
   - 一段 input 文本（如 RAG 召回了 2000 字的 context）
   - 期望 output 长度（如 500 字）
4. 输出：
   - input token 数
   - output token 数
   - 总成本（人民币）

**测多个场景**：
- 简单问答（input 50 字，output 100 字）
- RAG（input 5000 字，output 500 字）
- 长文档摘要（input 50000 字，output 1000 字）

### 任务 4.4：用量级估算

不看代码，估算：
- 100 万次 RAG 查询，假设每次 input 3000 tokens / output 300 tokens，DeepSeek 价格 ¥0.001/1k input、¥0.002/1k output（写本笔记时的近似价，以你查到的为准），总成本多少？
- 一个用户每天用 10 次，1 万用户一个月成本多少？

**FDE 必备直觉**：能在脑子里估算任何场景的大致成本。

### 任务 4.5（可选）：跑真实负载

如果你已经有 `my_com_rag` 在跑，看一周的实际 token 消耗和成本。

---

## 提示

### Token 的本质

LLM 不直接处理字符或词，它处理 **token**——一种"子词单元"。

训练前，模型先用 BPE（Byte Pair Encoding）等算法把语料切成 token：
```
"hello"     → ["hello"]              (1 token)
"happiness" → ["happiness"]           (1 token)
"unhappiness" → ["un", "happiness"]  (2 token)
"罕见字"     → 可能 2-3 个 token
```

### 为什么中英文差异大

英文 tokenizer 在英文语料上训练 → 英文常用词压缩好（每词 1 token）。
中文对它来说是"罕见序列" → 每个汉字可能 1-3 token。

**实测数据**（GPT-4 tokenizer）：
```
"Hello, how are you?"           → 6 tokens,  18 字符
"你好，你最近怎么样？"            → 11 tokens, 11 字符
```
英文每字符 ≈ 0.33 token，中文每字符 ≈ 1 token。

### 成本公式

```
单次调用成本 = (input_tokens × input_price + output_tokens × output_price) / 1000
```

注意单位通常是"每 1k token 多少美元 / 元"。

### 为什么 output 比 input 贵

输出 token 涉及**逐个生成**（每个 token 都要前向计算），输入 token 是一次性处理 + 缓存（KV cache）。
所以 output price 通常是 input 的 2-5 倍。

---

## 检查点

合上代码，回答：

1. "我有一个 5000 字的中文文档" → 大约多少 token？（你应该能秒答：5000-7500）
2. DeepSeek 一次 RAG（input 4000 + output 500）成本多少？说个数量级（应该 < ¥0.01）
3. 为什么输出 token 比输入 token 贵？
4. 为什么中文 token 比英文多？这对你的项目意味着什么？
5. 如果客户问"每月预算 ¥1000 够不够跑这个 AI 助手"，你怎么估算？

---

## 常见坑

### 坑 1：以为 1 字 = 1 token
中文最常见错误。1000 字中文 ≈ 1500 token，不是 1000。

### 坑 2：忽略 system prompt 的成本
system prompt 写 2000 字 → 每次请求都算这 2000 字的 input → 累积惊人。

### 坑 3：忽略历史 messages 的成本
多轮对话里，**所有历史消息每次都重新计费**（除非用了 prompt caching）。
聊 50 轮的总成本 ≈ O(N²)。

### 坑 4：用 GPT-3.5 价格估 GPT-4
价格差 10 倍以上。**永远查当前价格**——模型升级后价格会变。

### 坑 5：不监控真实成本
开发时觉得"便宜"，上线后百万次调用 → 账单爆炸。
生产必须接监控（见 `fde/06_observability/`）。

### 坑 6：以为 reranking / embedding 不花钱
embedding 按 token 计费；reranker 按对计费。
百万级数据 embedding 一次 → ¥几十到几百。

---

## 学完后

在 [my_notes.md](./my_notes.md) 加一段：
- 我之前对 token 的误解是 ___
- 我现在能秒估的场景：
  - 5000 字中文 ≈ ___ token
  - 1k 次 RAG 调用 ≈ ¥___
- 这让我意识到的成本风险：___

---

> 下一步：[exercise_05_streaming.md](./exercise_05_streaming.md)
