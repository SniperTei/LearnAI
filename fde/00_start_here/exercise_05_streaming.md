# 练习 5：流式输出（Streaming）

> 目标：理解为什么 ChatGPT 是"一个字一个字"显示的，以及怎么实现。
> 这是用户体验的关键。

---

## 学习目标

做完你应该能用人话讲清：

1. 流式输出（streaming）和非流式有什么本质区别？
2. 为什么流式**不便宜**也不更快（计算上），但**体验更好**？
3. 流式响应是什么格式？（SSE / chunked）
4. 怎么用 Python 接收流式响应？
5. 前端怎么消费流式数据（概念上）？

---

## 任务

### 任务 5.1：非流式 vs 流式，对比延迟

写 `compare.py`：
- 同一个 prompt（让它输出 500 字的故事）
- 用 `stream=False`（默认）跑一次，记录**首字节时间**和**总时间**
- 用 `stream=True` 跑一次，记录**首 token 时间**和**总时间**

**对比**：
- 总时间差不多（甚至流式略慢）
- 但**首字节时间**差异巨大（流式 1 秒 vs 非流式 8 秒）

**这就是流式的核心价值**：不是更快，是**让用户更早看到东西**。

### 任务 5.2：解析 SSE 流

OpenAI 兼容 API 的流式响应格式是 **SSE（Server-Sent Events）**：
```
data: {"choices":[{"delta":{"content":"H"}}]}
data: {"choices":[{"delta":{"content":"i"}}]}
data: {"choices":[{"delta":{"content":"!"}}]}
data: [DONE]
```

每个 `data:` 行是一个 JSON 片段，`delta.content` 是这次新增的 token。

写 `stream.py`：
- 调用 API，`stream=True`
- 用 `resp.iter_lines()` 或 `resp.iter_content()` 逐行读取
- 解析每行的 JSON，提取 `delta.content`
- 实时打印（不要等全部完成）

### 任务 5.3：在终端实现"打字机效果"

用 ANSI 控制符 + `flush=True`，让输出像 ChatGPT 那样逐字出现。

伪代码：
```python
for line in resp.iter_lines():
    if line starts with "data: ":
        parse JSON
        new_text = delta.content
        if new_text:
            print(new_text, end="", flush=True)   # 关键：end="" + flush=True
```

### 任务 5.4：处理边界情况

测试：
- 网络中断 → 流到一半挂了，你怎么处理？（部分结果是否保留？）
- `[DONE]` 信号 → 怎么判断流结束？
- 空行 / 心跳行 → 怎么过滤？
- 错误响应（key 错）→ 流模式下的错误响应长什么样？

### 任务 5.5（进阶）：把流式包装成 generator

写一个函数：
```python
def stream_chat(messages, **kwargs):
    """Yields 每个 token 字符串。"""
    # ... 调用 API
    for line in resp.iter_lines():
        # ... 解析
        yield delta_content
```

用法：
```python
for token in stream_chat(messages):
    print(token, end="", flush=True)
```

这种接口设计有什么好处？（提示：解耦、可组合）

---

## 提示

### 流式的本质

非流式：
```
Client → 请求
              Server 处理 N 秒（生成完整答案）
Client ← 完整响应
用户体验：等 8 秒 → 突然看到全部答案
```

流式：
```
Client → 请求
              Server 生成第 1 个 token (200ms)
Client ← 第 1 个 token
              Server 生成第 2 个 token (50ms)
Client ← 第 2 个 token
...
用户体验：200ms 看到第一个字，后续逐字出现
```

**关键认知**：
- 总生成时间几乎一样
- 用户的"感知等待"从 8s → 0.2s
- 用户**心理上**觉得快得多

### 为什么不便宜

模型还是要生成同样多的 token。
只是把"一次性返回"拆成"逐个返回"——**总计算量没变**。

### SSE 格式细节

OpenAI 兼容 API 用 Server-Sent Events：
- 每条消息以 `data: ` 开头
- 每条以 `\n\n` 分隔（不是 `\n`）
- 流结束时是 `data: [DONE]`
- 可能有心跳行（如空行或 `: keepalive`）

### Python 接收流

```python
resp = requests.post(url, json={..., "stream": True}, stream=True, headers=...)
for line in resp.iter_lines():
    if not line:
        continue
    line = line.decode("utf-8")
    if not line.startswith("data: "):
        continue
    data = line[6:]   # 去掉 "data: " 前缀
    if data == "[DONE]":
        break
    chunk = json.loads(data)
    delta = chunk["choices"][0]["delta"].get("content", "")
    if delta:
        print(delta, end="", flush=True)
```

**注意 `stream=True`** 在 `requests.post` 里——必须设，否则会一次性读完。

---

## 检查点

合上代码，回答：

1. 流式和非流式，**总时间**谁更短？（答案：几乎一样）
2. 为什么用户**觉得**流式更快？
3. SSE 的每行格式是什么？你怎么解析？
4. 如果用户中途关掉浏览器，模型还在生成，会发生什么？（思考题）
5. 流式模式下，错误（如 key 错）怎么呈现给客户端？

---

## 常见坑

### 坑 1：忘了 `stream=True` 在 requests
```python
resp = requests.post(url, json={..., "stream": True})          # ← 这是 API 参数
resp = requests.post(url, json={...}, stream=True)              # ← requests 参数，两个都要！
```
两个 `stream` 是不同的东西——一个是给 API 的参数，一个是给 requests 的参数。**两个都要设**。

### 坑 2：把整个响应读到内存再处理
```python
data = resp.json()   # ← 错，这会阻塞到流结束
```
要逐行读，不能 `resp.json()`。

### 坑 3：不处理 `\n\n` 分隔
SSE 标准是两个换行分隔消息。`iter_lines()` 会给你单行——但有些消息可能跨行。

### 坑 4：不在前端用 fetch + ReadableStream
后端会流了，但前端用 `await response.json()` 还是一次性读完——白搭。
前端要用 `fetch` + `ReadableStream` 或 `EventSource`。**这部分不用现在做，但要意识到**。

### 坑 5：忽略错误响应
流模式下，错误响应**不是流**——是个普通 JSON 错误。
要先检查 status_code，再开始迭代。

---

## 学完后

在 [my_notes.md](./my_notes.md) 加一段：
- 我之前对流式的误解是 ___
- 现在我知道流式的本质是 ___
- 一个用户体验改进：在我的项目里，应该在哪里用流式？___

---

## 🎉 5 个练习全部完成！

如果你认真做完了这 5 个练习，**你应该**：

- ✅ 看到一个 LLM 调用，能在脑子里画出 HTTP 请求/响应
- ✅ 能解释 temperature、system prompt 实际在做什么
- ✅ 能维护多轮对话，知道历史怎么传
- ✅ 看到一段文本，能秒估 token 数和成本
- ✅ 能实现流式输出，理解为什么这么设计

**这就是真正的"会用 LLM"的基线**。
比那些"我用过 100 个 AI 项目"但讲不清这些的人扎实 10 倍。

---

## 接下来怎么走

打开 [my_notes.md](./my_notes.md)，写一篇 **"5 个练习总结"**：
- 我之前对 LLM 的最大误解是什么
- 我现在最有信心的部分是什么
- 还想深入学什么

然后告诉我，我们一起看：
- 你之前 AI 写的 `lesson_22/` 或 `my_com_rag/` 代码
- 用你**现在的眼光**，能不能看懂 80%？

**如果你看不懂**——说明要么代码本身有问题（AI 乱写），要么概念还没完全消化。两种情况都值得复盘。

> 完成 5 个练习后，再去看 [../00_foundation/](../00_foundation/) 和 [../01_llm_foundations/](../01_llm_foundations/)，那时候你才看得懂。
