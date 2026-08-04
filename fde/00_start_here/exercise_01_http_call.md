# 练习 1：裸 HTTP 调用 LLM API

> 目标：理解 LLM API 本质上就是**一个 HTTP POST 请求**。
> 没有任何魔法。

---

## 学习目标

做完你应该能用人话讲清：

1. "调一次 LLM" 到底发生了什么？画出整个流程
2. 请求 body 里有哪些字段？每个字段干嘛的？
3. 响应 body 里有哪些字段？我们用的实际上是哪部分？
4. 为什么 `requests.post` 要带 headers？headers 里有什么？
5. 为什么 API key 不能写死在代码里？

**如果你答不上来，别往下走。**

---

## 任务

### 任务 1.1：用 `requests` 库调用 DeepSeek（或任何 OpenAI 兼容接口）

**只允许用**：`requests` + `python-dotenv`。
**禁止用**：`openai` SDK、`dashscope` SDK。

写一个 `chat.py`，做的事：
1. 从 `.env` 读 API key
2. 发 POST 请求到 `/v1/chat/completions`
3. 打印出 LLM 的回复

### 任务 1.2：把响应完整打印出来

不光打印"回复内容"——把整个响应 JSON 打印出来（用 `json.dumps(resp.json(), indent=2, ensure_ascii=False)`）。

观察：
- 顶层有哪些字段？（`id`、`object`、`created`、`model`、`choices`、`usage`...）
- 真正的回复文本在哪个嵌套路径下？
- `usage` 里有什么？数字大概多少？

### 任务 1.3：故意制造错误

- 把 API key 改错 → 看响应是什么（状态码？body 长啥样？）
- 把 `model` 名字写错 → 看响应
- 把 body 格式写错（少字段、字段拼错）→ 看响应

把每种错误的"症状"记下来。

---

## 提示（不是答案）

### 关于请求结构

LLM API（OpenAI 兼容）的请求大概长这样：

```
POST https://api.deepseek.com/v1/chat/completions
Headers:
    Content-Type: application/json
    Authorization: Bearer sk-xxxxxxx
Body (JSON):
    {
        "model": "...",
        "messages": [
            {"role": "user", "content": "..."}
        ]
    }
```

**关键字段**：
- `model`：用哪个模型（deepseek-chat / deepseek-reasoner 等）
- `messages`：消息列表，每条有 `role` 和 `content`

**先不要管**：`temperature`、`stream`、`max_tokens`（练习 2 再处理）

### 关于 requests.post

```python
import requests
resp = requests.post(
    url,
    headers={...},
    json={...},   # 用 json= 参数，requests 自动序列化 + 设 Content-Type
)
# resp.status_code  → 状态码
# resp.json()       → 解析 JSON body
# resp.text         → 原始文本（debug 用）
```

### 关于从 .env 读 key

```python
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("API_KEY")
```

---

## 检查点（自问）

写完后**合上代码**，回答：

1. 我刚才发的请求 URL 是什么？方法是什么？
2. body 里我传了哪几个字段？
3. 响应状态码是多少？为什么是这个数？
4. 我最终给用户看的字符串，从响应 JSON 里怎么取出来？写出来（如 `resp["choices"][0]["message"]["content"]`）
5. 如果不传 Authorization header 会怎样？

**答不出任何一个 → 重做**。

---

## 常见坑

### 坑 1：忘记 `Bearer` 前缀
```
Authorization: Bearer sk-xxx     ← 正确
Authorization: sk-xxx            ← 错误，会 401
```

### 坑 2：URL 路径写错
- 错：`https://api.deepseek.com`（少了路径）
- 错：`https://api.deepseek.com/chat/completions`（少了 `/v1`）
- 对：`https://api.deepseek.com/v1/chat/completions`

各家厂商路径前缀不同，**看官方文档**。

### 坑 3：忘了 `json=` 用了 `data=`
```python
requests.post(url, data={...})       # 错 → 发的是 form 数据
requests.post(url, json={...})       # 对 → 发的是 JSON
```

### 坑 4：把 API key 写代码里
**永远不要**把 key 写在 `.py` 文件里然后提交 git。
- 用 `.env`
- 确认 `.env` 在 `.gitignore` 里

### 坑 5：不处理 HTTP 错误
```python
resp = requests.post(...)
resp.raise_for_status()   # 状态码不是 2xx 就抛错
data = resp.json()
```
否则错误响应也会被当成功处理。

---

## 学完后

在 [my_notes.md](./my_notes.md) 加一段：
- 今天我学到的最重要的一点
- 我之前以为 LLM API 是 ___，现在我知道它是 ___
- 还有什么不懂（列出来）

然后告诉我你完成了，**用人话给我讲一遍流程**。我点评后才进入练习 2。

---

> 下一步：[exercise_02_params_and_roles.md](./exercise_02_params_and_roles.md)
