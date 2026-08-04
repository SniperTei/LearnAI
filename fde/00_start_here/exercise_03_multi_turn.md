# 练习 3：多轮对话（LLM 的"记忆"幻觉）

> 目标：搞懂一个最反直觉的事实——**LLM 没有记忆**。
> 我们看到的"多轮对话"，是每次把历史塞进 messages 里重新算。

---

## 学习目标

做完你应该能用人话讲清：

1. 为什么说 LLM 是"无状态的"？
2. 我和 LLM 进行了 5 轮对话，第 6 轮时 API 实际收到的 messages 长什么样？
3. 为什么多聊几轮，响应会变慢、变贵？
4. "LLM 忘了前面说的话" 是怎么发生的？
5. 如果我要做一个聊天机器人，怎么维护历史？

---

## 任务

### 任务 3.1：做一个能聊天的 CLI

写 `chat.py`：
- 终端输入一句话 → 调 API → 打印回复 → 等下一句
- **关键**：维护一个 `messages` 列表，每次把新的 user 和 assistant 消息追加进去

伪代码结构：
```python
messages = [{"role": "system", "content": "..."}]

while True:
    user_input = input("你: ")
    if user_input == "quit": break

    # 1. 把 user 输入加进 messages
    # 2. 把整个 messages 发给 API
    # 3. 拿到 assistant 回复
    # 4. 打印回复
    # 5. 把 assistant 回复也加进 messages（重要！）
```

**测一下**：
- "我叫张三"
- "1+1 等于几"
- "我刚才说我叫什么？"  ← 看模型能不能答上来

### 任务 3.2：打印每次请求的 messages

在发请求前，把整个 `messages` 打印出来。
**观察**：
- 第 1 轮：messages 有几条？
- 第 3 轮：几条？
- 第 10 轮：几条？messages 总长度（字符数）多少？

### 任务 3.3：故意让模型"失忆"

写第二个脚本 `amnesia.py`：
- **只**发当前这条 user 消息（不附带历史）
- 问 "我叫什么" → 模型答不上来
- 这证明了什么？

### 任务 3.4：估算历史会膨胀多大

写一个统计函数，每次请求后打印：
- messages 总字符数
- 如果用 GPT-4o 的 tokenizer（`pip install tiktoken`），算 messages 总 token 数
- 假设每条 user/assistant 平均 50 字，10 轮 / 50 轮 / 100 轮 各多少 token？

**意识到**：聊天越久，单次请求的 token 越多 → 越贵、越慢。

### 任务 3.5（进阶）：滑动窗口

如果消息太多，把最早的丢掉（保留 system 和最近 N 轮）：
- 实现"只保留最近 5 轮"
- 测试：5 轮后再问"我叫什么" → 还记得吗？
- 这就是最朴素的"对话管理"

---

## 提示

### LLM 的核心事实

```
LLM 是无状态的。
每次调用都是独立的事件。
它"记得"什么，是因为你**每次都把历史重新发过去**。
```

API 调用流程：
```
Turn 1:
    请求 messages: [system, user1]
    响应: assistant1

Turn 2:
    请求 messages: [system, user1, assistant1, user2]   ← 历史被重发
    响应: assistant2

Turn 3:
    请求 messages: [system, user1, assistant1, user2, assistant2, user3]
    ...
```

### 消息角色

- `system`：全局设定（一般只 1 条，放最前面）
- `user`：用户说的
- `assistant`：模型之前说的（**必须自己把上一轮回复加回去**）

新手最容易忘的就是**把 assistant 回复加进 messages**——结果模型看不到自己刚才说过啥。

### Token 计数

```python
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4o")
n = len(enc.encode("一段中文文本"))
print(n)
# 注意：中文每字通常 1-2 token，英文每词约 1-1.5 token
```

DeepSeek 用的是自己的 tokenizer，可以近似用 GPT 的估，准确数看官方文档。

---

## 检查点

合上代码，回答：

1. 我的 chat.py 第 5 轮时，发出去的 messages 一共有几条？
2. 我故意"失忆"的实验，模型答不上"我叫什么"——这说明什么？
3. 假设每条消息 100 token，聊 50 轮，第 50 轮的请求里有多少 token？
4. 如果一个用户聊了 200 轮还没退出，我的系统会出现什么问题？
5. 滑动窗口（只留最近 5 轮）的副作用是什么？

---

## 常见坑

### 坑 1：忘记把 assistant 回复加入 messages
```
请求: [system, user1, user2, user3]   ← 错
请求: [system, user1, assistant1, user2, assistant2, user3]   ← 对
```
少了 assistant 历史，模型答非所问。

### 坑 2：每轮重发历史 → 成本累积
聊 50 轮不是 50 次普通调用，是 50 次调用 + 历史总和。
总成本 ≈ Σ(前 N 轮的 token) ≈ O(N²)。

### 坑 3：滑动窗口丢 system
"我留最近 5 条" → 把 system 也丢了 → 模型行为变了。
**永远保留 system**，只裁剪 user/assistant。

### 坑 4：长对话超出 context window
对话太长 → messages 超过模型最大 context → API 报错。
需要监控 + 主动裁剪 / 摘要。

### 坑 5：以为 API key 关联会自动记住
"我用的同一个 API key，模型应该记得我吧？"——**不会**。
API key 只是身份验证，不存对话历史。
（部分平台有"Assistant API"会存，但那是平台层功能，不是 LLM 本身的能力。）

---

## 学完后

在 [my_notes.md](./my_notes.md) 加一段：
- 我之前以为 LLM 的"记忆"是 ___
- 现在我知道是 ___
- 这让我理解了为什么 ChatGPT 长聊后会"忘事"：___

---

> 下一步：[exercise_04_tokens_and_cost.md](./exercise_04_tokens_and_cost.md)
