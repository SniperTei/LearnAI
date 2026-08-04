# 00_start_here — 新手村

> **这里才是真正的起点。**
> 不要跳到 03_rag_deep、05_evaluation 那些——那里是"深化"，不是"入门"。

---

## 为什么有这个目录

你之前的学习路径有问题：
- 跳过了基础（Python 数据结构、HTTP、JSON）
- 直接做了"看起来很厉害"的项目（Streamlit 大屏、my_com_rag）
- 大部分代码是 AI 写的，你能跑但讲不清

**这个目录的目标**：用 5 个练习，把"调过 API"变成"真的懂 LLM 在干什么"。

---

## 学习方式（重要）

**纯动手**：我出题 + 给提示，**你自己写代码**，写完用人话讲给我听。

规则：
1. **不许复制 AI 给的代码**——哪怕我写了示例
2. **每写完一段，必须能回答**："这段代码在做什么？为什么这么写？"
3. **答不上来就停下来问**——不要硬着头皮往下走
4. **慢就是快**——5 个练习可能要 1–2 周，没关系

---

## 5 个练习总览

| # | 练习 | 你做完应该能讲清 | 预计时间 |
|---|------|----------------|---------|
| 1 | [裸 HTTP 调用](./exercise_01_http_call.md) | LLM API 本质上是什么？请求/响应长什么样？ | 2–3 小时 |
| 2 | [参数与角色](./exercise_02_params_and_roles.md) | temperature/system prompt 到底改了什么？ | 3–4 小时 |
| 3 | [多轮对话](./exercise_03_multi_turn.md) | 为什么 LLM 没有"记忆"？历史怎么传？ | 3–4 小时 |
| 4 | [Token 与成本](./exercise_04_tokens_and_cost.md) | "字"和 token 有什么区别？一次调用多少钱？ | 2–3 小时 |
| 5 | [流式输出](./exercise_05_streaming.md) | 流式为什么快？前端怎么消费？ | 3–4 小时 |

**总计 15–20 小时**。慢，但扎实。

---

## 准备工作

### 选一个 LLM 提供商

| 选项 | 优点 | 适合 |
|------|------|------|
| **DeepSeek** | 便宜、OpenAI 兼容、国内访问方便 | ⭐ 推荐（学习用） |
| 通义千问（dashscope） | 你已经用过 | OK |
| Moonshot（Kimi） | 长上下文好 | OK |
| OpenAI / Claude | 文档最全 | 需要海外卡 |

**强烈推荐 DeepSeek**：
- 接口和 OpenAI 100% 兼容（学到的知识能迁移）
- 极便宜（练完 5 个练习总成本 < ¥1）
- 国内直连

### 准备环境

```bash
# 一个干净的目录
mkdir fde/00_start_here/my_code
cd fde/00_start_here/my_code

# 虚拟环境
python -m venv .venv
source .venv/bin/activate

# 装最小依赖
pip install requests python-dotenv

# 不要装 openai / dashscope SDK —— 练习 1 必须用 requests 裸调
# SDK 在练习 3 之后再用
```

### 准备 API Key

```bash
# 创建 .env 文件（不要提交到 git！）
echo "API_KEY=sk-xxxxxxxxxxxxx" > .env
echo "BASE_URL=https://api.deepseek.com" >> .env
```

`.env` 已经在根 `.gitignore` 里，不会泄露。

---

## 学习笔记

每完成一个练习，在 [my_notes.md](./my_notes.md) 里写一段：
- 我学到了什么（用自己话）
- 哪里卡住了
- 还有什么不懂

**这一步比写代码本身更重要**——是把"用过"变成"理解"的关键。

---

## 现在开始

打开 [exercise_01_http_call.md](./exercise_01_http_call.md)。

> 第一条提醒：**别看答案，别抄代码**。卡 30 分钟再问我。
