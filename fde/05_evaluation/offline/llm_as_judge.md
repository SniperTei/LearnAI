# LLM-as-Judge：用大模型当评委

> LLM-as-Judge 是 FDE 最常用的评估方法。
> 便宜、快、可定制——但**有坑**。学清楚再用。

---

## 为什么用 LLM 当评委

### 传统评估的痛点
- **人工标注**：贵、慢、不可重复
- **规则匹配**：覆盖不了开放生成
- **Embedding 相似度**：只能判"像不像"，不能判"好不好"
- **BLEU / ROUGE**：对生成质量几乎无效（已被论文证实）

### LLM-as-Judge 的优势
- ✅ 可以评估主观维度（清晰度、有用性、礼貌）
- ✅ 可以解释为什么打这个分（输出 reasoning）
- ✅ 便宜（用 GPT-4o-mini 跑 100 条 < $1）
- ✅ 可定制（你的维度你定义）

### 代价
- ❌ LLM 自己有偏差（喜欢长答案、喜欢自己说过的内容）
- ❌ 不同模型 judge 结果不一致
- ❌ 主观维度上和人类一致性有限（≈ 0.7–0.85）

---

## LLM-as-Judge 的 3 种主要模式

### 模式 1：打分（Scoring）
给一个回答打 1–5 分。

```python
prompt = """
你是评估专家。请给下面 AI 的回答打分（1–5）。

用户问题: {question}
AI 回答: {answer}

评分维度: 有用性
1 = 完全没用 / 答非所问
3 = 部分有用
5 = 完美回答

请输出 JSON:
{{"score": <int>, "reason": "<string>"}}
"""
```

**优点**：直观，可比性强
**缺点**：分数有相对性（同样是 4 分，含义可能不同）

### 模式 2：两两对比（Pairwise）
A 和 B 哪个更好？

```python
prompt = """
用户问题: {question}
回答 A: {answer_a}
回答 B: {answer_b}

哪个更好？输出: "A" / "B" / "tie"
并说明理由（一句话）。
"""
```

**优点**：比打分更准（人对"A 比 B 好"判断比对绝对分敏感）
**缺点**：N 个候选要 N² 次对比，贵

详见 [pairwise_vs_scoring.md](./pairwise_vs_scoring.md)

### 模式 3：多维评估（Rubric）
按预设标准逐项评估。

```python
prompt = """
请按以下 rubric 评估 AI 回答:

维度 1: 准确性（0–2 分）
  0 = 有事实错误
  1 = 部分准确
  2 = 完全准确

维度 2: 完整性（0–2 分）
  0 = 缺关键信息
  1 = 基本完整
  2 = 完整

维度 3: 礼貌（0–1 分）
  0 = 不礼貌
  1 = 礼貌

用户问题: {question}
AI 回答: {answer}

输出 JSON: {{"accuracy": <int>, "completeness": <int>, "politeness": <int>, "notes": <string>}}
"""
```

**优点**：可定制、可解释、维度独立
**缺点**：rubric 设计本身是门手艺

**推荐**：FDE 默认用**多维评估**，因为它最贴近"业务质量"。

---

## 写好一个 Judge Prompt 的 5 个原则

### 1. **明确评分维度，不要让 LLM 自由发挥**
❌ "请评估这个回答" → 模型自己定义"好"，每次不一样
✅ "请按 [准确性、完整性、礼貌] 三维度评分"

### 2. **每个分数有清晰定义（rubric）**
```
准确性:
  0 = 有明显事实错误
  1 = 大方向对，细节有错
  2 = 完全准确
```
而不是"1–5 分"——5 分到底是啥？

### 3. **强制结构化输出**
- 要求输出 JSON
- 用 `response_format={"type": "json_object"}` 或 tool call 强制

### 4. **要求给出 reasoning**
- 让 LLM 解释为什么打这个分
- 便于人工审核 + 改进 judge prompt

### 5. **加 CoT（Chain of Thought）**
让模型先分析，再打分：

```
请按以下步骤评估:
1. 列出回答的关键信息点
2. 检查每条信息是否准确
3. 检查是否完整覆盖问题
4. 综合给出分数
```

**注意**：CoT 会让 token 成本上升 2–3 倍，但准确性显著提升。

---

## LLM-as-Judge 的典型偏差

### 偏差 1：长度偏好（Verbosity Bias）
LLM 倾向于给长答案打高分。
**对策**：在 prompt 里明确说"长度不影响评分"，或单独评"简洁性"。

### 偏差 2：自我偏好（Self-enhancement Bias）
GPT-4 当 judge 时，倾向给 GPT 系答案打高分。
**对策**：跨模型对比时，用第三方模型或人工校准。

### 偏差 3：位置偏好（Position Bias）
Pairwise 时，LLM 倾向第一个/第二个位置。
**对策**：交换 A/B 位置各跑一次，看是否一致。

### 偏差 4：迟钝（Sensitivity）
对小改动不敏感——比如把"15 天"改成"20 天"，judge 可能给同样的分。
**对策**：关键 fact 用规则匹配，不交给 judge。

### 偏差 5：不懂拒绝（Inability to Detect Refusal）
模型没回答（说"我不知道"），judge 可能仍然给中等分。
**对策**：先单独检测拒答，再交给 judge。

---

## 选什么模型当 Judge

| Judge 模型 | 成本 | 准确度 | 适用 |
|-----------|------|--------|------|
| GPT-4o | 高 | 高 | 关键评估、发布前 |
| GPT-4o-mini | 低 | 中 | 日常开发迭代 |
| Claude Sonnet | 中 | 高 | 长文本、推理任务 |
| Claude Haiku | 低 | 中 | 日常 |
| Llama 3 / Qwen 本地 | 仅硬件 | 视模型 | 数据不能出境 |
| Gemini Pro | 中 | 中-高 | 多模态评估 |

**经验法则**：
- **Judge 模型应该比被评估的模型更强，或至少同级**
- 评估 GPT-4 的输出，别用 GPT-4o-mini 当 judge
- 中文场景：Claude Sonnet / GPT-4o 表现稳定

---

## 实战：从零搭一个 LLM-as-Judge

### Step 1：定义评估维度
和业务方聊清楚："什么样的回答算好？"
- 客服场景：准确性、解决率、礼貌
- 法律场景：准确性、引用正确性、谨慎度
- 教育场景：知识点对、引导性、不直接给答案

### Step 2：写 judge prompt
模板：

```
你是 {domain} 领域的评估专家。
请按以下维度评估 AI 的回答。

【用户问题】
{question}

【AI 回答】
{answer}

【参考信息】（可选）
{context_or_ground_truth}

【评估维度】
{rubric_table}

【输出格式】
JSON: {{"dim1": <score>, "dim2": <score>, ..., "overall": <score>, "reason": "<text>"}}
```

### Step 3：人工校准
- 拿 20 条数据，人工 + LLM 同时打分
- 算一致性（Cohen's Kappa 或简单算同意率）
- 一致性 < 0.6 → 改 rubric
- 一致性 0.6–0.8 → 可用
- 一致性 > 0.8 → 良好

详见 [human_alignment.md](./human_alignment.md)

### Step 4：集成到 CI
每次改 prompt 或换模型，自动跑 judge。
不合格的 case 进 review queue。

---

## 一个最小可用的 Judge 类（Python）

```python
import json
from openai import OpenAI

class LLMJudge:
    def __init__(self, model="gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model

    def evaluate(self, question, answer, rubric, context=None):
        prompt = self._build_prompt(question, answer, rubric, context)
        resp = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
            temperature=0,  # 评估要稳定
        )
        return json.loads(resp.choices[0].message.content)

    def _build_prompt(self, q, a, rubric, ctx):
        ctx_block = f"\n【参考信息】\n{ctx}" if ctx else ""
        return f"""你是评估专家。按 rubric 给 AI 回答打分。

【用户问题】
{q}

【AI 回答】
{a}{ctx_block}

【评估维度和评分标准】
{rubric}

【要求】
1. 先简短分析（不超过 100 字）
2. 输出 JSON: {{"analysis": "...", "scores": {{...}}, "overall": 1-5}}
"""
```

**用法**：

```python
judge = LLMJudge(model="gpt-4o-mini")

rubric = """
准确性 (0-2):
  0 = 有事实错误
  1 = 大体对
  2 = 完全正确
完整性 (0-2):
  0 = 缺关键信息
  1 = 基本完整
  2 = 完整覆盖
"""

result = judge.evaluate(
    question="年假多少天？",
    answer="入职享 15 天年假。",
    rubric=rubric,
    context="员工手册：入职享 15 天年假"
)
```

---

## LLM-as-Judge 的边界

**适合**：
- 开放生成评估（客服回答、文档摘要、文案）
- 多维度质量评估
- Pairwise 对比

**不适合**：
- 精确事实核查（用规则 / 知识库匹配）
- 安全合规审查（用专门 classifier）
- 高频低成本评估（用 embedding + 抽样 judge）

---

## 自测题

1. 写一个评估"客服回答质量"的 rubric，至少 3 个维度
2. 如果发现 judge 给所有答案都打 4 分，你怎么调试？
3. Pairwise 比 Scoring 准，为什么不全用 Pairwise？
4. 你的 judge 和人工一致性 0.5，能直接上线吗？
5. 用 GPT-4o-mini 当 judge 评估 GPT-4o 的输出，合理吗？

---

> 下一步：[pairwise_vs_scoring.md](./pairwise_vs_scoring.md)
