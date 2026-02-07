"""
智能客服的不同实现方式对比
"""

print("=" * 70)
print("🤖 智能客服实现方式对比")
print("=" * 70)

print("""
场景: 用户问 "我的订单什么时候到？"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

方式1️⃣: 简单对话型 (❌ 不是 ReAct)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-3.5-turbo")
prompt = ChatPromptTemplate.from_template(
    "你是一个客服助手。用户问题: {question}"
)

chain = prompt | llm
result = chain.invoke({"question": "我的订单什么时候到？"})

特点:
✓ 只有 LLM 对话，没有工具调用
✓ LLM 只能基于训练数据回答
✗ 无法查询实时数据（如订单状态）
✗ 没有"行动"步骤

这不算 ReAct


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

方式2️⃣: RAG 增强型 (❌ 不是 ReAct)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from langchain_openai import ChatOpenAI
from langchain_core.vectorstores import FAISS
from langchain.chains import RetrievalQA

# 从知识库检索相关信息
retriever = vector_store.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    retriever=retriever
)

result = qa_chain.invoke("我的订单什么时候到？")

特点:
✓ 可以检索知识库
✓ 基于企业文档回答
✗ 没有工具调用和行动
✗ 只是"检索 + 生成"

这不算 ReAct


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

方式3️⃣: Agent 工具调用型 (✅ 类似 ReAct)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from langchain.agents import create_agent
from langchain_core.tools import tool

@tool
def check_order_status(order_id: str) -> str:
    \"\"\"查询订单状态\"\"\"
    # 调用订单系统 API
    return f"订单 {order_id} 预计明天送达"

@tool
def process_refund(order_id: str) -> str:
    \"\"\"处理退款\"\"\"
    # 调用退款 API
    return f"订单 {order_id} 退款成功"

agent = create_agent(
    model=ChatOpenAI(),
    tools=[check_order_status, process_refund],
    system_prompt="你是客服助手"
)

# 用户对话
result = agent.invoke({
    "messages": [("user", "我的订单什么时候到？")]
})

工作流程:
[用户] 我的订单什么时候到？
  ↓
[LLM 推理] 需要查询订单信息，但用户没给订单号
  ↓
[LLM 行动] 询问订单号
  ↓
[用户] ORDER12345
  ↓
[LLM 推理] 现在可以查询了，调用 check_order_status
  ↓
[LLM 行动] check_order_status(order_id="ORDER12345")
  ↓
[观察结果] "订单 ORDER12345 预计明天送达"
  ↓
[LLM 最终回复] "您的订单 ORDER12345 预计明天送达，请保持电话畅通"

特点:
✓ LLM 推理下一步该做什么
✓ 调用工具（行动）
✓ 观察工具结果
✓ 循环直到解决问题

✅ 这算 ReAct 模式！


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

方式4️⃣: 复杂客服 Agent (✅ 标准 ReAct)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

用户: "我买的鞋子有质量问题，要退款"

Conversation Log:

Thought: 用户提到要退款，我需要先查询订单详情
Action: check_order[user_id="U123"]
Observation: 找到订单 O456，商品：运动鞋，状态：已发货

Thought: 订单已发货，需要先检查退货政策
Action: check_return_policy[product="运动鞋", condition="质量问题"]
Observation: 质量问题支持全额退款

Thought: 符合退货条件，可以直接处理退款
Action: process_refund[order_id="O456", reason="质量问题"]
Observation: 退款申请已提交，3-5个工作日到账

Thought: 退款已处理，现在可以告知用户
Final Answer: 您的退款申请已提交！由于是质量问题，我们会全额退款，
              金额将在3-5个工作日退回原支付账户。给您带来的不便敬请谅解！

✅ 这完全符合 ReAct 模式！


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 总结对比表
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────────┬────────────┬──────────┬────────┐
│   实现方式   │ LLM 推理   │ 工具调用 │ ReAct? │
├─────────────┼────────────┼──────────┼────────┤
│ 简单对话     │    无      │    无    │   ❌   │
│ RAG 增强     │    简单    │    无    │   ❌   │
│ Agent 工具   │    有      │    有    │   ✅   │
│ 复杂 Agent   │    循环    │    多步  │   ✅   │
└─────────────┴────────────┴──────────┴────────┘


🎯 结论
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

智能客服是否算 ReAct，看是否满足：

1. ✅ LLM 需要推理（判断该做什么）
2. ✅ 有行动步骤（调用工具/API）
3. ✅ 观察结果（基于工具输出）
4. ✅ 循环迭代（多轮推理和行动）

如果只是简单问答 → 不算 ReAct
如果使用了 Agent + 工具 → 算 ReAct 模式

实际项目中，大部分智能客服都是混合方案：
- 简单问题 → 直接 LLM 回答（不算 ReAct）
- 复杂问题 → Agent + 工具（算 ReAct）
""")

print("=" * 70)
