"""
演示 LangChain Function Calling 的实际过程
"""
import os
from dotenv import load_dotenv
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain.agents import create_agent
from weather_tools import tools

load_dotenv()

# 创建 LLM
llm = ChatTongyi(model="qwen-plus", temperature=0)

# 创建 Agent
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="你是一个天气查询助手"
)

print("=" * 60)
print("🔍 Function Calling 调试演示")
print("=" * 60)

# 调用 Agent
print("\n📝 用户输入: '北京今天天气怎么样'\n")
print("-" * 60)

result = agent.invoke(
    {"messages": [("user", "北京今天天气怎么样")]},
    config={"configurable": {"thread_id": "debug"}}
)

print("-" * 60)
print("\n📊 Agent 执行过程（messages 列表）:\n")

for i, msg in enumerate(result["messages"]):
    print(f"\n[消息 {i+1}] 类型: {type(msg).__name__}")
    print(f"内容: {msg.content[:100] if hasattr(msg, 'content') else msg}...")

    # 检查是否有工具调用
    if hasattr(msg, 'tool_calls') and msg.tool_calls:
        print(f"🔧 工具调用: {len(msg.tool_calls)} 个")
        for j, tool_call in enumerate(msg.tool_calls):
            print(f"   [{j+1}] 函数名: {tool_call['name']}")
            print(f"       参数: {tool_call['args']}")

print("\n" + "=" * 60)
print("✅ 完整流程:")
print("   1. 用户输入 → LLM 分析")
print("   2. LLM 决定调用 get_weather(city='北京')")
print("   3. 执行工具函数，获取天气数据")
print("   4. 将结果返回给 LLM")
print("   5. LLM 生成自然语言回复")
print("=" * 60)
