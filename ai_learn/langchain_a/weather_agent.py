"""
LangChain 天气查询 Agent
使用 LangChain 框架创建智能天气查询助手
支持 DashScope (阿里云通义千问)、Anthropic (Claude) 和 OpenAI
"""
import os
from dotenv import load_dotenv

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from weather_tools import tools

# 加载环境变量
load_dotenv()


def get_llm():
    """
    自动选择可用的 LLM
    优先级: DashScope > Anthropic > OpenAI
    """
    dashscope_key = os.getenv("DASHSCOPE_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if dashscope_key:
        print("🤖 使用阿里云通义千问模型 (DashScope)")
        return ChatTongyi(
            model="qwen-plus",
            temperature=0
        )
    elif anthropic_key:
        print("🤖 使用 Anthropic Claude 模型")
        return ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            temperature=0
        )
    elif openai_key:
        print("🤖 使用 OpenAI GPT 模型")
        return ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0
        )
    else:
        raise ValueError(
            "未找到 API Key！\n"
            "请在 .env 文件中设置以下任意一个：\n"
            "  - DASHSCOPE_API_KEY (阿里云通义千问，推荐)\n"
            "  - ANTHROPIC_API_KEY (Anthropic Claude)\n"
            "  - OPENAI_API_KEY (OpenAI GPT)"
        )


def create_weather_agent():
    """
    创建天气查询 Agent
    使用 LangChain 的 create_agent API (基于 LangGraph)
    """
    # 自动选择 LLM
    llm = get_llm()

    # 系统提示词
    system_prompt = """你是一个友好的天气查询助手。你可以帮助用户查询各地天气和天气预报。

你有以下工具可以使用:
- get_weather: 查询指定城市当前天气。参数: city (城市名称)
- get_forecast: 获取城市未来几天的天气预报。参数: city (城市名称), days (天数，1-3)

使用指南:
1. 当用户问及当前天气时，使用 get_weather 工具
2. 当用户问及未来天气、预报、明天、后天等时，使用 get_forecast 工具
3. 从用户输入中提取城市名称
4. 回复时要友好、简洁，用表情符号让信息更生动

请用中文回复用户。"""

    # 使用 create_agent API (LangChain 1.0+ 的新架构)
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt
    )

    return agent


def main():
    """
    主函数：运行交互式天气查询助手
    """
    print("=" * 60)
    print("🌤️  LangChain 天气查询助手")
    print("=" * 60)
    print("\n你可以问我类似这样的问题:")
    print("  - 北京今天天气怎么样？")
    print("  - 上海明天会下雨吗？")
    print("  - New York 的天气")
    print("  - 查询广州未来3天天气预报")
    print("\n输入 'quit' 或 'exit' 退出\n")
    print("=" * 60)

    # 创建 Agent
    try:
        agent = create_weather_agent()
        print("\n✅ 助手已启动！开始提问吧~\n")
    except Exception as e:
        print(f"\n❌ 初始化失败: {e}")
        print("\n💡 配置提示:")
        print("1. 创建 .env 文件: cp .env.example .env")
        print("2. 编辑 .env，填入你的 API Key:")
        print("   - DashScope: https://dashscope.console.aliyun.com/apiKey")
        print("   - Anthropic: https://console.anthropic.com/settings/keys")
        print("   - OpenAI: https://platform.openai.com/api-keys")
        import traceback
        traceback.print_exc()
        return

    # 交互循环
    thread_id = "session_1"
    while True:
        try:
            user_input = input("\n🤔 你: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                print("\n👋 再见！")
                break

            print("\n🤖 助手: ", end="", flush=True)

            # 调用 agent (LangChain 1.0+ 新格式)
            result = agent.invoke(
                {"messages": [("user", user_input)]},
                config={"configurable": {"thread_id": thread_id}}
            )

            # 提取最后一条消息
            messages = result["messages"]
            last_message = messages[-1]

            if hasattr(last_message, 'content'):
                print(last_message.content)
            else:
                print(str(last_message))

        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 出错了: {e}")
            import traceback
            traceback.print_exc()
            print("\n请检查:")
            print("  - API Key 是否正确")
            print("  - 网络连接是否正常")
            print("  - API 额度是否充足")


def simple_query(query: str):
    """
    简单的查询函数（用于程序化调用）

    Args:
        query: 查询文本

    Returns:
        Agent 的回复
    """
    agent = create_weather_agent()
    result = agent.invoke(
        {"messages": [("user", query)]},
        config={"configurable": {"thread_id": "simple_query"}}
    )

    messages = result["messages"]
    last_message = messages[-1]

    if hasattr(last_message, 'content'):
        return last_message.content
    return str(last_message)


if __name__ == "__main__":
    main()
