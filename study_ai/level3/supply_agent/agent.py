"""supply_agent：供应链智能助手（MCP 版）

架构（甲方乙方各就各位）：
  agent.py（我，乙方/客户端）
    ├── 经 MCPAdapter 拉起 4 个本地 MCP Server 子进程（甲方，各管一本账）
    │     ├── inventory_server.py   库存账（只读）
    │     │     └── FastMCP over stdio：工具调用走 JSON-RPC 管道
    │     ├── supplier_server.py    供应商账（只读）
    │     ├── logistics_server.py   运单账（只读）
    │     └── procurement_server.py 采购单账（读写，带确认闸门）
    └── LangChain create_agent（qwen3.7-flash）+ 拿到的 9 把 MCP 工具 → 终端聊天

运行：conda activate study_ai && python agent.py
"""
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.mcp import MCPAdapter
from langchain_openai import ChatOpenAI

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")  # DASHSCOPE_API_KEY，学 FoodRAG

SERVER_NAMES = ("inventory", "supplier", "logistics", "procurement")

# 【学徒注】关键坑：command 必须用 sys.executable（当前解释器的绝对路径），
# 不能写裸 "python"——MCP Server 是子进程，裸 "python" 由 PATH 说了算，
# 容易飘到系统 Python（那里没装 mcp 包，子进程秒死还难排查）
MCP_CONFIG = {"mcpServers": {
    name: {"command": sys.executable,
           "args": [str(BASE_DIR / "servers" / f"{name}_server.py")],
           "transport": "stdio"}
    for name in SERVER_NAMES}}

SYSTEM_PROMPT = """你是「供小链」，一家小型工厂的供应链智能助手。你通过 MCP 工具查询和操作工厂的四本账：库存、供应商、物流、采购订单。

工作规则：
1. 回答库存、供应商、物流、订单问题前，先调用对应工具查账，绝不凭空编造数字。工具说"未找到"就如实告诉用户。
2. 用户问题含糊时（比如只说"轴承"没给 SKU），先用搜索类工具查，再向用户确认。
3. 涉及下单或取消订单（写操作）：工具带 confirmed 参数。你必须先向用户复述订单要素（哪家供应商、什么物料、数量），取得用户明确同意后，才带 confirmed=true 调用。用户没同意之前绝不确认。
4. 回答用简洁中文，数字带单位。答完可以顺带提一句相关的提醒（比如库存低于安全线）。
5. 和供应链无关的问题，礼貌说明你只管供应链，不瞎聊。"""


def build_model() -> ChatOpenAI:
    return ChatOpenAI(
        model="qwen3.7-flash",  # FoodRAG 同款，DashScope 兼容模式
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0,
    )


def _short(text: str, limit: int = 160) -> str:
    """trace 截断器：工具返回值可能很长，屏幕上只留头部"""
    return text if len(text) <= limit else text[:limit] + " …"


async def chat(agent) -> None:
    print("供小链已上线（4 个 MCP Server：库存/供应商/物流/采购）。空回车退出。")
    # 【学徒注】对话历史手动记账（学 FoodRAG ask.py 金鱼记忆的教训）：
    # create_agent 本身不记上一轮——不给它喂全历史，用户说"确认"时它会失忆，
    # 上一轮复述的订单要素就断线了。这里每轮把历史 + 新问题一起喂进去，
    # 再把这轮产生的新消息（含工具调用记录）追加进历史。
    history: list = []
    while True:
        try:
            question = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break

        print("供小链>", end=" ", flush=True)
        # astream + stream_mode="updates"：每步产出一次增量，
        # 我们在这里截获 AIMessage 的 tool_calls（它要干什么）
        # 和 ToolMessage 的结果（账本回了什么）——延续 trace 教学传统
        final_text = ""
        new_msgs: list = []
        try:
            async for update in agent.astream(
                    {"messages": history + [{"role": "user", "content": question}]},
                    stream_mode="updates"):
                for node_out in update.values():
                    for msg in node_out.get("messages", []):
                        new_msgs.append(msg)  # 收集本轮全部新消息，滚进历史
                        for tc in getattr(msg, "tool_calls", None) or []:
                            print(f"\n  [调用工具] {tc['name']}({tc['args']})")
                        if getattr(msg, "type", "") == "tool":
                            print(f"  [账本回复] {_short(str(msg.content))}")
                        # 不带 tool_calls 的 AIMessage = 收工轮，content 就是答案
                        if getattr(msg, "type", "") == "ai" and not getattr(msg, "tool_calls", None):
                            final_text = msg.content
        except Exception as e:
            print(f"\n[出错] {type(e).__name__}: {e}")
            continue
        history.extend(new_msgs)  # 失败轮不进账，成功轮全进账（含工具轨迹）
        if final_text:
            print(final_text)
    print("\n供小链下班。")


async def main() -> None:
    async with MCPAdapter(MCP_CONFIG) as adapter:
        tools = await adapter.list_tools()
        print("已挂载 MCP 工具：")
        for t in tools:
            print(f"  - {t.name}")
        agent = create_agent(build_model(), tools, system_prompt=SYSTEM_PROMPT)
        await chat(agent)


if __name__ == "__main__":
    asyncio.run(main())
