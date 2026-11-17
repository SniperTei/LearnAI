"""supply_agent：供应链智能助手（MCP 版，中间商模式）

架构（甲方乙方各就各位，按角色挂载）：
  agent.py（我，乙方/客户端）
    ├── 经 MCPAdapter 拉起 MCP Server 子进程（甲方，各管一本账）
    │     ├── inventory_server.py   库存账（读 + 收货入库/销售出库写）
    │     ├── supplier_server.py    供应商账（只读）      ┐ 买手侧
    │     ├── procurement_server.py 采购单账（写，闸门）   ┘
    │     ├── customer_server.py    客户账（只读）        ┐ 卖手侧
    │     ├── sales_server.py       销售单账（写，闸门）   ┘
    │     └── logistics_server.py   运单账（只读）        ← 两头共用的核心
    └── LangChain create_agent（qwen3.7-flash）+ 挂到的 MCP 工具 → 终端聊天

【APP_ROLE 挂载即开关】（需求 9：进货出货可选可拆）：
  trader（默认）= 核心两座 + 买手侧 + 卖手侧（中间商，又买又卖）
  buyer         = 核心两座 + 买手侧（只买不卖——LLM 工具名片里没有销售
                  工具，"卖货"它想干也没门，只能礼貌拒绝）
  seller        = 核心两座 + 卖手侧（只卖不买）
只改挂载配置和 system prompt，不改任何 Server 代码——"不挂载 = 不存在"。

运行：conda activate study_ai && python agent.py
角色：APP_ROLE=buyer python agent.py（trader/buyer/seller，默认 trader）
传输：MCP_TRANSPORT=inventory:http python agent.py（L2 混合传输——
      逗号分隔可多座，被点名的座走 HTTP 常驻，其余照旧 stdio 子进程；
      前提：被点名的座已自己 --http 跑起来，.env 里有 MCP_API_KEY）
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

# 角色分组（需求 9 的架构落地）：
CORE = ("inventory", "logistics")            # 中间环节：仓库（双向写）+ 物流
BUY_SIDE = ("supplier", "procurement")       # 买手侧：供应商情报 + 采购写账
SELL_SIDE = ("customer", "sales")            # 卖手侧：客户情报 + 销售写账
ROLE_SERVERS = {
    "trader": CORE + BUY_SIDE + SELL_SIDE,   # 中间商：两头都做
    "buyer": CORE + BUY_SIDE,                # 只采购不销售
    "seller": CORE + SELL_SIDE,              # 只销售不采购
}

# 角色说明书：账本清单 + 业务边界，按角色拼进 system prompt
ROLE_ACCOUNTS = {
    "trader": "六本账：库存、供应商、物流、采购订单、客户、销售订单",
    "buyer": "四本账：库存、供应商、物流、采购订单",
    "seller": "四本账：库存、客户、物流、销售订单",
}
ROLE_BUSINESS = {
    "trader": "本公司是中间商：向供应商采购进货，也向客户销售出货，赚差价。",
    "buyer": "本公司只采购不销售——不从任何客户手里接销售单，用户要卖货时礼貌说明本公司不做销售。",
    "seller": "本公司只销售不采购——不从任何供应商进货，用户要采购时礼貌说明本公司不做采购。",
}

# 各座 HTTP 端口（与 servers/http_entry.py 的默认端口必须一致，改要两边改）
SERVER_PORTS = {
    "inventory": 8101, "supplier": 8102, "logistics": 8103,
    "procurement": 8104, "customer": 8105, "sales": 8106,
}


def _http_servers() -> set:
    """解析 MCP_TRANSPORT=inventory:http,sales:http → {"inventory", "sales"}

    【学徒注】L2 混合传输的机关：默认 stdio（agent 当子进程拉起，用完即走），
    被点名的座改走 HTTP 常驻（跨进程共享、可独立重启、可跨机）。
    两边对 LLM 完全透明——挂到的都是同一批工具，名片一字不差，
    这就是协议标准化的红利：换传输只改配置，不改任何业务代码。
    """
    raw = os.environ.get("MCP_TRANSPORT", "").strip()
    http = set()
    for item in filter(None, (p.strip() for p in raw.split(","))):
        name, sep, mode = item.partition(":")
        if sep and mode.strip() == "http":
            http.add(name.strip())
    return http


def build_mcp_config(server_names: tuple) -> dict:
    """按角色挑 Server 生成挂载配置（挂载即开关的唯一机关）。

    传输按座自选（L2）：MCP_TRANSPORT 点名的座走 HTTP url + Bearer 认证头，
    其余照旧 stdio 子进程。"""
    http_servers = _http_servers()
    if http_servers:
        api_key = os.environ.get("MCP_API_KEY")
        if not api_key:
            # fail fast：HTTP 座的钥匙没有 = 连不上还不明所以，不如当场报清
            sys.exit("[配置错误] MCP_TRANSPORT 点了 HTTP 座，但 .env 里没有 MCP_API_KEY")
        unknown = http_servers - set(SERVER_PORTS)
        if unknown:
            sys.exit(f"[配置错误] MCP_TRANSPORT 里没有这座 Server：{', '.join(sorted(unknown))}")

    servers = {}
    for name in server_names:
        if name in http_servers:
            # 【学徒注】HTTP 形态：url 指向常驻进程 + 认证头带钥匙。
            # 和 stdio 条目住同一张配置表——MCPAdapter 只认配置不认传输方式
            servers[name] = {
                "url": f"http://127.0.0.1:{SERVER_PORTS[name]}/mcp",
                # 【学徒注】踩坑：fastmcp 配置校验要连字符 "streamable-http"，
                # 下划线 "streamable_http" 直接 pydantic 报 literal_error
                "transport": "streamable-http",
                "headers": {"Authorization": f"Bearer {os.environ['MCP_API_KEY']}"},
            }
        else:
            # 【学徒注】关键坑：command 必须用 sys.executable（当前解释器的
            # 绝对路径），不能写裸 "python"——MCP Server 是子进程，裸 "python"
            # 由 PATH 说了算，容易飘到系统 Python（那里没装 mcp 包，子进程秒死还难排查）
            servers[name] = {"command": sys.executable,
                             "args": [str(BASE_DIR / "servers" / f"{name}_server.py")],
                             "transport": "stdio"}
    return {"mcpServers": servers}


def build_system_prompt(role: str) -> str:
    return f"""你是「供小链」，一家小型贸易公司的供应链智能助手。你通过 MCP 工具查询和操作公司的{ROLE_ACCOUNTS[role]}。
{ROLE_BUSINESS[role]}

工作规则：
1. 回答库存、供应商、客户、物流、订单问题前，先调用对应工具查账，绝不凭空编造数字。工具说"未找到"就如实告诉用户。
2. 用户问题含糊时（比如只说"轴承"没给 SKU），先用搜索类工具查，再向用户确认。
3. 涉及下单、取消订单、收货入库、销售出库（写操作）：工具带 confirmed 参数。你必须先向用户复述订单要素（哪家公司、什么物料、数量），取得用户明确同意后，才带 confirmed=true 调用。用户没同意之前绝不确认。
4. 回答用简洁中文，数字带单位。答完可以顺带提一句相关的提醒（比如库存低于安全线）。
5. 和本公司业务无关的要求，礼貌说明做不了，不瞎聊。"""


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


async def chat(agent, role: str, server_names: tuple) -> None:
    servers_desc = "/".join(server_names)
    print(f"供小链已上线（角色 {role}，{len(server_names)} 个 MCP Server：{servers_desc}）。空回车退出。")
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
    # 【学徒注】挂载即开关：角色只决定两件事——挂哪几座 Server、
    # system prompt 里怎么介绍公司。Server 代码一行不动
    role = os.environ.get("APP_ROLE", "trader")
    if role not in ROLE_SERVERS:
        print(f"[配置错误] 未知 APP_ROLE={role}（可选：trader / buyer / seller）")
        return
    server_names = ROLE_SERVERS[role]
    http_servers = _http_servers() & set(server_names)
    if http_servers:
        print(f"HTTP 常驻座：{'/'.join(sorted(http_servers))}（其余 stdio 子进程）")

    async with MCPAdapter(build_mcp_config(server_names)) as adapter:
        tools = await adapter.list_tools()
        print(f"已挂载 MCP 工具（APP_ROLE={role}）：")
        for t in tools:
            print(f"  - {t.name}")
        agent = create_agent(build_model(), tools,
                             system_prompt=build_system_prompt(role))
        await chat(agent, role, server_names)


if __name__ == "__main__":
    asyncio.run(main())
