"""库存 MCP Server（查询 + 收货入库写账，stdio 传输）

【学徒注】站 0 起不再是最简单的只读 Server：
  - query_stock(sku, warehouse)   查单个 SKU 余量（读）
  - list_low_stock()              列出低于安全线的 SKU（读）
  - receive_goods(order_id, conf) 确认收货入库（写：同时动订单账和库存账）

教学要点：
1. stdio 传输时 stdout 就是协议通道，绝不能 print() 调试（会污染协议报文）。
2. 查不到就诚实返回"未找到"——学 FoodRAG 防幻觉心法：宁可说没有，不编一个。
3. 无状态设计：每次工具调用都重读账本文件。代价是每次都有磁盘 IO，
   换来的是永远读到最新数据（就算别的进程改了账本也不怕）。
4. 【写权限的域观念】收货入库本来就是库存域的活（货进的是我的仓库），
   所以 receive_goods 住本 Server 而不是采购 Server——写权限从站 0 之前的
   "一个 Server 一支笔"（procurement 独占写）升级为"每域管自己的写"。
   代价：库存管家伸手改了采购账（orders.json）——跨账本写。
5. 【站 0 埋雷，第 1 站结案】receive_goods 要写两次文件（先订单账后库存账）：
   - 两写之间进程崩 = 半写状态（单收了货、库存却没涨）
   - 两个收货员并发收同一 SKU = 丢更新（后写的把先写的冲掉）
   两案在 scripts/lost_update_demo.py 立案，第 1 站换 SQLite 用事务治。
"""
import json
from datetime import date
from pathlib import Path

# 【学徒注】官方 mcp 2.x SDK 已把 FastMCP 改名 MCPServer；这里用独立的
# fastmcp 包（langchain[mcp] 自带的运行时），@mcp.tool() 风格不变
from fastmcp import FastMCP

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INVENTORY_FILE = DATA_DIR / "inventory.json"
ORDERS_FILE = DATA_DIR / "orders.json"  # 跨账本写：收货要动采购账（见模块注释第 4 条）

mcp = FastMCP("inventory")


def _load() -> list:
    """读库存账（每次现读，不缓存——见模块注释第 3 条）"""
    return json.loads(INVENTORY_FILE.read_text(encoding="utf-8"))


def _read(path: Path) -> list:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, data: list) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


@mcp.tool()
def query_stock(sku: str, warehouse: str = "") -> str:
    """查询某个 SKU 的库存余量。sku 为物料编号（如 A003）；
    warehouse 可选，填仓库名（如 五金仓）可进一步限定，留空查所有仓库。
    查不到返回"未找到"。
    """
    hits = [r for r in _load()
            if r["sku"].upper() == sku.upper()
            and (not warehouse or r["warehouse"] == warehouse)]
    if not hits:
        # 【学徒注】诚实返回，不编数据。把已存在的同类 SKU 编号提示给 LLM，
        # 它能自己纠错（比如用户把 A3 说成 A003 之前先说错了编号）
        all_skus = ", ".join(r["sku"] for r in _load())
        return f"未找到 SKU {sku}。现有 SKU：{all_skus}"
    return json.dumps(hits, ensure_ascii=False)


@mcp.tool()
def list_low_stock() -> str:
    """列出所有库存量低于安全线（safety_stock）的 SKU，
    即"快用完了、该补货"的物料清单。返回按缺口从大到小排序。
    """
    low = [r for r in _load() if r["quantity"] < r["safety_stock"]]
    if not low:
        return "当前没有低于安全线的物料。"
    low.sort(key=lambda r: r["quantity"] - r["safety_stock"])
    return json.dumps(low, ensure_ascii=False)


@mcp.tool()
def receive_goods(order_id: str, confirmed: bool = False) -> str:
    """确认采购订单到货并入库（写操作，会同时修改订单账和库存账）。
    参数：order_id 订单号（如 PO-2026-1001）、confirmed 是否已取得用户明确同意。
    【重要合同】confirmed=false 时本工具不会写账，必须先向用户复述要收货的
    订单（哪家供应商、什么物料、多少数量），取得用户明确同意后，再带
    confirmed=true 重新调用。只有"已下单"状态的订单能收货；收货后订单状态
    变为"已收货"，订单数量自动加进对应 SKU 的库存。
    """
    if not confirmed:
        # 【学徒注】确认闸门，与 create_order/cancel_order 同款合同：
        # 不写账，把"该怎么补手续"说清楚，LLM 拿到会转述给用户走确认流程
        return (f"收货未确认：请先向用户复述要收货的订单（{order_id}），"
                "取得用户明确同意后，再带 confirmed=true 重新调用。")

    # 读一份 → 改这份 → 存这份（写账纪律，学 procurement_server）
    orders = _read(ORDERS_FILE)
    order = next((o for o in orders
                  if o["order_id"].upper() == order_id.upper()), None)
    if order is None:
        return f"未找到订单 {order_id}，无法收货。"
    if order["status"] == "已收货":
        return f"订单 {order['order_id']} 已收过货，无需重复操作。"
    if order["status"] != "已下单":
        # 已取消等状态：诚实拒收，不偷偷改账
        return (f"订单 {order['order_id']} 当前状态为「{order['status']}」，"
                "只有已下单的订单才能收货。")

    inventory = _read(INVENTORY_FILE)
    item = next((r for r in inventory
                 if r["sku"].upper() == order["sku"].upper()), None)
    if item is None:
        return f"收货失败：订单物料 {order['sku']} 在库存账中不存在。"

    today = date.today().isoformat()
    # 【学徒注】站 0 故意保留的教学坑：两次文件写。写①和写②之间进程崩
    # = 半写状态（订单"已收货"但库存没涨，账对不上）。第 1 站用 SQLite
    # 单事务治它——要么全成、要么全滚。
    order["status"] = "已收货"
    order["received_at"] = today
    _write(ORDERS_FILE, orders)          # 写①：采购账
    item["quantity"] += order["quantity"]
    item["updated_at"] = today
    _write(INVENTORY_FILE, inventory)     # 写②：库存账

    return json.dumps({
        "order_id": order["order_id"],
        "status": order["status"],
        "received_at": order["received_at"],
        "sku": item["sku"],
        "received_quantity": order["quantity"],
        "stock_after": item["quantity"],
    }, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(show_banner=False, log_level="ERROR")  # 默认 stdio 传输——由 agent.py 作为子进程拉起；关横幅和INFO日志，stderr留干净
