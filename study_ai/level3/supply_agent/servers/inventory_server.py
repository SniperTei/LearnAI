"""库存 MCP Server（查询 + 收货入库/销售出库写账，stdio 传输）

【学徒注】中间商的仓库，两只手：
  - query_stock(sku, warehouse)    查单个 SKU 余量（读）
  - list_low_stock()               列出低于安全线的 SKU（读）
  - receive_goods(order_id, conf)  确认收货入库（写：动采购账+库存账）
  - outbound_goods(order_id, conf) 确认销售出库（写：动销售账+库存账）

教学要点：
1. stdio 传输时 stdout 就是协议通道，绝不能 print() 调试（会污染协议报文）。
2. 查不到就诚实返回"未找到"——学 FoodRAG 防幻觉心法：宁可说没有，不编一个。
3. 无状态设计：每次工具调用都重读账本文件。代价是每次都有磁盘 IO，
   换来的是永远读到最新数据（就算别的进程改了账本也不怕）。
4. 【写权限的域观念】货进货出都是库存域的活（进出都是我的仓库）：
   收货入库 receive_goods / 销售出库 outbound_goods 都住本 Server。
   代价是跨账本写——库存管家两只手分别伸进采购账（orders.json）和
   销售账（sales_orders.json）。
5. 【站 0 埋雷，两案不治只立案】receive_goods/outbound_goods 都要写两次
   文件（订单账 + 库存账），两写之间崩 = 半写；并发 = 丢更新。
   演示见 scripts/lost_update_demo.py；真药方（SQLite 事务）记录在
   supply_plan.md，学习版不施工。
6. 【学习版最简取舍】出库直接扣总数，不做批次/先进先出——批次带进价、
   按批出库是真实系统的事（supply_plan.md 站 2/3）。
"""
import json
from datetime import date
from pathlib import Path

# 【学徒注】官方 mcp 2.x SDK 已把 FastMCP 改名 MCPServer；这里用独立的
# fastmcp 包（langchain[mcp] 自带的运行时），@mcp.tool() 风格不变
from fastmcp import FastMCP

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INVENTORY_FILE = DATA_DIR / "inventory.json"
ORDERS_FILE = DATA_DIR / "orders.json"        # 跨账本写：收货要动采购账（见模块注释第 4 条）
SALES_ORDERS_FILE = DATA_DIR / "sales_orders.json"  # 跨账本写：出库要动销售账

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


@mcp.tool()
def outbound_goods(order_id: str, confirmed: bool = False) -> str:
    """确认销售订单出库发货（写操作，会同时修改销售账和库存账）。
    参数：order_id 销售单号（如 SO-2026-2001）、confirmed 是否已取得用户明确同意。
    【重要合同】confirmed=false 时本工具不会写账，必须先向用户复述要出库的
    订单（哪家客户、什么物料、多少数量），取得用户明确同意后，再带
    confirmed=true 重新调用。只有"已下单"状态的销售单能出库；出库后销售单
    状态变为"已发货"，库存自动扣减对应数量。库存不足时会诚实拒绝，不会
    把库存扣成负数。
    """
    if not confirmed:
        return (f"出库未确认：请先向用户复述要出库的销售单（{order_id}），"
                "取得用户明确同意后，再带 confirmed=true 重新调用。")

    # 读一份 → 改这份 → 存这份（写账纪律，与 receive_goods 同款）
    orders = _read(SALES_ORDERS_FILE)
    order = next((o for o in orders
                  if o["order_id"].upper() == order_id.upper()), None)
    if order is None:
        return f"未找到销售单 {order_id}，无法出库。"
    if order["status"] == "已发货":
        return f"销售单 {order['order_id']} 已出过库，无需重复操作。"
    if order["status"] != "已下单":
        return (f"销售单 {order['order_id']} 当前状态为「{order['status']}」，"
                "只有已下单的销售单才能出库。")

    inventory = _read(INVENTORY_FILE)
    item = next((r for r in inventory
                 if r["sku"].upper() == order["sku"].upper()), None)
    if item is None:
        return f"出库失败：销售单物料 {order['sku']} 在库存账中不存在。"
    # 【学徒注】出库前先验库存——下单不查库存（先接单后备货），
    # 出库这一刻才见真章：不够就诚实拒，绝不扣成负数
    if item["quantity"] < order["quantity"]:
        return (f"出库失败：{item['sku']}（{item['name']}）库存仅剩 "
                f"{item['quantity']} {item['unit']}，销售单需要 "
                f"{order['quantity']} {item['unit']}，库存不足。")

    today = date.today().isoformat()
    # 【学徒注】与 receive_goods 同款两次文件写（教学坑不治只立案，见模块注释第 5 条）
    order["status"] = "已发货"
    order["shipped_at"] = today
    _write(SALES_ORDERS_FILE, orders)   # 写①：销售账
    item["quantity"] -= order["quantity"]
    item["updated_at"] = today
    _write(INVENTORY_FILE, inventory)   # 写②：库存账

    return json.dumps({
        "order_id": order["order_id"],
        "status": order["status"],
        "shipped_at": order["shipped_at"],
        "sku": item["sku"],
        "out_quantity": order["quantity"],
        "stock_after": item["quantity"],
    }, ensure_ascii=False)


if __name__ == "__main__":
    import sys
    if "--http" in sys.argv:
        # 【学徒注】L2：HTTP 常驻模式（先迁本座——写 Server、跨账本写最重）。
        # 端口 8101，与 agent.py 的 SERVER_PORTS 对齐
        from http_entry import run_http
        run_http(mcp, default_port=8101)
    else:
        mcp.run(show_banner=False, log_level="ERROR")  # stdio 默认：由 agent.py 作为子进程拉起；关横幅和INFO日志，stderr留干净
