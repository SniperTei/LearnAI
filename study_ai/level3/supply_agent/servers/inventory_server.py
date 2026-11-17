"""库存 MCP Server（只读，stdio 传输）

【学徒注】本项目 4 个 Server 里最简单的一个，只提供查询：
  - query_stock(sku, warehouse)  查单个 SKU 余量
  - list_low_stock()             列出低于安全线的 SKU

教学要点：
1. stdio 传输时 stdout 就是协议通道，绝不能 print() 调试（会污染协议报文）。
2. 查不到就诚实返回"未找到"——学 FoodRAG 防幻觉心法：宁可说没有，不编一个。
3. 无状态设计：每次工具调用都重读账本文件。代价是每次都有磁盘 IO，
   换来的是永远读到最新数据（就算别的进程改了账本也不怕）。
"""
import json
from pathlib import Path

# 【学徒注】官方 mcp 2.x SDK 已把 FastMCP 改名 MCPServer；这里用独立的
# fastmcp 包（langchain[mcp] 自带的运行时），@mcp.tool() 风格不变
from fastmcp import FastMCP

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "inventory.json"

mcp = FastMCP("inventory")


def _load() -> list:
    """读账本（每次现读，不缓存——见模块注释第 3 条）"""
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))


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


if __name__ == "__main__":
    mcp.run(show_banner=False, log_level="ERROR")  # 默认 stdio 传输——由 agent.py 作为子进程拉起；关横幅和INFO日志，stderr留干净
