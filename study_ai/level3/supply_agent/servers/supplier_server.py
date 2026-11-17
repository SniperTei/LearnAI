"""供应商 MCP Server（只读，stdio 传输）

工具：
  - search_suppliers(keyword)   按关键词搜供应商（名称/品类/别名都认）
  - get_supplier(supplier_id)   查单家供应商详情

教学要点：
1. 别名归一（学 FoodRAG 锁菜别名）：搜"华精"也能命中"华精轴承制造"，
   命中后返回完整正式信息——用户的口语叫法和账本的正式名对得上。
2. 搜索结果按交期从短到长排好序再返回——"交期最短的是哪家"这种问题
   LLM 拿到数据第一行就是答案，减少它自己排序出错的机会。
   （确定性信息能用代码算好就不用 LLM 算——学 FoodRAG 来源盖章）
"""
import json
from pathlib import Path

from fastmcp import FastMCP

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "suppliers.json"

mcp = FastMCP("supplier")


def _load() -> list:
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))


@mcp.tool()
def search_suppliers(keyword: str) -> str:
    """按关键词搜索供应商。关键词可以命中：供应商名称、供应品类（如 轴承/纸箱）、
    或别名（如 华精、恒力）。返回匹配的供应商列表，已按交期（lead_time_days）
    从短到长排序。没有匹配时返回"未找到"。
    """
    kw = keyword.strip()
    hits = [s for s in _load()
            if kw in s["name"] or kw in s["categories"] or kw in s["aliases"]]
    if not hits:
        # 【学徒注】诚实返回 + 告诉 LLM 现有品类菜单，引导它换个词再搜
        all_cats = sorted({c for s in _load() for c in s["categories"]})
        return f"未找到匹配「{kw}」的供应商。现有供应品类：{', '.join(all_cats)}"
    hits.sort(key=lambda s: s["lead_time_days"])  # 交期短→长，见模块注释第 2 条
    return json.dumps(hits, ensure_ascii=False)


@mcp.tool()
def get_supplier(supplier_id: str) -> str:
    """按供应商编号（如 SUP-001）查询单家供应商的完整信息：
    名称、品类、交期、评级、价格档、联系方式。
    """
    for s in _load():
        if s["id"].upper() == supplier_id.upper():
            return json.dumps(s, ensure_ascii=False)
    return f"未找到供应商 {supplier_id}。"


if __name__ == "__main__":
    import sys
    if "--http" in sys.argv:
        from http_entry import run_http
        run_http(mcp, default_port=8102)  # 端口与 agent.py 的 SERVER_PORTS 对齐
    else:
        mcp.run(show_banner=False, log_level="ERROR")  # stdio 默认：agent 子进程；stderr留干净
