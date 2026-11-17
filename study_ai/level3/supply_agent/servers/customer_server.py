"""客户 MCP Server（只读，stdio 传输）——卖手侧的情报员

【学徒注】与 supplier_server 完全对称地长出来：买手侧情报员管供应商，
这里管客户。新增一个域不是改老代码，是照着老样子长新 Server——
"一域一 Server"架构扩展性的活证明（需求 9：进货出货可拆分）。

  - search_customers(keyword)   按关键词搜客户（名称/品类/别名都认）
  - get_customer(customer_id)   查单家客户详情

教学要点：
1. 别名归一（同 supplier）：搜"福田贸易"也能命中"福田贸易商行"。
2. 排序的取舍：供应商按交期排（"谁交期最短"是常问问题，排序是答案的一部分）；
   客户暂无天然排序需求，按编号稳定输出——同样的账本永远同样的顺序，
   确定性本身就是给 LLM 的礼物。
"""
import json
from pathlib import Path

from fastmcp import FastMCP

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "customers.json"

mcp = FastMCP("customer")


def _load() -> list:
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))


@mcp.tool()
def search_customers(keyword: str) -> str:
    """按关键词搜索客户。关键词可以命中：客户名称、常采品类（如 轴承/纸箱）、
    或别名（如 南山电器、福田贸易）。返回匹配的客户列表，按编号排序。
    没有匹配时返回"未找到"。
    """
    kw = keyword.strip()
    # 【学徒注】与 supplier 的一处不同：供应商 categories 存品类词（"轴承"），
    # 整元素相等就能命中；客户 categories 存的是常采货名（"深沟球轴承"），
    # 必须逐项子串匹配——搜"轴承"才能命中"深沟球轴承"。eval 测18逮住的真 bug
    hits = [c for c in _load()
            if kw in c["name"] or any(kw in cat for cat in c["categories"])
            or kw in c["aliases"]]
    if not hits:
        # 【学徒注】诚实返回 + 品类菜单引导换词（同 supplier 套路）
        all_cats = sorted({c for x in _load() for c in x["categories"]})
        return f"未找到匹配「{kw}」的客户。现有客户常采品类：{', '.join(all_cats)}"
    hits.sort(key=lambda c: c["id"])  # 编号稳定排序，见模块注释第 2 条
    return json.dumps(hits, ensure_ascii=False)


@mcp.tool()
def get_customer(customer_id: str) -> str:
    """按客户编号（如 CUS-001）查询单家客户的完整信息：
    名称、常采品类、联系方式。
    """
    for c in _load():
        if c["id"].upper() == customer_id.upper():
            return json.dumps(c, ensure_ascii=False)
    return f"未找到客户 {customer_id}。"


if __name__ == "__main__":
    import sys
    if "--http" in sys.argv:
        from http_entry import run_http
        run_http(mcp, default_port=8105)  # 端口与 agent.py 的 SERVER_PORTS 对齐
    else:
        mcp.run(show_banner=False, log_level="ERROR")  # stdio 默认：agent 子进程；stderr留干净
