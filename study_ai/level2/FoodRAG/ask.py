"""拼：路由选仓 → 检索证据 → 受控生成（本地 qwen3:4b，一行切换 DashScope）。

【设计笔记】

和 search.py 同款路由：问做法走菜品仓，找店走饭店仓，
路由理由和证据一起打印——答案权哪来的，全程可追溯（trace 思想）。

受控生成（学 drug_rag qa_chain，这条不能砍）：
    证据塞提示词，"只准依据证据回答，没有就说没有"。
    饭店问题和做菜问题共用一套规则，答案末尾标来源。

LLM 可切换：PROVIDER 改成 "dashscope" 即换供应商。
qwen3:4b 是思考模型，可能吐 <think> 段——剥掉再给用户看。
"""

from __future__ import annotations

import argparse
import datetime
import os
import re
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from router import DB_CONFIG, normalize, resolve_db
from search import build_filter, get_store

# ==== 切换开关：ollama / dashscope ====
PROVIDER = "ollama"

SYSTEM_PROMPT = """你是金燕美食助手，负责回答做菜问题和饭店推荐。规则：
1. 只能依据下面提供的证据回答，证据里没有的信息就明说"资料里没有"，不要编造。
2. 做菜问题给出关键步骤和注意点；饭店问题给出店名、人均、地址等关键信息。
3. 涉及多家店/多道菜时逐家列出，不要混在一起。
（来源行不用你写，程序会根据实际检索到的证据自动盖章。）"""


def build_llm():
    """增强层工厂：一处开关，两家供应商。"""
    if PROVIDER == "ollama":
        return ChatOllama(model="qwen3:4b", temperature=0.1)
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model="qwen3.7-flash-2026-07-15",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        temperature=0.1,
    )


def _clean(text: str) -> str:
    """剥掉 qwen3 思考模型的 <think> 段。"""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ==== 槽位补全（学 AdvRAG_v2 例5）：缺的槽不瞎猜，反问补齐 ====

# 周五六日落这组就先问再定（周五可能犒劳自己也可能出去，用户定的规矩）
ASK_WEEKDAYS = {4, 5, 6}  # weekday(): 周一=0 … 周日=6
WEEKDAY_NAMES = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


def _target_weekday(question: str) -> tuple[int, str]:
    """问题里说没说日子：周末/周X/明天 优先，没提就当"今天"。"""
    q = normalize(question)
    today = datetime.date.today()
    if "周末" in q:
        return 5, "周末"  # 当周六处理（周六周日反正都要问）
    for i, name in enumerate(WEEKDAY_NAMES):
        if name in q:
            return i, name
    if "明天" in q:
        d = today + datetime.timedelta(days=1)
        return d.weekday(), "明天"
    return today.weekday(), "今天"


def _ask(prompt: str) -> str:
    """反问一轮；管道/非交互环境 EOF 不炸，返回空串走默认。"""
    try:
        return input(f"{prompt} ").strip()
    except EOFError:
        return ""


def clarify(question: str) -> tuple[str, str, str | list | None, int | None]:
    """槽位补全：只在判单树落到兜底（啥信号都没有）时由 ask() 调用。

    周一~四  → 默认在家吃（menu），不问（用户作息：工作日晚饭在家）
    周五六日 → 问①在家/出去；在家再问②两人/请朋友；请朋友再问③几位
        ≤4人：纯硬菜局（难度=费工夫，k=5）
        >4人：硬菜不够凑一桌 → 放宽"费工夫+一般"混搭，k=人数+2（上限8）
    问题里明确说了"在家吃/出去吃"的，判单树场景词分支早接走了，到不了这。
    返回 (db, reason, 难度片[str或list], 建议k)。
    """
    wd, day_desc = _target_weekday(question)
    if wd not in ASK_WEEKDAYS:
        return "menu", f"{day_desc}·默认在家吃（周一~周四）", None, None

    ans = _ask(f"（{day_desc}）在家吃还是出去吃？")
    if any(w in ans for w in ["出去", "外食", "下馆子", "饭店", "店"]):
        return "restaurant", f"{day_desc}·槽位补全：出去吃", None, None

    ans2 = _ask("在家吃呀。就夫妻两人，还是请朋友来？")
    if not any(w in ans2 for w in ["朋友", "请客", "聚餐", "客"]):
        return "menu", f"{day_desc}·槽位补全：在家两人餐", None, None

    ans3 = _ask("好嘞，一共几位一起吃？")
    m = re.search(r"\d+", ans3)
    if m:
        guests = int(m.group(0))
    else:  # 中文数字兜底（"六个人""两位"）
        cn = {"两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
        guests = next((v for ch, v in cn.items() if ch in ans3), 4)  # 没数字按小局4人
    if guests > 4:
        return ("menu", f"{day_desc}·槽位补全：请朋友{guests}人（硬菜+一般混搭）",
                ["费工夫", "一般"], min(guests + 2, 8))
    return "menu", f"{day_desc}·槽位补全：请朋友{guests}人（硬菜局）", "费工夫", 5


def ask(question: str, k: int = 3, max_price: int | None = None,
        difficulty: str | list[str] | None = None) -> None:
    # ---- 路由 + 查：复用 search.py 的判单树 ----
    db, reason = resolve_db(question)

    # ---- 槽位补全：判单树落到兜底（无任何信号）才启动，按日子补默认或反问 ----
    if reason.startswith("兜底"):
        db, reason, slot_difficulty, slot_k = clarify(question)
        if slot_difficulty and difficulty is None:
            difficulty = slot_difficulty  # 反问挖出的难度档（CLI --difficulty 仍最高优先）
        if slot_k:
            k = max(k, slot_k)  # 请朋友凑一桌，多取几块证据

    dish_filter = build_filter(question, db, type_filter=None, max_price=max_price,
                               difficulty=difficulty)
    t0 = time.perf_counter()
    docs = get_store(db).similarity_search_with_relevance_scores(
        question, k=k, filter=dish_filter
    )
    retrieve_sec = time.perf_counter() - t0
    if not docs:
        print("没检索到任何证据，无法回答。")
        return

    label_key = DB_CONFIG[db]["label_key"]
    evidence = "\n\n".join(
        f"【证据{i}】{doc.page_content}" for i, (doc, _) in enumerate(docs, start=1)
    )

    # ---- trace：先看路由和证据，再看答案 ----
    print(f"仓库: {db}（{reason}）")
    print(f"证据: {[doc.metadata[label_key] for doc, _ in docs]}")
    print(f"检索耗时: {retrieve_sec:.3f}秒")
    print("=" * 50)

    # ---- 拼：受控生成（LLM 是增强层：挂了不拖垮主链路，退回抽取式兜底）----
    t0 = time.perf_counter()
    try:
        llm = build_llm()  # 构建也可能炸（如 dashscope 模式没配 key），一并圈进保险杠
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"问题：{question}\n\n证据：\n{evidence}"),
        ]
        answer = _clean(llm.invoke(messages).content)
        print(f"生成耗时: {time.perf_counter() - t0:.3f}秒")
        print(answer)
    except Exception as exc:
        # 学 drug_rag _controlled_llm_answer 的 except 分支：Ollama 没开/key 失效/模型404
        # 都走这里——检索已经成功，证据直接上桌，用户至少看得到菜谱
        print(f"生成层提示：LLM 不可用（{type(exc).__name__}），已退回抽取式兜底答案。")
        for i, (doc, _) in enumerate(docs, start=1):
            print(f"\n【证据{i}】{doc.metadata[label_key]}")
            print(doc.page_content)

    # ---- 来源盖章：检索层已知的信息不让 LLM 抄——确定性的事交给代码 ----
    # 名字从 metadata 现取（菜名/展示名），保序去重防同名连锁重复
    labels = list(dict.fromkeys(doc.metadata[label_key] for doc, _ in docs))
    print(f"\n来源：{'、'.join(labels)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="金燕美食问答（不带问题进入对话模式）")
    parser.add_argument("question", nargs="?", default=None, help="想问什么（不填进入对话模式）")
    parser.add_argument("--k", type=int, default=3, help="取几块证据（默认3）")
    parser.add_argument("--max-price", type=int, default=None,
                        help="人均上限（如 --max-price 50 → 人均≤50，仅饭店仓生效）")
    parser.add_argument("--difficulty", choices=["简单", "一般", "费工夫"], default=None,
                        help="按难度过滤（仅菜品仓生效）")
    args = parser.parse_args()

    if args.question:  # 带问题 = 一次性问答（脚本/调试用，行为不变）
        ask(args.question, k=args.k, max_price=args.max_price, difficulty=args.difficulty)
    else:  # 不带问题 = 对话模式：一直聊到回车退出
        print("金燕美食助手已上线，直接开问（直接回车退出）")
        while True:
            try:
                q = input("\n问点什么> ").strip()
            except EOFError:
                break
            if not q:
                print("慢用，回见。")
                break
            ask(q, k=args.k, max_price=args.max_price, difficulty=args.difficulty)
