"""拼：路由选仓 → 检索证据 → 受控生成（本地 qwen3:4b，一行切换 DashScope）。

【设计笔记】

和 search.py 同款路由：问做法走菜品仓，找店走饭店仓，
路由理由和证据一起打印——答案权哪来的，全程可追溯（trace 思想）。

受控生成（学 drug_rag qa_chain，这条不能砍）：
    证据塞提示词，"只准依据证据回答，没有就说没有"。
    饭店问题和做菜问题共用一套规则，答案末尾标来源。

LLM 可切换：PROVIDER 改成 "dashscope" 即换供应商。
qwen3:4b 是思考模型，可能吐 <think> 段——剥掉再给用户看。

对话历史（学 langchain 第7关记忆三件套的手动版）：
    对话模式记最近 2 轮（问题, 答案），拼 prompt 时插在系统提示词和新问题中间
    （Human/AI 成对 = 手动版 MessagesPlaceholder）。只喂生成层，检索词仍是当前问题——
    省略式追问（"它怎么做"）靠历史能听懂，但检索层锁定还要靠历史带锁（见下）。

web 化改造（学 drug_rag 的分层，但砍掉 session）：
    ask_core() = 无 I/O 的问答核：不 print 不 input，CLI 和 web.py 共用。
    反问槽位的答案由调用方攒在 answers 字典里回传——服务端零状态，
    刷新页面/重启服务都不用管上下文（上下文全在前端手里）。
    CLI 的 ask() 降级成包装：收到"缺槽位"就用终端 input() 补，循环到出答案。
"""

from __future__ import annotations

import argparse
import datetime
import os
import re
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from router import (DB_CONFIG, extract_difficulty, load_rows, match_areas, match_db_names,
                    normalize, resolve_db)
from search import build_filter, get_store

# ==== 切换开关：ollama / dashscope ====
PROVIDER = "ollama"

SYSTEM_PROMPT = """你是金燕美食助手，负责回答做菜问题和饭店推荐。规则：
1. 只能依据下面提供的证据回答，证据里没有的信息就明说"资料里没有"，不要编造。
2. 做菜问题给出关键步骤和注意点；饭店问题给出店名、人均、地址等关键信息。
3. 涉及多家店/多道菜时逐家列出，不要混在一起。
4. 证据里的菜是按荤素、难度搭配好捞回来的：两人餐挑2~3道呈现（荤素都要有），
   朋友聚餐多列几道、硬菜放前面；工作日的局不要推荐费工夫的菜。
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

# 一天里的哪顿（实战补：问"周二中午吃什么"被当成在家做饭——用户作息工作日中午回不了家）
LUNCH_WORDS = ["中午", "午餐", "午饭", "工作餐"]
DINNER_WORDS = ["晚上", "晚餐", "晚饭", "夜里", "夜宵"]

# 槽位识别词表（认答案用，"出去吃呀"→出去；ask_core 和 CLI 包装共用）
OUT_WORDS = ["出去", "外食", "下馆子", "饭店", "店"]
FRIEND_WORDS = ["朋友", "请客", "聚餐", "客"]

# 问句预填词表（实战补：用户明说"邀请姐姐姐夫来家里吃饭"还被连问三道——
# 问句里已说的槽不该再问，只反问真正缺的，这才是"槽位补全"的本意）。
# 词表命门同款：没写的说法不触发（"娘家人来"就没写），漏网再补。
# ⚠️ 出去先查再查在家：两词同现时（"不在家吃，出去"）出去必须赢——
#    和判单树"外卖系排家饭系前面"是同一个坑，子串包含闹的。
PREFILL_HOME_WORDS = ["在家", "家里", "自己做", "家宴"]
PREFILL_GUEST_WORDS = ["邀请", "请客", "朋友", "亲戚", "姐姐", "哥", "同事", "爸妈", "父母", "公婆", "客人", "做客"]


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


def _target_meal(question: str) -> str:
    """问题里说没说哪顿饭：中午/晚上 优先，没提默认当晚上（用户问"吃啥"多半指晚饭）。"""
    q = normalize(question)
    if any(w in q for w in LUNCH_WORDS):
        return "中午"
    if any(w in q for w in DINNER_WORDS):
        return "晚上"
    return "晚上"


def _parse_count(raw: str) -> int:
    """从答案里挖人数：阿拉伯数字优先，中文数字兜底，没数字按小局 4 人。"""
    m = re.search(r"\d+", raw)
    if m:
        return int(m.group(0))
    cn = {"两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    return next((v for ch, v in cn.items() if ch in raw), 4)  # 没数字按小局4人


def _clarify_slots(question: str, answers: dict) -> dict:
    """槽位状态机（无 I/O 版 clarify）：查 answers 里攒了几个槽，缺第一个就报哪个。

    工作日中午（周一~五）→ 饭店仓·工作餐（用户作息：中午在外回不了家）
    周一~四晚上 → 默认在家吃（menu）+ 只要简单菜（工作日累，不炖排骨），不问
    周五六日 → 问①在家/出去；在家再问②两人/请朋友；请朋友再问③几位
        ≤4人：纯硬菜局（难度=费工夫，k=5）
        >4人：硬菜不够凑一桌 → 放宽"费工夫+一般"混搭，k=人数+2（上限8）
    问题里明确说了"在家吃/出去吃"的，判单树场景词分支早接走了，到不了这。
    难度优先级：CLI --difficulty > 问题原话（"来个硬菜"）> 默认档——问句明说要硬菜
    时不该被"工作日=简单"压死（ask_core 里把关）。
    返回 {"ask": (槽名, 提示语)} 或 {"db":…, "reason":…, "difficulty":…, "k":…}。
    """
    wd, day_desc = _target_weekday(question)
    meal = _target_meal(question)
    # 工作日中午：在外工作餐，直接饭店仓（周末中午照旧走下面的反问流程）
    if meal == "中午" and wd not in {5, 6}:
        return {"db": "restaurant", "reason": f"{day_desc}·工作日中午默认工作餐",
                "difficulty": None, "k": None}
    if wd not in ASK_WEEKDAYS:
        # 周一~四默认两人餐（11-03 用户规矩）：没说几人=2人；简单至少2、一般1~2、
        # 费工夫不要（工作日累不炖排骨，问句点名要硬菜也不给）；荤素都要有——
        # 三路配额结构性全保证。
        quotas = [("快手荤菜", {"$and": [{"类型": "荤菜"}, {"难度": "简单"}]}, 1),
                  ("快手素菜", {"$and": [{"类型": "素菜"}, {"难度": "简单"}]}, 1),
                  ("一般菜", {"难度": "一般"}, 2)]
        reason = f"{day_desc}·默认在家吃·两人餐（快手荤1+快手素1+一般2，不要费工夫）"
        if extract_difficulty(question) == "费工夫":
            reason += "·工作日不给费工夫"  # trace 如实记：问句要硬菜被规矩挡了
        return {"db": "menu", "reason": reason,
                "quotas": quotas, "difficulty": None, "k": 4}

    # 槽位预填：问句原话里先挖 dine/party，挖到的槽不再问（answers 里的显式答案仍优先）
    q = normalize(question)
    prefill_dine = None
    if any(w in q for w in OUT_WORDS):
        prefill_dine = "出去"
    elif any(w in q for w in PREFILL_HOME_WORDS):
        prefill_dine = "在家"
    prefill_party = "客人" if any(w in q for w in PREFILL_GUEST_WORDS) else None
    dine = answers.get("dine") or prefill_dine
    party = answers.get("party") or prefill_party

    if not dine:
        return {"ask": ("dine", f"（{day_desc}）在家吃还是出去吃？")}
    if any(w in dine for w in OUT_WORDS):
        src = "问句已说" if dine == prefill_dine else "槽位补全"
        return {"db": "restaurant", "reason": f"{day_desc}·{src}：出去吃",
                "difficulty": None, "k": None}

    if not party:
        return {"ask": ("party", "在家吃呀。就夫妻两人，还是请朋友来？")}
    if not any(w in party for w in FRIEND_WORDS):
        # 周末两人餐：周末有时间，难度不限，但荤素搭配照旧（11-03 普适规矩）
        quotas = [("荤菜", {"类型": "荤菜"}, 2), ("素菜", {"类型": "素菜"}, 1)]
        return {"db": "menu",
                "reason": f"{day_desc}·槽位补全：在家两人餐（荤2素1搭配）",
                "quotas": quotas, "difficulty": None, "k": 3}

    raw_count = answers.get("count", "")
    if not raw_count:
        return {"ask": ("count", "好嘞，一共几位一起吃？")}
    guests = _parse_count(raw_count)
    if guests > 4:
        # 三路配额（迷你版多路召回，学 drug_rag stages）：一桌好菜不靠一路捞——
        # 硬菜撑场面（费工夫5道全是荤系，天生压桌）/ 快手素菜解腻（简单素菜16道库存足）
        # / 一般菜垫数量。结构性保证 荤+素×三档难度全齐，不赌向量检索的运气。
        # 三路按难度互斥（费工夫/简单/一般），天然不撞车；总名额 k=人数+2 上限8。
        k = min(guests + 2, 8)
        n_hard = min((guests + 1) // 2, 4)  # 硬菜=人数一半向上取整
        n_fill = k - n_hard - 2
        quotas = [("硬菜", {"难度": "费工夫"}, n_hard),
                  ("快手素菜", {"$and": [{"类型": "素菜"}, {"难度": "简单"}]}, 2),
                  ("一般填充", {"难度": "一般"}, n_fill)]
        return {"db": "menu",
                "reason": f"{day_desc}·槽位补全：请朋友{guests}人（硬菜{n_hard}+快手素2+一般{n_fill}三路配额）",
                "quotas": quotas, "difficulty": None, "k": k}
    # ≤4人硬菜局：小局硬菜撑住，但"只要在家做都要荤素搭配"（11-03 普适规矩）——补2道快手素解腻
    quotas = [("硬菜", {"难度": "费工夫"}, 5), ("快手素菜", {"$and": [{"类型": "素菜"}, {"难度": "简单"}]}, 2)]
    return {"db": "menu", "reason": f"{day_desc}·槽位补全：请朋友{guests}人（硬菜局5+快手素2）",
            "quotas": quotas, "difficulty": None, "k": 7}


# 追问标志词：问题里出现这些词才算"接着上文问"，才允许历史带锁。
# 防的是换话题误伤："推荐个简单的菜"没这些词→新话题，不带锁；
# 已知误伤："其他"含"它"（"有其他的吗"会被当成追问）——词表命门，漏网再补。
FOLLOWUP_MARKERS = ["它", "这道", "这个", "这家", "那道", "那个", "那家", "刚才", "前面", "上面", "第二"]


def ask_core(question: str, k: int = 3, max_price: int | None = None,
             difficulty: str | list[str] | None = None,
             history: list[dict] | None = None,
             answers: dict | None = None) -> dict:
    """问答核（无 I/O）：路由→检索→受控生成，全程不 print 不 input。

    answers = 反问槽位的累积答案（{"dine":…, "party":…, "count":…, "area":…}），
    由调用方持有并回传——web 前端每轮带上，CLI 循环里攒，服务端零状态。
    返回三种结果（CLI 和 web 都认这套合同）：
        {"type": "input",  "slot": 槽名, "prompt": 提示语}      还缺槽位，拿 prompt 去问用户
        {"type": "miss",   "reply": 文本}                        没搜到/没检索到，本轮不进历史
        {"type": "answer", "reply": 文本, "lock": 名字|None, "trace": {…}}   正式答案
    trace = 监控录像的数据版（CLI 拿去打印，web 拿去渲染折叠面板）。
    """
    answers = answers or {}

    # ---- 路由 + 查：复用 search.py 的判单树 ----
    db, reason = resolve_db(question)

    # ---- 历史带锁（迷你版 query_rewriter）：追问没信号 → 借上一轮锁定的名字拼检索词 ----
    # 学 drug_rag 分工：改写词（rewritten_query）归检索，原问题（question）归生成。
    # 只有"兜底+有历史+上轮锁过名+问题像追问"四条全中才带——换话题不被旧名字绑架。
    q_retrieval = question
    if (reason.startswith("兜底") and history and history[-1].get("lock")
            and any(m in question for m in FOLLOWUP_MARKERS)):
        q_retrieval = f"{history[-1]['lock']} {question}"
        db, reason = resolve_db(q_retrieval)
        reason += "·历史带锁"

    # ---- 外卖地址槽：没带位置就反问，认不出商圈直接如实说"没搜索到" ----
    # （宁可没有，不推错的——账本里没收录的位置，推别片的店就是事故）
    area: list[str] | None = None
    if reason.startswith("场景词: 外卖系"):
        area = match_areas(question)
        if not area:
            if "area" not in answers:
                return {"type": "input", "slot": "area",
                        "prompt": "点外卖呀。你在哪个商圈/位置附近？（如：坂田、龙华、罗湖万象城）"}
            area = match_areas(answers["area"])
            if not area:
                all_areas = sorted({r["metadata"]["商圈"] for r in load_rows("restaurant")
                                    if r["metadata"].get("商圈")})
                return {"type": "miss",
                        "reply": f"没搜索到「{answers['area']}」附近收录的店。"
                                 f"目前账本里的商圈：{'、'.join(all_areas)}"}
        reason += f"·址槽:{area}"
    elif reason.startswith("命中商圈"):
        # 问句点名的商圈直接当过滤片（eval#10 逮住的缝：路由认识商圈、过滤不认——
        # 两张皮，检索全仓裸搜把别片的店也捞回来。缝上：路由命中啥，过滤卡就带啥）
        area = match_areas(question)
        reason += f"·商圈片:{area}"

    # ---- 槽位补全：判单树落到兜底（无任何信号）才启动，按日子补默认或反问 ----
    slot_quotas: list | None = None  # >4人局的三路配额（有它检索走多路，见下）
    if reason.startswith("兜底"):
        slot_result = _clarify_slots(question, answers)
        if "ask" in slot_result:
            slot, prompt = slot_result["ask"]
            return {"type": "input", "slot": slot, "prompt": prompt}
        db, reason = slot_result["db"], slot_result["reason"]
        slot_quotas = slot_result.get("quotas")
        # 难度三级优先：CLI --difficulty > 问题原话（"来个硬菜"）> 默认档（"工作日=简单"）
        if slot_result["difficulty"] and difficulty is None and extract_difficulty(question) is None:
            difficulty = slot_result["difficulty"]
        elif slot_result["difficulty"] and difficulty is None:
            # 问句原话压过了默认档——trace 得说实话，不然"推荐简单的"配一锅费工夫没法看
            reason += f"·问句要{extract_difficulty(question)}档，不压"
        if slot_result["k"]:
            k = max(k, slot_result["k"])  # 请朋友凑一桌，多取几块证据

    # 本轮锁定的名字记进历史，供下一轮追问带锁（"锅包肉怎么做"→lock=锅包肉）
    lock = " ".join(match_db_names(q_retrieval, db)) or None
    label_key = DB_CONFIG[db]["label_key"]
    # 配额模式下问句难度让位：三路自带难度片且互斥，再叠一片必 0 命中
    # （no_difficulty 连问句提取也关——"想吃硬菜"挖出费工夫片会和配额片打架）
    dish_filter = build_filter(q_retrieval, db, type_filter=None, max_price=max_price,
                               difficulty=None if slot_quotas else difficulty, area=area,
                               no_difficulty=bool(slot_quotas))
    t0 = time.perf_counter()
    if slot_quotas:
        # 三路配额检索：每路 = 公共卡(名字片等) + 配额片 合并出击，倒进同一筐
        docs, seen = [], set()
        for route_label, piece, n in slot_quotas:
            part = get_store(db).similarity_search_with_relevance_scores(
                q_retrieval, k=n, filter=_merge_filter(dish_filter, piece)
            )
            for doc, score in part:
                name = doc.metadata[label_key]
                if name not in seen:  # 同菜不占两个名额（难度互斥本不撞，去重是保险）
                    seen.add(name)
                    docs.append((doc, score))
    else:
        docs = get_store(db).similarity_search_with_relevance_scores(
            q_retrieval, k=k, filter=dish_filter
        )
    retrieve_sec = time.perf_counter() - t0
    if not docs:
        return {"type": "miss", "reply": "没检索到任何证据，无法回答。"}

    trace = {
        "db": db,
        "reason": reason,
        "evidence": [doc.metadata[label_key] for doc, _ in docs],  # 菜名/展示名列表
        "retrieve_sec": retrieve_sec,
    }

    # ---- 拼：受控生成（LLM 是增强层：挂了不拖垮主链路，退回抽取式兜底）----
    t0 = time.perf_counter()
    try:
        llm = build_llm()  # 构建也可能炸（如 dashscope 模式没配 key），一并圈进保险杠
        # 历史插在系统提示词和新问题中间：Human/AI 成对 = 手动版 MessagesPlaceholder。
        # 只带最近 2 轮——qwen3:4b 本地又慢又小，历史塞多了纯属拖累。
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        for turn in (history or [])[-2:]:
            messages.append(HumanMessage(content=turn["question"]))
            messages.append(AIMessage(content=turn["answer"]))
        messages.append(HumanMessage(content=f"问题：{question}\n\n证据：\n{evidence_of(docs, label_key)}"))
        answer = _clean(llm.invoke(messages).content)
        trace["gen_sec"] = time.perf_counter() - t0
        trace["fallback"] = False
    except Exception as exc:
        # 学 drug_rag _controlled_llm_answer 的 except 分支：Ollama 没开/key 失效/模型404
        # 都走这里——检索已经成功，证据直接上桌，用户至少看得到菜谱
        trace["fallback"] = True
        trace["error"] = type(exc).__name__
        answer = "\n".join(
            f"【证据{i}】{doc.metadata[label_key]}\n{doc.page_content}"
            for i, (doc, _) in enumerate(docs, start=1)
        )

    # ---- 来源盖章：检索层已知的信息不让 LLM 抄——确定性的事交给代码 ----
    # 名字从 metadata 现取（菜名/展示名），保序去重防同名连锁重复
    trace["sources"] = list(dict.fromkeys(doc.metadata[label_key] for doc, _ in docs))
    return {"type": "answer", "reply": answer, "lock": lock, "trace": trace}


def _merge_filter(base: dict | None, extra: dict) -> dict:
    """两张过滤卡合一：base 是单片就直接并列，是 $and 就拆开续片（拼装规矩同 build_filter）。"""
    if base is None:
        return extra
    base_pieces = base["$and"] if "$and" in base else [base]
    return {"$and": base_pieces + [extra]}


def evidence_of(docs: list, label_key: str) -> str:
    """证据拼装：LLM 餐盘和抽取式兜底共用一块证据（改格式只动一处）。"""
    return "\n\n".join(f"【证据{i}】{doc.page_content}" for i, (doc, _) in enumerate(docs, start=1))


def _ask(prompt: str) -> str:
    """反问一轮；管道/非交互环境 EOF 不炸，返回空串走默认。"""
    try:
        return input(f"{prompt} ").strip()
    except EOFError:
        return ""


def ask(question: str, k: int = 3, max_price: int | None = None,
        difficulty: str | list[str] | None = None,
        history: list[dict] | None = None) -> tuple[str, str | None] | None:
    """CLI 包装：拿终端 input() 喂槽位，循环到出结果；打印版式和原来一模一样。
    返回 (答案文本, 本轮锁定的名字)；没检索到/没搜到返回 None（不进历史）。"""
    answers: dict = {}
    while True:
        result = ask_core(question, k=k, max_price=max_price, difficulty=difficulty,
                          history=history, answers=answers)
        if result["type"] == "input":
            answers[result["slot"]] = _ask(result["prompt"])
            continue
        if result["type"] == "miss":
            print(result["reply"])
            return None

        # ---- trace：先看路由和证据，再看答案 ----
        trace = result["trace"]
        print(f"仓库: {trace['db']}（{trace['reason']}）")
        print(f"证据: {trace['evidence']}")
        print(f"检索耗时: {trace['retrieve_sec']:.3f}秒")
        print("=" * 50)
        if trace["fallback"]:
            print(f"生成层提示：LLM 不可用（{trace['error']}），已退回抽取式兜底答案。")
            print(result["reply"])
        else:
            print(f"生成耗时: {trace['gen_sec']:.3f}秒")
            print(result["reply"])
        print(f"\n来源：{'、'.join(trace['sources'])}")
        return result["reply"], result["lock"]


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
        history: list[dict] = []  # 记忆仓库：最近 2 轮（问, 答）
        while True:
            try:
                q = input("\n问点什么> ").strip()
            except EOFError:
                break
            if not q:
                print("慢用，回见。")
                break
            result = ask(q, k=args.k, max_price=args.max_price,
                         difficulty=args.difficulty, history=history)
            if result:  # 没检索到/没搜到的轮次不进历史（记了也是脏账）
                answer, lock = result
                history.append({"question": q, "answer": answer, "lock": lock})
                del history[:-2]  # 只留最近 2 轮，金鱼记忆防撑爆小模型
