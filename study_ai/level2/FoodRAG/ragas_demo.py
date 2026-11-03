"""RAGAS 四指标计算过程演示台：把判卷官的草稿纸摊开给你看。

【设计笔记】

学 RAGAS 的骨架，手搓实现（不装框架）：四个指标每一步的中间产物全打印——
拆出的说法、逐条对账的判决和依据、反推出的问题、P@k 表——
公式（见 RAGAS.md）怎么落地的，一眼看到底。

和 eval.py 的分工（三层体检体系）：
    eval.py    = 客观题（路由/槽位/Hit@k），确定性代码判卷，秒出，每次改代码都跑；
    本文件     = 作文题（RAGAS 四指标），LLM 当裁判，慢，偶尔跑一发看生成层质量。

裁判的三个工种（拆说法/打判决/反推问题）全走 with_structured_output（第8关的表格合同）；
相似度用自家 bge-m3 嵌入算余弦——和检索层同一门手艺。

标准答案（ground_truth）从哪来：账本原文（问句锁定的菜的 page_content 拼接）——
它独立于检索结果（检索漏了这块，Recall 就会掉），demo 够用；
线上真评估得人写标准答案（贵，出题三来源见 RAGAS.md）。

耗时警告：真实生成 1 次 + 裁判 6 次 LLM 调用，本地 qwen3:4b 要几分钟——
ask.PROVIDER 切 "dashscope" 可秒级（build_llm 同款开关）。
"""

from __future__ import annotations

import argparse
import math

from pydantic import BaseModel, Field

from ask import PROVIDER, ask_core, build_llm
from router import DB_CONFIG, load_rows, match_db_names
from search import EMBEDDINGS

MAX_CLAIMS = 10  # 说法数封顶：qwen3:4b 判多了又慢又抖


# ==== 裁判的表格合同（LLM 只准往格子里填，不能自由发挥）====

class Claims(BaseModel):
    """拆说法合同的格子。"""
    claims: list[str] = Field(description="把文本拆成一条条独立的最小说法（只要事实内容，不要客套话）")


class Verdict(BaseModel):
    编号: int = Field(description="对应第几条说法")
    有出处: bool = Field(description="这条说法能否从证据里找到出处")
    依据: str = Field(description="有出处就抄证据原句片段，没有就写'证据中无'")


class VerdictList(BaseModel):
    判决: list[Verdict]


class ReverseQuestions(BaseModel):
    """反推问题合同的格子。"""
    questions: list[str] = Field(description="3 个这份答案像在回答的问题，每问一句，不带序号")


class ChunkUseful(BaseModel):
    编号: int = Field(description="对应第几块证据")
    有用: bool = Field(description="这块证据对回答该问题有用吗")
    理由: str


class ChunkVerdicts(BaseModel):
    判决: list[ChunkUseful]


def judge(schema: type[BaseModel], prompt: str):
    """喊裁判干活：带表格合同的一次调用。"""
    structured = build_llm().with_structured_output(schema)
    return structured.invoke(prompt)


def cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度：和检索层同一门手艺（bge-m3 嵌入 → 向量夹角）。"""
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def _fmt_evidence(contexts: list[str]) -> str:
    return "\n\n".join(f"【证据{i}】{c}" for i, c in enumerate(contexts, start=1))


# ==== 指标①：Faithfulness 忠实度 = 有出处的说法 ÷ 总说法 ====

def faithfulness(answer: str, contexts: list[str]) -> float | None:
    print("\n" + "=" * 60)
    print("指标① Faithfulness 忠实度（答案的每句话都得出示小票）")
    print("=" * 60)
    claims_model = judge(Claims, f"把下面的答案拆成一条条独立的最小说法（最多{MAX_CLAIMS}条）。\n答案：\n{answer}")
    claims = claims_model.claims[:MAX_CLAIMS]
    if not claims:
        print("× 拆说法失败（裁判返回空）")
        return None
    print(f"① 拆说法（LLM 干活Ⅰ）：拆出 {len(claims)} 条")
    claim_lines = "\n".join(f"{i}. {c}" for i, c in enumerate(claims, start=1))
    print(claim_lines)

    verdicts_model = judge(VerdictList,
        "逐条判决下面的说法能否从证据中找到出处。只能依据证据，证据里没有的就算没出处。\n\n"
        f"证据：\n{_fmt_evidence(contexts)}\n\n说法：\n{claim_lines}")
    verdicts = {v.编号: v for v in verdicts_model.判决}

    print(f"② 逐条对账（LLM 干活Ⅱ）：")
    supported = 0
    for i, c in enumerate(claims, start=1):
        v = verdicts.get(i)
        if v is None:
            print(f"   {i}. {c}\n      × 裁判漏判，按没出处算")
            continue
        mark = "✓" if v.有出处 else "✗"
        supported += int(v.有出处)
        print(f"   {i}. {mark} {c}\n      依据：{v.依据}")

    print(f"③ 套公式（代码干，零 LLM）：{supported}/{len(claims)} = {supported / len(claims):.3f}")
    return supported / len(claims)


# ==== 指标②：Answer Relevancy = 反推问题与原问题的相似度均值 ====

def answer_relevancy(question: str, answer: str) -> float | None:
    print("\n" + "=" * 60)
    print("指标② Answer Relevancy 答案相关性（答非所问检测）")
    print("=" * 60)
    rq = judge(ReverseQuestions, f"下面是一份答案。写出 3 个这份答案像在回答的问题。\n答案：\n{answer}")
    questions = [q for q in rq.questions if q.strip()][:3]
    if not questions:
        print("× 反推失败（裁判返回空）")
        return None

    print(f"① 反推问题（LLM 干活）：")
    q_vec = EMBEDDINGS.embed_query(question)
    sims = []
    for i, q in enumerate(questions, start=1):
        sim = cosine(q_vec, EMBEDDINGS.embed_query(q))
        sims.append(sim)
        print(f"   q{i}: {q}")
        print(f"       cos(原问题, q{i}) = {sim:.3f}")
    score = sum(sims) / len(sims)
    print(f"② 套公式（嵌入+代码干）：({'+'.join(f'{s:.3f}' for s in sims)}) ÷ {len(sims)} = {score:.3f}")
    return score


# ==== 指标③：Context Precision = 平均精确率 AP ====

def context_precision(question: str, contexts: list[str]) -> float | None:
    print("\n" + "=" * 60)
    print("指标③ Context Precision 上下文精确率（有用的是否排在前面）")
    print("=" * 60)
    numbered = "\n".join(f"【证据{i}】{c}" for i, c in enumerate(contexts, start=1))
    vd = judge(ChunkVerdicts,
        f"问题：{question}\n\n逐块判决下面的证据对回答该问题有没有用（能不能直接用上其中内容）。\n\n{numbered}")
    verdicts = {v.编号: v for v in vd.判决}

    useful_at = []  # (排名k, 是否有用)
    print("① 逐块判有用（LLM 干活）：")
    for i, c in enumerate(contexts, start=1):
        v = verdicts.get(i)
        ok = bool(v and v.有用)
        useful_at.append(ok)
        print(f"   【证据{i}】{'✓' if ok else '✗'} {(v.理由 if v else '裁判漏判按无用')[:50]}")

    print("② 算 AP（代码干，P@k=前k名里有用几块÷k，只在有用的位置计分）：")
    hits = 0
    ap_sum = 0.0
    for k, ok in enumerate(useful_at, start=1):
        if ok:
            hits += 1
            pk = hits / k
            ap_sum += pk
            print(f"   P@{k} = {hits}/{k} = {pk:.3f}  ← 累加")
    total_useful = sum(useful_at)
    if total_useful == 0:
        print("   全部无用，AP = 0")
        return 0.0
    score = ap_sum / total_useful
    print(f"③ AP = 累加和÷有用总数 = {ap_sum:.3f}÷{total_useful} = {score:.3f}")
    return score


# ==== 指标④：Context Recall = 对得上的说法 ÷ 标准答案总说法 ====

def context_recall(ground_truth: str, contexts: list[str]) -> float | None:
    print("\n" + "=" * 60)
    print("指标④ Context Recall 上下文召回率（该捞的捞齐没——要标准答案）")
    print("=" * 60)
    gt_claims_model = judge(Claims,
        f"把下面的标准答案拆成一条条独立的最小说法（最多{MAX_CLAIMS}条）。\n标准答案：\n{ground_truth}")
    gt_claims = gt_claims_model.claims[:MAX_CLAIMS]
    if not gt_claims:
        print("× 拆标准答案失败")
        return None
    claim_lines = "\n".join(f"{i}. {c}" for i, c in enumerate(gt_claims, start=1))
    print(f"① 拆标准答案（LLM 干活Ⅰ）：拆出 {len(gt_claims)} 条")
    print(claim_lines)

    vd = judge(VerdictList,
        "逐条判决下面的说法能否从证据中找到出处（判的是证据够不够，不是答案好不好）。\n\n"
        f"证据：\n{_fmt_evidence(contexts)}\n\n说法：\n{claim_lines}")
    verdicts = {v.编号: v for v in vd.判决}

    print("② 逐条对账（LLM 干活Ⅱ）：")
    supported = 0
    for i, c in enumerate(gt_claims, start=1):
        v = verdicts.get(i)
        ok = bool(v and v.有出处)
        supported += int(ok)
        print(f"   {i}. {'✓' if ok else '✗'} {c}")

    print(f"③ 套公式（代码干）：{supported}/{len(gt_claims)} = {supported / len(gt_claims):.3f}")
    return supported / len(gt_claims)


def build_ground_truth(question: str, db: str) -> str | None:
    """标准答案=账本原文（问句锁定的菜的 page_content 拼接）——独立于检索，检索漏了它就掉分。"""
    names = match_db_names(question, db)
    if not names:
        return None
    label_key = DB_CONFIG[db]["label_key"]
    rows = [r for r in load_rows(db) if r["metadata"][label_key] in names]
    return "\n\n".join(r["page_content"] for r in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS 四指标计算过程演示（LLM当裁判，慢）")
    parser.add_argument("question", nargs="?", default="锅包肉怎么做", help="要评估的问题")
    args = parser.parse_args()

    print(f"裁判模型供应商：{PROVIDER}（本地 qwen3:4b 要等几分钟，切 dashscope 秒级）")

    # ---- 先跑真实问答：四指标的原料（问题/证据/答案）全从这来 ----
    print(f"\n第一步：真实问答（问句：{args.question}）……")
    result = ask_core(args.question)
    if result["type"] != "answer":
        print(f"没拿到答案（{result['type']}），四指标没原料，收工。")
        return
    trace = result["trace"]
    answer, contexts = result["reply"], trace["contexts"]
    print(f"仓库: {trace['db']}（{trace['reason']}）")
    print(f"证据: {trace['evidence']}")
    print(f"答案:\n{answer[:300]}{'……' if len(answer) > 300 else ''}")
    if trace.get("fallback"):
        print("⚠ 这答案是兜底证据原文（LLM挂了），忠实度会虚高——仅供参考。")

    scores: dict[str, float | None] = {}
    scores["Faithfulness 忠实度"] = faithfulness(answer, contexts)
    scores["Answer Relevancy 相关性"] = answer_relevancy(args.question, answer)
    scores["Context Precision 精确率"] = context_precision(args.question, contexts)
    gt = build_ground_truth(args.question, trace["db"])
    if gt is None:
        print("\n指标④ Context Recall：跳过——问句没锁定菜名，没有账本标准答案（人写才够，demo不做假账）。")
        scores["Context Recall 召回率"] = None
    else:
        scores["Context Recall 召回率"] = context_recall(gt, contexts)

    print("\n" + "=" * 60)
    print("记分卡")
    print("=" * 60)
    for name, score in scores.items():
        print(f"  {name}: {f'{score:.3f}' if score is not None else '—（跳过）'}")
    print("\n读分指南：Faithfulness低=生成的锅；Faithfulness高但Recall低=检索/语料的锅；")
    print("Precision低=有用的没排前；Relevancy低=答非所问。分锅口诀见 RAGAS.md。")


if __name__ == "__main__":
    main()
