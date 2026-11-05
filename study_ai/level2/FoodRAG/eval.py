"""判卷台：批量跑考题，对照标准答案，出记分卡（确定性判卷，不烧 LLM）。

【设计笔记】

学 RAGAS 的分层思想，砍到能手搓（公式详见 RAGAS.md）：
    RAGAS 管两层——上下文对(Recall/Precision) + 答案对(Faithfulness/Relevancy)，判卷官是 LLM。
    咱们这层管决策——选仓/槽位/拒答/证据命中(Hit@k)/证据属性，判卷官是确定性代码：
    零成本、秒出、结果永远稳定——便宜稳定的指标先用上，贵的会抖的后加（RAGAS式忠实度）。

不烧 LLM 的原理：把 ask.build_llm 换成必炸的假货——
    生成层走 except 分支退抽取式兜底，但 trace(db/reason/evidence/lock)原样带出，
    判卷要的东西一样不少；顺带每跑一遍都免费回归一次"LLM挂了退证据"的兜底路径（一鱼两吃）。

考题从哪来（RAGAS.md 落地路线）：bug 变考题（最真）> 自动造题（凑数）> 人工手写（最贵）。
惯例：每个真实 bug 进考卷，改完代码必跑 eval——修好的 bug 再犯当场被抓。

考卷字段全是可选检查项，判卷只查写了的（出题的人说了算）：
    question            问题（必填）
    answers/history     预喂的槽位答案/对话历史（可选）
    expect_type         answer / miss / input（必填）
    expect_db           选仓必须等于（answer 时查）
    expect_names        必须出现在证据里的名字 = Hit@k（answer 时查）
    expect_evidence_count 证据块数必须等于（answer 时查）
    expect_all          每块证据的账本字段必须等于/属于（值可为 list）——如 {"难度": "简单"}
    expect_lte          每块证据的账本字段必须 ≤——如 {"人均": 50}
    expect_reason_contains  路由理由必须包含（验"历史带锁"等标记）
    note                出题理由（记分卡展示用）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import ask
from router import DB_CONFIG, load_rows

CASES_JSONL = Path(__file__).parent / "data" / "eval_cases.jsonl"


# ---- 不烧 LLM：换必炸假货，答案走兜底返回，trace 照样带出 ----
def _no_llm():
    raise RuntimeError("判卷不烧LLM：这层考路由+检索，生成层质量后考（见RAGAS.md）")


ask.build_llm = _no_llm


def build_meta_index() -> dict[str, dict]:
    """两本账本拼成 {展示标签: metadata}——判卷的 ground truth。

    trace.evidence 里只有名字（菜名/展示名），要查证据的难度/商圈/人均，
    拿名字回账本反查 metadata——账本就是标准答案库。
    """
    index: dict[str, dict] = {}
    for db in ("menu", "restaurant"):
        label_key = DB_CONFIG[db]["label_key"]
        for row in load_rows(db):
            index[row["metadata"][label_key]] = row["metadata"]
    return index


def run_case(case: dict, meta_index: dict[str, dict]) -> list[str]:
    """跑一题，返回失败清单（空列表 = 通过）。"""
    fails: list[str] = []
    result = ask.ask_core(case["question"], history=case.get("history"),
                          answers=case.get("answers", {}), use_cache=False,
                          eaten=case.get("eaten", []))  # []=不避让：考卷不吃真饭账，保持确定性

    # ① 类型三合同：answer / miss / input
    if result["type"] != case["expect_type"]:
        return [f"类型 期望{case['expect_type']} 实际{result['type']}"]

    if result["type"] == "input":
        # ② 反问槽位对不对（该问的问了没、问错槽没有）
        if case.get("expect_slot") and result["slot"] != case["expect_slot"]:
            fails.append(f"槽位 期望{case['expect_slot']} 实际{result['slot']}")
        return fails

    if result["type"] == "miss":
        return fails  # ③ 该说没有的说了没有 = 满分（宁可没有不推错的）

    trace = result["trace"]

    # ④ 选仓
    if case.get("expect_db") and trace["db"] != case["expect_db"]:
        fails.append(f"仓库 期望{case['expect_db']} 实际{trace['db']}")

    # ⑤ Hit@k：期望名字必须全在证据里
    evidence = set(trace["evidence"])
    for name in case.get("expect_names", []):
        if name not in evidence:
            fails.append(f"证据缺[{name}]（实际{trace['evidence']}）")

    # ⑤b 排除名单：吃过的绝不能出现（吃饭账本联动的专属检查——Hit@k 的反面）
    for name in case.get("expect_not_names", []):
        if name in evidence:
            fails.append(f"避让失败，[{name}]吃过还被推荐（实际{trace['evidence']}）")

    # ⑥ 证据块数（如 6人局 k=8）
    if case.get("expect_evidence_count") and len(trace["evidence"]) != case["expect_evidence_count"]:
        fails.append(f"块数 期望{case['expect_evidence_count']} 实际{len(trace['evidence'])}")

    # ⑦ 证据属性全查：拿名字回账本反查（值 list = 属于关系，单值 = 相等）
    for key, want in case.get("expect_all", {}).items():
        for name in trace["evidence"]:
            got = meta_index.get(name, {}).get(key, "（账本无此字段）")
            ok = got in want if isinstance(want, list) else got == want
            if not ok:
                fails.append(f"[{name}]的{key} 期望{want} 实际{got}")

    # ⑧ 数值上界：人均≤X 这类（None/超界都算挂）
    for key, cap in case.get("expect_lte", {}).items():
        for name in trace["evidence"]:
            got = meta_index.get(name, {}).get(key)
            if got is None or got > cap:
                fails.append(f"[{name}]的{key} 期望≤{cap} 实际{got}")

    # ⑩ 集合覆盖：evidence 的账本字段必须至少含列表里每个值各一次
    #    （荤素搭配/难度混搭配额查这个——expect_all 管"全是"，expect_cover 管"都得有"）
    for key, values in case.get("expect_cover", {}).items():
        got_values = {meta_index.get(n, {}).get(key) for n in trace["evidence"]}
        for v in values:
            if v not in got_values:
                fails.append(f"覆盖[{key}]缺[{v}]（实际{sorted(x for x in got_values if x)}）")

    # ⑨ 路由理由标记（"历史带锁"这类过程证据）
    if case.get("expect_reason_contains") and case["expect_reason_contains"] not in trace["reason"]:
        fails.append(f"理由缺[{case['expect_reason_contains']}]（实际{trace['reason']}）")

    return fails


def main() -> None:
    parser = argparse.ArgumentParser(description="FoodRAG 判卷台（确定性评估，不烧LLM）")
    parser.add_argument("--cases", default=str(CASES_JSONL), help="考卷路径（默认 data/eval_cases.jsonl）")
    args = parser.parse_args()

    cases = [json.loads(line) for line in Path(args.cases).read_text(encoding="utf-8").splitlines() if line.strip()]
    meta_index = build_meta_index()

    print(f"开考：{len(cases)} 题（判卷不烧LLM，只考路由+检索+槽位）")
    print("=" * 60)
    passed = 0
    for i, case in enumerate(cases, start=1):
        fails = run_case(case, meta_index)
        if not fails:
            passed += 1
            print(f"[PASS] {i:02d} {case['question']}（{case.get('note', '')}）")
        else:
            print(f"[FAIL] {i:02d} {case['question']}（{case.get('note', '')}）")
            for f in fails:
                print(f"       × {f}")

    print("=" * 60)
    print(f"记分卡：{passed}/{len(cases)} 通过")
    if passed < len(cases):
        print("有挂科——改完代码重跑，别让修好的 bug 复活。")
        sys.exit(1)
    print("全绿。")
    sys.exit(0)


if __name__ == "__main__":
    main()
