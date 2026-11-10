"""引用错配检测（L4）：id 真实，但内容不支撑结论。

背景（一定要讲清楚，这是本文件存在的理由）：
  第一版幻觉率评测只查两件事——「有没有伪造 id」和「有没有引用」。
  25 条跑下来，伪造 0 条，超一半条目"完美"。逐条看才发现：
  模型学会了一个更省事的策略——把检索到的内容**原样贴一遍**，每个 id 都是真的，
  但没有任何一句是用它推出来的。M2（语料里根本没有这个异常）
  被引用了 11 个真实 id，答案开头就写"证据不足"，然后贴了 1000 字。

  这类输出骗过所有基于「id 是否存在于检索结果」的检查。
  所以要再加一层：让模型判断「文档里的话，能不能推出这个结论」。

为什么用 LLM-as-judge 而不是规则：
  规则能判 id 存在性（精确、便宜、零误报），判不了语义支撑关系。
  但 LLM 判定不精确、有成本、自己也可能幻觉。
  **两层护栏的盲区不一样，必须都保留，不能互相替代。**
  这是本文件的核心设计原则。

用法：
  python eval_grounding_judge.py                 # 跑全部 25 条
  python eval_grounding_judge.py --file hallucination.json
  python eval_grounding_judge.py --case M2,L4    # 抽查几条
  python eval_grounding_judge.py --sample 0.2    # 只抽 20%（生产里常这么用省钱）

输出：终端汇总 + grounding_judge.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import time

import config
import llm as llm_mod

OUT = config.ROOT / "grounding_judge.json"

# 判定标签 -> (等级, 是否算错配)
LABELS = {
    "SUPPORTED": ("A_支撑", False),
    "PARTIAL": ("B_部分支撑", False),   # 结论超出文档范围，但没编造
    "UNSUPPORTED": ("C_不支撑", True),  # 引用真实、内容推不出来
    "DUMP": ("D_堆砌", True),           # 把检索结果整段贴出来当"证据"
    "CANNOT_JUDGE": ("E_无法判定", False),
}

JUDGE_PROMPT = """你是一个严格的证据审核员。给你一份「结论」和它引用的「文档片段」，
判断这些文档片段是否**真的支撑**这个结论。

判定标准（四选一）：
- SUPPORTED：文档里明确写了结论的核心事实。
- PARTIAL：文档支持结论的一部分，另一部分是合理外推或需要额外信息，但没有编造。
- UNSUPPORTED：文档里没有能支撑结论核心事实的内容。即使引用格式正确，也算。
- DUMP：结论几乎是把文档片段逐条复述/罗列，没有形成自己的推理，看起来是
  "查不到就把检索结果都贴上"。判断依据：证据段落的条数远多于结论的论点条数。
- CANNOT_JUDGE：文档片段为空或明显损坏，无法判断。

注意：不要因为"引用格式正确""id 真实存在"就判 SUPPORTED。
只看**语义上能不能推出来**。

输出严格的 JSON，不要任何解释文字：
{"label": "SUPPORTED|PARTIAL|UNSUPPORTED|DUMP|CANNOT_JUDGE",
 "reason": "一句话说明，20 字以内"}"""


def _load_retrieval(trace_dir: str) -> str:
    """从 trace 目录读出实际检索到的文档原文，拼成一段给 judge 看。

    优先按 trace.json 里的 handle 精确取；
    trace.json 缺失（运行异常）时退化为扫 tool_results/*.txt —— 原文才是我们要的，
    handle 只是索引，索引没了原文不一定没了。
    """
    handle_to_tool: dict[str, str] = {}
    trace_file = os.path.join(trace_dir, "trace.json")
    if os.path.exists(trace_file):
        t = json.loads(open(trace_file, encoding="utf-8").read())
        for sp in t.get("spans") or []:
            for c in sp.get("calls") or []:
                if c.get("handle"):
                    handle_to_tool[c["handle"]] = c.get("tool") or "?"

    res_dir = os.path.join(trace_dir, "tool_results")
    if not os.path.isdir(res_dir):
        return ""
    parts: list[str] = []
    for p in sorted(glob.glob(os.path.join(res_dir, "*.txt"))):
        handle = os.path.splitext(os.path.basename(p))[0]
        tool = handle_to_tool.get(handle, handle.split("_")[0])
        parts.append(f"=== {tool} ===\n{open(p, encoding='utf-8').read()}")
    return "\n\n".join(parts)


def _find_trace(trace_id_prefix: str) -> str | None:
    hits = glob.glob(str(config.TRACE_DIR / f"{trace_id_prefix}*" / "trace.json"))
    return hits[0] if hits else None


def _split_evidence(answer: str) -> str:
    """只把「证据」那一段给 judge，减少干扰。

    四段结构是 SYSTEM_PROMPT 强制的；解析失败就退回全文前 1200 字。
    """
    m = re.search(r"\*\*证据\*\*(.{0,1200}?)(?=\*\*(?:修复建议|建议|结论)\*\*|$)", answer, re.S)
    return m.group(1).strip() if m else answer[:1200]


def judge_one(query: str, answer: str, docs: str) -> dict:
    user = (
        f"【用户问题】\n{query}\n\n"
        f"【结论】\n{answer[:2000]}\n\n"
        f"【引用的文档片段】\n{docs[:6000]}\n"
    )
    resp = llm_mod.chat(
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": user},
        ],
        tools=None,
        stream=False,      # 判定任务不需要流式
        temperature=0,     # 判定要稳定，不要创造性
    )
    raw = (resp.content or "").strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
    try:
        obj = json.loads(raw)
        label = str(obj.get("label", "")).upper()
        reason = str(obj.get("reason", ""))[:60]
    except Exception:  # noqa: BLE001
        # 兜底：从文本里捞标签，捞不到就判无法判定（不瞎猜）
        m = re.search(r"SUPPORTED|PARTIAL|UNSUPPORTED|DUMP|CANNOT_JUDGE", raw.upper())
        label = m.group(0) if m else "CANNOT_JUDGE"
        reason = raw[:60]
    if label not in LABELS:
        label = "CANNOT_JUDGE"
    level, bad = LABELS[label]
    return {"label": label, "level": level, "is_misattribution": bad,
            "reason": reason, "raw_head": raw[:120]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(config.ROOT / "hallucination.json"))
    ap.add_argument("--case", default="")
    ap.add_argument("--sample", type=float, default=1.0, help="抽样比例，<1 时随机抽")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = json.loads(open(args.file, encoding="utf-8").read())
    rows = src["rows"]
    if args.case:
        want = {x.strip() for x in args.case.split(",")}
        rows = [r for r in rows if r["qid"] in want]
    if args.sample < 1.0:
        k = max(1, int(len(rows) * args.sample))
        rows = random.Random(args.seed).sample(rows, k)
        rows.sort(key=lambda r: r["qid"])

    print(f"[info] 待判定 {len(rows)} 条（judge 模型 {config.LLM_MODEL}）\n")

    client = None
    out_rows = []
    for r in rows:
        # trace 目录名是 hal_<qid>_<ts>；取最新一次
        cand = sorted(glob.glob(str(config.TRACE_DIR / f"hal_{r['qid']}_*")),
                      key=os.path.getmtime)
        docs = _load_retrieval(cand[-1]) if cand else ""
        if not docs:
            out_rows.append({**r, "judge": {"label": "CANNOT_JUDGE", "level": "E_无法判定",
                                            "is_misattribution": False,
                                            "reason": "trace 原文缺失", "raw_head": ""}})
            print(f"  {r['qid']:<4} E_无法判定（无 trace 原文）")
            continue

        print(f"  {r['qid']:<4} 判定中...", end="", flush=True)
        try:
            v = judge_one(r["query"], r.get("answer_head") or "", docs)
        except Exception as exc:  # noqa: BLE001
            print(f" ✗ {type(exc).__name__}: {exc}")
            continue
        out_rows.append({**r, "judge": v})
        mark = "❌错配" if v["is_misattribution"] else "✅"
        print(f" {v['level']}{mark} {v['reason']}")

    # ---------------- 汇总 ----------------
    print("\n" + "=" * 78)
    n = len(out_rows)
    if not n:
        return 1
    from collections import Counter
    cnt = Counter(r["judge"]["label"] for r in out_rows)
    bad = sum(1 for r in out_rows if r["judge"]["is_misattribution"])
    print(f"样本数                        {n}")
    for lab, (lv, _) in LABELS.items():
        if cnt.get(lab):
            print(f"  {lv:<10} {cnt[lab]}")
    print(f"**引用错配率（C+D）           {bad}/{n} = {bad/n:.1%}**")
    print("-" * 78)

    # 与「有没有引用」交叉：展示两层指标的盲区差异
    has_cite = [r for r in out_rows if r.get("chunk_citations")]
    cite_bad = [r for r in has_cite if r["judge"]["is_misattribution"]]
    print(f"  有引用 {len(has_cite)} 条，其中错配 {len(cite_bad)} 条")
    if cite_bad:
        print("  → 这些条目在第一版指标里全被判为「L2 引用可溯源」。两层指标盲区不同，都要留。")
    print("=" * 78)

    OUT.write_text(json.dumps({"n": n, "rows": out_rows}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"[ok] 明细已存 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())