"""幻觉率评测：答案里引用的证据，是否真的来自本次检索结果。

为什么不能只看"有没有 [doc=" 标记：
  模型完全可以编一个格式正确但不存在的 id —— `[doc=xxx id=deadbeef]`。
  这种答案带引用、_check_grounding 会放行，但它比"没有引用"更危险：
  因为它看起来有出处。

所以这里做三级判定：
  L1 无引用   —— 答案里没有任何证据标记（可能诚实，也可能是纯参数记忆）
  L2 引用存在 —— 有标记，且 id 都在本次实际检索到的 chunk 里  ✅
  L3 引用伪造 —— 有标记，但至少一个 id 不在检索结果里          ❌
  L4 引用错配 —— id 存在，但该 chunk 与问题无关（需要人工/模型判断，本脚本不判）

用法：
  python eval_hallucination.py            # 跑评测集全部 25 条
  python eval_hallucination.py --limit 8  # 先跑 8 条看看
  python eval_hallucination.py --case S1,M5

输出：终端汇总 + hallucination.json
"""
from __future__ import annotations

import argparse
import json
import re
import time

import config
from agent import Agent
from context import ResultStore
from eval_set import EVAL_SET
from rag import Retriever
from tools import Tools

OUT = config.ROOT / "hallucination.json"

# 证据标记的三种形态（来自 SYSTEM_PROMPT 的约定）
PATTERNS = [
    re.compile(r"\[doc=([^\s\]]+)\s+id=([0-9a-f]+)\]"),      # [doc=手册 id=195fd4ad638e]
    re.compile(r"\[id=([0-9a-f]{8,})\]", ),                   # [id=195fd4ad638e]
    re.compile(r"commit=([0-9a-f]{6,})", re.IGNORECASE),      # commit=abc1234
]


def extract_citations(answer: str) -> list[tuple[str, str]]:
    """返回 [(kind, ref)]，kind ∈ {'chunk', 'commit'}。"""
    found: list[tuple[str, str]] = []
    for m in PATTERNS[0].finditer(answer):
        found.append(("chunk", m.group(2)))
    for m in PATTERNS[1].finditer(answer):
        found.append(("chunk", m.group(1)))
    for m in PATTERNS[2].finditer(answer):
        found.append(("commit", m.group(1)))
    # 去重保序
    seen, out = set(), []
    for x in found:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def classify(answer: str, retrieved_ids: set[str], retrieved_commits: set[str]) -> dict:
    cites = extract_citations(answer)
    chunk_refs = [r for k, r in cites if k == "chunk"]
    commit_refs = [r for k, r in cites if k == "commit"]

    said_insufficient = any(w in answer for w in ("证据不足", "信息缺失", "无法确定"))

    if not cites:
        level = "L1_无引用" if not said_insufficient else "L1b_诚实说不知道"
        bad = False
    else:
        # 只校验能校验的：chunk id 必须在本次检索结果里
        # commit sha 不校验（git 查询结果没进这一层），单独计为 unverifiable
        fake = [r for r in chunk_refs if r not in retrieved_ids]
        if fake:
            level = "L3_引用伪造"
            bad = True
        elif chunk_refs or commit_refs:
            level = "L2_引用可溯源"
            bad = False
        else:
            level = "L1_无引用"
            bad = False

    return {
        "citations": chunk_refs + commit_refs,
        "chunk_citations": chunk_refs,
        "commit_citations": commit_refs,
        "fake_citations": [r for r in chunk_refs if r not in retrieved_ids],
        "unverifiable_commits": [r for r in commit_refs if r not in retrieved_commits],
        "level": level,
        "is_hallucination": bad,
        "said_insufficient": said_insufficient,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--case", default="", help="只跑指定 qid，逗号分隔")
    args = ap.parse_args()

    retriever = Retriever()
    retriever.load()
    all_chunk_ids = {c.chunk_id for c in retriever.chunks}

    cases = EVAL_SET
    if args.case:
        want = {x.strip() for x in args.case.split(",")}
        cases = [q for q in EVAL_SET if q.qid in want]
    if args.limit:
        cases = cases[: args.limit]

    print(f"[info] {len(cases)} 条 × 完整 Agent 运行（模型 {config.LLM_MODEL}）\n")

    rows = []
    for q in cases:
        # 关键：每次运行用独立 trace，并且从结果里取"本次实际检索到的 chunk"
        trace_id = f"hal_{q.qid}_{int(time.time())}"
        store = ResultStore(trace_id)
        tools = Tools(retriever, store, user_groups=["dev"])
        agent = Agent(tools)

        print(f"  {q.qid:<4} {q.query[:38]:<40}", end="", flush=True)
        t0 = time.perf_counter()
        try:
            res = agent.run(q.query)
        except Exception as exc:  # noqa: BLE001
            print(f" ✗ {type(exc).__name__}: {exc}")
            continue
        ms = int((time.perf_counter() - t0) * 1000)

        # 从 trace 里读本次实际给过模型的 chunk id（工具返回里出现的 id=）
        trace_file = config.TRACE_DIR / trace_id / "trace.json"
        retrieved_ids: set[str] = set()
        retrieved_commits: set[str] = set()
        if trace_file.exists():
            tr = json.loads(trace_file.read_text(encoding="utf-8"))
            for m in tr.get("messages", []):
                if m.get("role") == "tool":
                    content = m.get("content") or ""
                    retrieved_ids |= set(re.findall(r"id=([0-9a-f]{8,})", content))
                    retrieved_commits |= set(re.findall(r"\b([0-9a-f]{7,40})\b", content))
        # 兜底：trace 没写到（或解析失败）时，退化为"全语料 id 都算可引用"
        if not retrieved_ids:
            retrieved_ids = all_chunk_ids

        verdict = classify(res.answer or "", retrieved_ids, retrieved_commits)
        rows.append({
            "qid": q.qid, "kind": q.kind, "query": q.query,
            "steps": res.steps, "degraded": res.degraded,
            "degrade_reason": res.degrade_reason,
            "latency_ms": ms, "answer_chars": len(res.answer or ""),
            "answer_head": (res.answer or "")[:160].replace("\n", " "),
            **verdict,
        })
        mark = "❌伪造" if verdict["is_hallucination"] else ("✅" if verdict["level"].startswith("L2") else "○")
        print(f" {res.steps}步 {verdict['level']}{mark}")

    # ---------------- 汇总 ----------------
    print("\n" + "=" * 78)
    n = len(rows)
    if not n:
        print("没有成功运行的样本")
        return 1

    def cnt(level: str) -> int:
        return sum(1 for r in rows if r["level"] == level)

    n_fake = sum(1 for r in rows if r["is_hallucination"])
    n_cite = sum(1 for r in rows if r["chunk_citations"])
    n_honest = cnt("L1b_诚实说不知道")
    n_nocite = cnt("L1_无引用")

    print(f"样本数                    {n}")
    print(f"有 chunk 引用             {n_cite}  ({n_cite/n:.0%})")
    print(f"  引用全部可溯源 (L2)      {cnt('L2_引用可溯源')}")
    print(f"  引用含伪造 id (L3)       {n_fake}  ← 幻觉率 {n_fake/n:.1%}")
    print(f"无引用但诚实说不知道 (L1b)  {n_honest}")
    print(f"无引用且未声明 (L1)        {n_nocite}  ← 可能是纯参数记忆，需人工看")
    print(f"Agent 自己标了降级         {sum(1 for r in rows if r['degraded'])}")
    print("-" * 78)

    # 分档
    for kind in ("lexical", "semantic", "mixed"):
        sub = [r for r in rows if r["kind"] == kind]
        if not sub:
            continue
        f = sum(1 for r in sub if r["is_hallucination"])
        c = sum(1 for r in sub if r["chunk_citations"])
        print(f"  {kind:<10} n={len(sub):<3} 有引用 {c}  伪造 {f}")

    bad = [r for r in rows if r["is_hallucination"]]
    if bad:
        print("\n[!] 伪造引用明细：")
        for r in bad:
            print(f"  {r['qid']}: 引用了不存在的 {r['fake_citations']}")
    nocite = [r for r in rows if r["level"] == "L1_无引用"]
    if nocite:
        print("\n[!] 无引用且没声明证据不足（需人工判断是否是参数记忆）：")
        for r in nocite:
            print(f"  {r['qid']}: {r['answer_head'][:80]}")
    print("=" * 78)

    OUT.write_text(json.dumps({"n": n, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] 明细已存 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())