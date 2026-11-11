"""检索评测：recall@k / MRR / nDCG@k，四组对比（bm25 / vector / hybrid / hybrid+rerank）。

用法：
    python eval_rag.py                 # 跑三组（无 embedding key 时 vector/hybrid 自动标注 SKIP）
    python eval_rag.py --k 1,3,5       # 改 k 列表
    python eval_rag.py --only bm25     # 只跑一组
    python eval_rag.py --dump          # 打印每条 query 的 top5 明细（调错用）

为什么自己写指标而不是用 ragas/trulens：
    指标就四行公式，引入框架只会多一层黑盒，出了问题不好定位。
    自己写还能顺手做"gold 唯一匹配"的硬校验。

三个指标为什么都要看：
    recall@5   —— 召回管道上限：gold 没进 top5，后面 rerank 再强也救不回来。
    MRR        —— 第一个正确结果的排名，看"最相关的那份排第几"，是用户实际感受。
    nDCG@5     —— 有多 gold 时看"所有相关的是否都挤在前面"，recall 只数个数不看位置。

诚实声明：评测集是我看了语料写的（dev set），recall@5 的理论上限是 1.0
（top5 已经占了 24 个 chunk 的 21%）。所以**看三组之间的相对差异，不看绝对值**。
真正有说服力的是「语义档里 vector 明显强于 bm25」这种跨档一致的结论。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import config
from eval_set import EVAL_SET, Query, stats
from rag import Retriever

MODES = ("bm25", "vector", "hybrid", "hybrid+rerank")
MODE_LABEL = {"bm25": "BM25", "vector": "向量", "hybrid": "混合+RRF",
              "hybrid+rerank": "混合+RRF+rerank"}


def split_mode(mode: str) -> tuple[str, bool | None]:
    """'hybrid+rerank' -> ('hybrid', True)；'hybrid' -> ('hybrid', False)。

    显式传 False 而不是 None：评测要的是「已知的对照组」，不能跟着 env 变量漂。
    """
    if mode.endswith("+rerank"):
        return mode[: -len("+rerank")], True
    return mode, False


# ---------------- gold 定位（宁可崩，不要静默算错） ----------------

def resolve_gold(r: Retriever) -> dict[str, list[str]]:
    """把评测集里的「章节路径尾部」映射成 chunk_id。

    规则：header_path 以 gold_path 结尾（按 ' > ' 分段比较，避免 "2. HikariCP" 吃掉 "2. HikariCP 连接池" 的歧义）。
    允许匹配到多个 chunk（超长小节被 _split_long 切开后算多个 chunk，任意一个召回都算命中）。
    匹配到 0 个 → 直接抛错，说明评测集和语料对不上，继续算下去的数字全是错的。
    """
    index: dict[str, list[str]] = {}
    for c in r.chunks:
        parts = c.header_path.split(" > ")
        for n in range(1, len(parts) + 1):
            index.setdefault(" > ".join(parts[-n:]), []).append(c.chunk_id)

    resolved: dict[str, list[str]] = {}
    missing: list[str] = []
    for q in EVAL_SET:
        ids: list[str] = []
        for g in q.gold:
            hit = index.get(g, [])
            if not hit:
                missing.append(f"{q.qid}: {g}")
            ids.extend(hit)
        if ids:
            resolved[q.qid] = sorted(set(ids))

    if missing:
        raise RuntimeError("以下 gold 在语料里匹配不到任何 chunk，检查评测集的章节路径：\n  "
                           + "\n  ".join(missing))
    return resolved


# ---------------- 指标 ----------------

def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & gold) / len(gold)


def mrr(ranked: list[str], gold: set[str]) -> float:
    for i, cid in enumerate(ranked, start=1):
        if cid in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    """二值相关性（相关=1，不相关=0）下的 nDCG。

    注意这里用的是「有多少 gold 就把 IDCG 按多少算」的最严格版本：
    只召回 1 个 gold 里有 2 个 → 即使排第一，nDCG 也只有 1/log2(3) / (1+1/log2(3)) ≈ 0.61。
    这比很多库默认的「宽松版」更苛刻，好处是不会出现「只找到一个还拿满分」的假象。
    """
    if not gold:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 1) for i, cid in enumerate(ranked[:k], start=1) if cid in gold)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / idcg if idcg else 0.0


# ---------------- 跑一组 ----------------

def run_mode(r: Retriever, mode: str, gold_map: dict[str, list[str]], ks: list[int],
             dump: bool = False) -> dict:
    depth = max(max(ks), config.RETRIEVE_TOP_K)   # 多取一些，方便看 gold 掉到第几名
    base_mode, use_rerank = split_mode(mode)
    rows = []
    errors = []
    rerank_stats: list[dict] = []
    for q in EVAL_SET:
        st: dict = {}
        try:
            hits = r.search(q.query, top_k=depth, mode=base_mode, rerank=use_rerank, stats=st)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{q.qid}: {exc}")
            continue
        rerank_stats.append({"qid": q.qid, **st})
        ranked = [h["chunk_id"] for h in hits]
        gold = set(gold_map.get(q.qid, []))
        rows.append({"q": q, "ranked": ranked, "gold": gold,
                     "first_hit_rank": next((i for i, c in enumerate(ranked, 1) if c in gold), None),
                     "top_scores": hits[0].get("scores", [])})

    if errors:
        # 一组里有 query 跑失败，整组作废：部分成功的数字没有意义
        raise RuntimeError(f"[{mode}] 有 {len(errors)} 条 query 执行失败：\n  " + "\n  ".join(errors))

    scored = [row for row in rows if row["gold"]]           # gold 非空才计入 recall/mrr/ndcg
    controls = [row for row in rows if not row["gold"]]     # 对照项单独统计

    def agg(subset: list[dict]) -> dict:
        if not subset:
            return {}
        out: dict[str, float] = {}
        for k in ks:
            out[f"recall@{k}"] = sum(recall_at_k(x["ranked"], x["gold"], k) for x in subset) / len(subset)
        out["mrr"] = sum(mrr(x["ranked"], x["gold"]) for x in subset) / len(subset)
        out[f"ndcg@{max(ks)}"] = sum(ndcg_at_k(x["ranked"], x["gold"], max(ks)) for x in subset) / len(subset)
        out["miss"] = sum(1 for x in subset if not (set(x["ranked"][:max(ks)]) & x["gold"]))
        return out

    by_kind: dict[str, dict] = {}
    for kind in ("lexical", "semantic", "mixed"):
        by_kind[kind] = agg([x for x in scored if x["q"].kind == kind])

    result = {
        "mode": mode,
        "base_mode": base_mode,
        "overall": agg(scored),
        "by_kind": by_kind,
        "rerank_stats": rerank_stats,
        # 对照项：gold 为空，看 top1 相似度是否"硬凑"。这里只看 top1 是什么，
        # 因为「不相关时能不能说不相关」本质上要模型判，检索层只能观察它给了什么。
        "control": [
            {"qid": x["q"].qid, "query": x["q"].query,
             "top1": next((h for h in r.search(x["q"].query, top_k=1, mode=base_mode,
                                               rerank=use_rerank)), None)}
            for x in controls
        ],
        "rows": rows,
    }
    if dump:
        _dump(result, r)
    return result


def _dump(res: dict, r: Retriever) -> None:
    by_id = {c.chunk_id: c.header_path for c in r.chunks}
    print(f"\n===== {MODE_LABEL[res['mode']]} 明细 =====")
    for x in res["rows"]:
        q: Query = x["q"]
        top5 = x["ranked"][:5]
        flag = "✓" if (x["gold"] and set(top5) & x["gold"]) else ("·" if not x["gold"] else "✗")
        print(f"[{q.qid}] {flag} 首命中={x['first_hit_rank']}  {q.query}")
        for rk, cid in enumerate(top5, start=1):
            mark = "*" if cid in x["gold"] else " "
            print(f"        {rk}.{mark} {by_id.get(cid, cid)}")
        if q.gold:
            print(f"        gold: {list(q.gold)}")
        if q.confuser:
            print(f"        易混: {q.confuser}")


# ---------------- 报告 ----------------

def _fmt(v: float | int) -> str:
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def report(results: dict[str, dict], ks: list[int], skipped: dict[str, str]) -> str:
    lines: list[str] = []
    lines.append("=" * 78)
    st = stats()
    lines.append(f"评测集 {st['total']} 条（计入评分 {st['scored']} 条，对照项 {st['total'] - st['scored']} 条）"
                 f"  分档：lexical {st['by_kind']['lexical']} / semantic {st['by_kind']['semantic']} / mixed {st['by_kind']['mixed']}")
    lines.append("=" * 78)

    cols = [f"recall@{k}" for k in ks] + ["mrr", f"ndcg@{max(ks)}", "miss"]
    head = f"{'档位':<10}{'方法':<12}" + "".join(f"{c:>11}" for c in cols)
    for mode, res in results.items():
        for scope in ("overall", "lexical", "semantic", "mixed"):
            m = res["overall"] if scope == "overall" else res["by_kind"][scope]
            if not m:
                continue
            name = "全部" if scope == "overall" else scope
            row = f"{name:<10}{MODE_LABEL[mode]:<12}" + "".join(f"{_fmt(m[c]):>11}" for c in cols)
            lines.append(row)
        lines.append("-" * 78)

    def _sev(res: dict) -> str:
        covered = [x for x in res["rows"] if x["gold"] and x["first_hit_rank"]]
        worst = max((x["first_hit_rank"] for x in covered), default=0)
        return f"最大首命中排名={worst}"
    for mode, res in results.items():
        lines.append(f"[覆盖] {MODE_LABEL[mode]}：{_sev(res)}  "
                     f"（越大说明越靠后的块才被捡回来，1 表示排第一就命中）")

    # rerank 闸门统计：拦截了多少次、平均耗时、被拦的题号。
    # 这组数字比 recall 更能说明问题——它记录了「模型在哪些题上承认自己没把握」。
    for mode, res in results.items():
        rs = [x for x in res.get("rerank_stats", []) if x.get("rerank_requested")]
        if not rs:
            continue
        applied = [x for x in rs if x.get("rerank_applied")]
        blocked = [x for x in rs if not x.get("rerank_applied")]
        lat = sorted(x.get("rerank_ms", 0) for x in rs if (x.get("rerank_ms") or 0) > 0)
        p50 = lat[len(lat) // 2] if lat else 0
        errs = [x for x in rs if x.get("rerank_error")]
        lines.append(f"[rerank] {MODE_LABEL[mode]}：采信 {len(applied)}/{len(rs)}，"
                     f"闸门拦截 {len(blocked)} 条 {[x['qid'] for x in blocked]}，"
                     f"延迟 p50={p50:.0f}ms p95={lat[int(len(lat) * 0.95)] if lat else 0:.0f}ms")
        if errs:
            lines.append(f"[rerank] 降级（调用失败）{len(errs)} 次：{errs[0].get('rerank_error')}")

    for mode, why in skipped.items():
        lines.append(f"[SKIP] {MODE_LABEL[mode]}：{why}")

    # 上界警告：语料只有 24 个 chunk，top5 就占了 21%，指标天然接近满分。
    # 不写这段话，很容易把「1.000」当成「检索做得好」，这是评测里最常见的自我欺骗。
    if results and all(r["overall"].get(f"recall@{max(ks)}", 0) >= 0.999 for r in results.values()):
        lines.append(f"[!] 所有方法 recall@{max(ks)} 都到 1.000：语料只有在这批 query 上被覆盖完了，"
                     f"**不能**据此说检索效果好——{max(ks)} 个块已占 24 个 chunk 的 {max(ks)/24:.0%}。")
        lines.append("[!] 要拉开差距请看：首命中排名（mrr）、rerank 后的 top1 准确率，"
                     "或把语料扩到百级 chunk 后重测。")

    # 对照项
    for mode, res in results.items():
        for c in res["control"]:
            t1 = c["top1"]
            t1s = f"{t1['header_path']}（rank1）" if t1 else "无结果"
            lines.append(f"[对照项 {c['qid']} | {MODE_LABEL[mode]}] top1 = {t1s}")
    lines.append("=" * 78)
    return "\n".join(lines)


def save_json(results: dict[str, dict], skipped: dict[str, str]) -> Path:
    out = config.INDEX_DIR / "eval_rag_result.json"
    payload = {
        "modes": {m: {"overall": r["overall"], "by_kind": r["by_kind"],
                      "control": r["control"]} for m, r in results.items()},
        "skipped": skipped,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", default="1,3,5")
    ap.add_argument("--only", default="", help="逗号分隔：bm25,vector,hybrid,hybrid+rerank")
    ap.add_argument("--dump", action="store_true")
    args = ap.parse_args()
    ks = [int(x) for x in args.k.split(",") if x.strip()]
    modes = [m for m in (args.only.split(",") if args.only else MODES) if m]

    r = Retriever()
    r.load()
    print(f"[info] 装载完成：{len(r.chunks)} 个 chunk，向量索引={'启用' if r.embedding_enabled else '未启用'}")

    gold_map = resolve_gold(r)
    print(f"[info] gold 解析完成：{len(gold_map)} 条 query 有 gold（对照项 gold 为空，已单独统计）")

    results: dict[str, dict] = {}
    skipped: dict[str, str] = {}
    for mode in modes:
        base_mode, _ = split_mode(mode)
        if base_mode in ("vector", "hybrid") and not r.embedding_enabled:
            skipped[mode] = "向量索引未启用（无 EMBEDDING_API_KEY 或 build_index 时构建失败）"
            continue
        try:
            results[mode] = run_mode(r, mode, gold_map, ks, dump=args.dump)
        except Exception as exc:  # noqa: BLE001
            skipped[mode] = f"执行失败：{exc}"

    print(report(results, ks, skipped))
    path = save_json(results, skipped)
    print(f"[ok] 结果已存 {path}")
    return 0 if not skipped else 2


if __name__ == "__main__":
    raise SystemExit(main())