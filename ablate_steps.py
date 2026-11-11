"""MAX_STEPS 消融：3 / 5 / 8 / 12 四档，看在哪些题上"步数不够"。

为什么要做这个消融（MAX_STEPS 取值的依据）：
  - 定太小：复杂问题走不完，模型在"证据不够"的状态下被迫给结论
  - 定太大：失败路径的代价线性放大（每一步都是一次完整 LLM 调用，prompt 随对话增长）
  所以正确做法不是拍脑袋取 8，而是测出"多少步之后边际收益归零"。

题目设计：从 1 步能答到需要多轮取证，形成难度梯度。
每条记录：是否收敛、实际用了几步、prompt tokens、工具调用次数、是否降级、答案是否带证据。

用法：
  python ablate_steps.py                 # 默认 5,8,12
  python ablate_steps.py --steps 3,5,8   # 自定义
  python ablate_steps.py --repeat 2      # 每档重复跑（LLM 非确定性，单次不可信）
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import config
from agent import Agent
from context import ResultStore
from rag import Retriever
from tools import Tools

OUT = config.ROOT / "ablation_steps.json"

# 难度梯度：从「一步就能答」到「必须多轮取证」
# 注意每道题都写清「预期最少几步」，这样"步数不够"的判断才有依据，而不是看答案顺不顺眼。
TASKS = [
    {
        "name": "T1_单次检索",
        "input": "查一下 Redis 限流的滑动窗口是怎么实现的",
        "expect_min_steps": 1,
        "why": "单工具单次调用即可",
    },
    {
        "name": "T2_日志定位",
        "input": "线上报 UserService 空指针，日志在 logs/app.log，帮我找根因",
        "expect_min_steps": 2,
        "why": "读日志 → 基于日志判断（或检索文档）出结论",
    },
    {
        "name": "T3_日志+文档",
        "input": "线上报 UserService 空指针，日志在 logs/app.log。帮我定位根因，并说明这个异常在项目里通常怎么修",
        "expect_min_steps": 3,
        "why": "读日志 → 检索文档 → 综合出「根因+修复」两段",
    },
    {
        "name": "T4_多源交叉",
        "input": ("线上报 UserService 空指针，日志在 logs/app.log。"
                  "请给出根因、对应的修复方式，并查一下最近有没有相关代码改动可能是诱因。"),
        "expect_min_steps": 4,
        "why": "日志 + 文档 + git 三个来源，任何一个缺失都答不全",
    },
    {
        "name": "T5_工具失败纠错",
        "input": "看下 logs/not_exist.log 里的报错",
        "expect_min_steps": 2,
        "why": "读不存在的日志 → 按错误提示换文件。步数少反而危险：可能一次失败就放弃了",
        "is_failure_path": True,
    },
]


def run_one(task: dict, retriever: Retriever, max_steps: int) -> dict:
    store = ResultStore(trace_id=f"abl_{max_steps}_{task['name']}")
    tools = Tools(retriever, store, user_groups=["dev"])
    agent = Agent(tools, max_steps=max_steps)
    t0 = time.perf_counter()
    res = agent.run(task["input"])
    latency = int((time.perf_counter() - t0) * 1000)

    ans = res.answer or ""
    return {
        "task": task["name"],
        "max_steps": max_steps,
        "expect_min_steps": task["expect_min_steps"],
        "steps": res.steps,
        "converged": res.steps < max_steps,      # 没走满 = 模型自己收敛了
        "degraded": res.degraded,
        "degrade_reason": res.degrade_reason,
        "prompt_tokens": res.prompt_tokens,
        "completion_tokens": res.completion_tokens,
        "tool_calls": res.tool_calls_total,
        "tool_failed": res.tool_calls_failed,
        "duplicate_calls": res.duplicate_calls,
        "latency_ms": latency,
        "answer_chars": len(ans),
        # 四段式里有没有「证据」段 —— 用引用标记近似判断答案质量
        "has_evidence": ("[doc=" in ans) or ("commit=" in ans.lower()),
        "says_insufficient": any(w in ans for w in ("证据不足", "信息缺失", "无法确定")),
        "answer_head": ans[:120].replace("\n", " "),
    }


def summarize(rows: list[dict], steps: int) -> dict:
    sub = [r for r in rows if r["max_steps"] == steps]
    if not sub:
        return {}

    def med(key: str) -> float:
        return statistics.median(r[key] for r in sub)

    def spread(key: str) -> tuple[float, float]:
        vals = [r[key] for r in sub]
        return min(vals), max(vals)

    tk_lo, tk_hi = spread("prompt_tokens")
    return {
        "max_steps": steps,
        "n": len(sub),
        "converged": sum(r["converged"] for r in sub),
        "degraded": sum(r["degraded"] for r in sub),
        "has_evidence": sum(r["has_evidence"] for r in sub),
        "tool_calls": sum(r["tool_calls"] for r in sub),
        "prompt_tokens": sum(r["prompt_tokens"] for r in sub),
        "latency_ms": sum(r["latency_ms"] for r in sub),
        "latency_s": sum(r["latency_ms"] for r in sub) / 1000,
        # 单次不可信：同一档位多次运行的波动量级，是判断"档位差异是否显著"的标尺
        "tk_min": tk_lo, "tk_max": tk_hi,
        "steps_median": med("steps"),
        "steps_min": min(r["steps"] for r in sub),
        "steps_max": max(r["steps"] for r in sub),
    }


def per_task_median(rows: list[dict], task: str, steps: int, key: str) -> float | None:
    sub = [r for r in rows if r["task"] == task and r["max_steps"] == steps]
    return statistics.median(r[key] for r in sub) if sub else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="5,8,12")
    ap.add_argument("--repeat", type=int, default=1, help="每档重复次数，取中位数（LLM 非确定性）")
    ap.add_argument("--only", default="", help="只跑指定任务名（逗号分隔）")
    args = ap.parse_args()

    step_list = [int(x) for x in args.steps.split(",") if x.strip()]
    tasks = TASKS
    if args.only:
        want = {x.strip() for x in args.only.split(",")}
        tasks = [t for t in TASKS if t["name"] in want or t["name"].split("_")[0] in want]

    retriever = Retriever()
    retriever.load()
    print(f"[info] {len(retriever.chunks)} chunk，模型 {config.LLM_MODEL}，"
          f"任务 {len(tasks)} 个 × 档位 {step_list} × 重复 {args.repeat}\n")

    rows: list[dict] = []
    for ms in step_list:
        for t in tasks:
            for rep in range(args.repeat):
                print(f"  MAX_STEPS={ms:<3} {t['name']:<18}", end="", flush=True)
                try:
                    r = run_one(t, retriever, ms)
                except Exception as exc:  # noqa: BLE001
                    print(f" ✗ {type(exc).__name__}: {exc}")
                    continue
                r["rep"] = rep
                rows.append(r)
                flag = "收敛" if r["converged"] else "耗尽"
                deg = f" 降级({r['degrade_reason']})" if r["degraded"] else ""
                ev = "有证据" if r["has_evidence"] else "无证据"
                print(f" {r['steps']}步 {flag} {r['prompt_tokens']}tk {ev}{deg}")

    # ---- 汇总表 ----
    w = 22
    print("\n" + "=" * (18 + 6 + w * len(step_list)))
    print(f"{'任务':<18}{'预期≥':>6}" + "".join(f"{'MAX'+str(ms):>{w}}" for ms in step_list))
    by_task: dict[str, dict[int, list[dict]]] = {}
    for r in rows:
        by_task.setdefault(r["task"], {}).setdefault(r["max_steps"], []).append(r)
    for t in tasks:
        cells = []
        for ms in step_list:
            sub = by_task.get(t["name"], {}).get(ms, [])
            if not sub:
                cells.append("-")
                continue
            conv = sum(x["converged"] for x in sub)
            steps = statistics.median(x["steps"] for x in sub)
            tools = statistics.median(x["tool_calls"] for x in sub)
            cells.append(f"{conv}/{len(sub)} {steps:.0f}步/{tools:.0f}工具")
        print(f"{t['name']:<18}{t['expect_min_steps']:>6}" + "".join(f"{c:>{w}}" for c in cells))

    print("-" * (18 + 6 + w * len(step_list)))
    sums = [summarize(rows, ms) for ms in step_list]
    def line(label: str, key: str, fmt: str = "{}") -> str:
        return f"{label:<18}{'':>6}" + "".join(f"{fmt.format(s.get(key, 0)):>{w}}" for s in sums)
    print(line("收敛题数", "converged", "{}/" + str(sums[0]["n"] if sums else 0)))
    print(line("降级题数", "degraded"))
    print(line("有证据题数", "has_evidence"))
    print(line("工具调用总数", "tool_calls"))
    print(line("总 prompt tk", "prompt_tokens"))
    if args.repeat > 1:
        # 波动区间：判断"档位差异"是否大于"同档位抖动"的唯一依据
        # 注意 Python 3.11 的 f-string 不允许嵌套同类引号，先算好字符串再格式化
        tk_cells = [f"{s['tk_min']}~{s['tk_max']}" for s in sums]
        step_cells = [f"{s['steps_median']:.0f} ({s['steps_min']}~{s['steps_max']})" for s in sums]
        print(f"{'tk 波动区间':<18}{'':>6}" + "".join(f"{c:>{w}}" for c in tk_cells))
        print(f"{'步数 中位/区间':<18}{'':>6}" + "".join(f"{c:>{w}}" for c in step_cells))
    print(line("总耗时 s", "latency_s", "{:.1f}"))
    print("=" * (18 + 6 + w * len(step_list)))

    payload = {"steps": step_list, "repeat": args.repeat,
               "summary": sums, "rows": rows}
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] 明细已存 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())