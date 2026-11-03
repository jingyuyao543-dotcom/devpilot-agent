"""回归测试：跑固定 case，比对关键数字，发现"改 A 坏 B"。

为什么需要它（面试可讲）：
  本项目真实踩过 —— 为了让 FetchToolResult 不被重复压缩，改了 3 行代码，
  结果预算被撑爆，_enforce_budget() 把第一步的日志扔了，
  token 从 10451 掉到 4817、工具从 6 次掉到 4 次，模型只会说"证据不足"。
  这种"数值突变"如果只看最终答案，很容易被忽略（答案确实是通顺的中文）。
  回归脚本把关键数字记下来逐次比对，突变立刻暴露。

用法：
  python regress.py                 # 跑全部 case 并与上次比对
  python regress.py --case 0        # 只跑第 1 个 case
  python regress.py --save-baseline # 把本次结果存为新基线（确认改动无误后手动执行）

输出：
  终端表格 + regression_log.jsonl（每行一次完整运行记录）
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import config
from agent import Agent
from context import ResultStore
from rag import Retriever
from tools import Tools

LOG_PATH = config.ROOT / "regression_log.jsonl"

# ---------- 固定 case ----------
# 选 case 的原则：覆盖不同的执行路径，而不是"多"。3 条足够守住主要回归面。
CASES = [
    {
        "name": "npe_full_chain",
        "input": "线上报 UserService 空指针，日志在 logs/app.log，帮忙定位",
        "desc": "完整链路：读日志→检索→给出四段结论（主力 case）",
    },
    {
        "name": "doc_only",
        "input": "查一下 Redis 限流的滑动窗口是怎么实现的",
        "desc": "纯知识检索：不需要读日志，验证 RAG 路径",
    },
    {
        "name": "log_missing",
        "input": "看下 logs/not_exist.log 里的报错",
        "desc": "工具失败路径：日志不存在，验证错误回填与自我纠正",
    },
]

# ---------- 阈值：超过这个比例的变化就报警 ----------
THRESHOLDS = {
    "steps": 0.30,              # 步数变动超过 30%
    "prompt_tokens": 0.25,      # prompt token 变动超过 25%
    "tool_calls_total": 0.30,   # 工具调用次数变动超过 30%
    "tool_calls_failed": 0.50,  # 工具失败次数上涨 50%（失败变多通常意味着在"瞎试"）
    "answer_chars": 0.40,       # 答案长度变动超过 40%（太短通常意味着"证据不足"）
    "latency_ms": 0.60,         # 耗时波动容忍度大一些（网络抖动）
}


@dataclass
class Snapshot:
    name: str
    timestamp: str
    steps: int
    prompt_tokens: int
    completion_tokens: int
    tool_calls_total: int
    tool_calls_failed: int
    latency_ms: int
    degraded: bool
    answer_chars: int
    answer_head: str
    has_evidence: bool          # 答案里是否出现了 [doc= 形式的证据引用
    says_insufficient: bool     # 答案里是否出现了"证据不足"


def _snapshot(case: dict, res) -> Snapshot:
    ans = res.answer or ""
    return Snapshot(
        name=case["name"],
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        steps=res.steps,
        prompt_tokens=res.prompt_tokens,
        completion_tokens=res.completion_tokens,
        tool_calls_total=res.tool_calls_total,
        tool_calls_failed=res.tool_calls_failed,
        latency_ms=res.latency_ms,
        degraded=res.degraded,
        answer_chars=len(ans),
        answer_head=ans.strip().replace("\n", " ")[:80],
        has_evidence="[doc=" in ans,
        says_insufficient="证据不足" in ans,
    )


def _load_last_run() -> dict[str, Snapshot]:
    """取每个 case 最近一次的结果（按 run_id 取最大，不依赖文件顺序）。"""
    if not LOG_PATH.exists():
        return {}
    latest: dict[str, Snapshot] = {}
    best_run = ""
    for line in LOG_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid = obj.get("run_id", "")
        if rid < best_run:
            continue
        best_run = rid
        for snap in obj.get("snapshots", []):
            latest[snap["name"]] = Snapshot(**snap)
    return latest


def _load_baseline() -> dict[str, Snapshot]:
    """基线：标记了 baseline=True 的最近一次运行（按 run_id 取最大）。"""
    if not LOG_PATH.exists():
        return {}
    baseline: dict[str, Snapshot] = {}
    best_run = ""
    for line in LOG_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("baseline") and obj.get("run_id", "") > best_run:
            best_run = obj.get("run_id", "")
            baseline = {s["name"]: Snapshot(**s) for s in obj.get("snapshots", [])}
    return baseline


def _diff(cur: Snapshot, ref: Snapshot) -> list[str]:
    """返回超过阈值的字段说明列表。"""
    warns: list[str] = []
    for field, tol in THRESHOLDS.items():
        a, b = getattr(ref, field), getattr(cur, field)
        if b == 0:
            continue
        ratio = abs(a - b) / b
        if ratio > tol:
            pct = (a - b) / b
            warns.append(f"{field}: {b} → {a} ({pct:+.0%})")
    # 语义级检查：证据引用消失 / 突然说证据不足，比数字更值得报警
    if ref.has_evidence and not cur.has_evidence:
        warns.append("证据引用消失（答案里没有 [doc= 了）")
    if not ref.says_insufficient and cur.says_insufficient:
        warns.append("开始说'证据不足'（通常意味着上下文被挤掉了）")
    if not ref.degraded and cur.degraded:
        warns.append("出现降级（degraded=True）")
    return warns


def run_one(case: dict, retriever: Retriever, show_answer: bool = False) -> Snapshot:
    store = ResultStore(trace_id=f"reg_{case['name']}")
    tools = Tools(retriever, store, user_groups=["dev"])
    agent = Agent(tools)

    if show_answer:
        print(f"\n--- {case['name']}: {case['input']} ---")
        res = agent.run(case["input"], on_text=lambda t: print(t, end="", flush=True))
        print()
    else:
        print(f"  跑 {case['name']} ...", end="", flush=True)
        res = agent.run(case["input"])
        print(f" 完成（{res.steps}步 / {res.prompt_tokens}tk）")

    return _snapshot(case, res)


def main() -> int:
    args = sys.argv[1:]
    only = None
    if "--case" in args:
        only = int(args[args.index("--case") + 1])
    save_baseline = "--save-baseline" in args
    show_answer = "--show" in args

    retriever = Retriever()
    retriever.load()

    cases = [CASES[only]] if only is not None else CASES
    print(f"\n回归测试：{len(cases)} 个 case，模型 {config.LLM_MODEL}\n")

    snaps: list[Snapshot] = []
    for case in cases:
        try:
            snaps.append(run_one(case, retriever, show_answer=show_answer))
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ {case['name']} 运行失败：{type(exc).__name__}: {exc}")
            return 1

    # ---- 与基线比对 ----
    last = _load_last_run()
    baseline = _load_baseline()
    run_id = time.strftime("%Y%m%d_%H%M%S")

    print("\n" + "=" * 78)
    print(f"{'case':<18}{'步数':>6}{'prompt':>8}{'compl':>8}{'工具':>7}{'失败':>6}{'耗时ms':>9}{'答案字':>8}")
    print("-" * 78)
    all_warns: list[str] = []
    for s in snaps:
        print(f"{s.name:<18}{s.steps:>6}{s.prompt_tokens:>8}{s.completion_tokens:>8}"
              f"{s.tool_calls_total:>7}{s.tool_calls_failed:>6}{s.latency_ms:>9}{s.answer_chars:>8}")
        ref = baseline.get(s.name) or last.get(s.name)
        ref_name = "基线" if s.name in baseline else "上次运行"
        warns = _diff(s, ref) if ref else []
        if warns:
            for w in warns:
                all_warns.append(f"[{s.name}] 对比{ref_name}：{w}")
    print("=" * 78)

    if baseline:
        ts = next(iter(baseline.values())).timestamp
        print(f"\n（比对对象：基线 {ts}）")

    if all_warns:
        print(f"\n⚠️  发现 {len(all_warns)} 处变化（仅供参考，不一定是 bug）：")
        for w in all_warns:
            print(f"   · {w}")
        print("\n   如果这些变化是预期的（比如刚做了优化），执行：")
        print("   python regress.py --save-baseline   把它存为新基线")
    elif baseline:
        print("\n✓ 与基线一致，无异常变化。")
    else:
        print("\n（首次运行，无基线可对比。确认结果合理后执行 --save-baseline）")

    # ---- 落盘 ----
    record = {
        "run_id": run_id,
        "baseline": save_baseline,
        "model": config.LLM_MODEL,
        "snapshots": [asdict(s) for s in snaps],
    }
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    if save_baseline:
        print(f"\n✓ 已存为新基线（{run_id}）")
    print(f"\n记录已追加到 {LOG_PATH.name}（共 {sum(1 for _ in LOG_PATH.read_text(encoding='utf-8').splitlines())} 次运行）")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())