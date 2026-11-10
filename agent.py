"""手写 ReAct 执行循环 —— 本项目最核心的一段，面试很可能让你现场手撕。

包含的设计点（每一条都对应一个追问）：
- MAX_STEPS 上限 + 三段终止（模型收敛 / 步数耗尽 / 死循环）
- 同一轮多 tool_call 并发执行，结果按 tool_call_id 严格对齐
- 工具异常转文本回填，模型自我纠正，而不是抛异常中断
- 参数 hash 滑窗检测死循环
- 两级超时（单工具 / 整任务），整任务超时返回部分结论
- 每步 trace 落盘，供评测与 bad case 归因
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
from dataclasses import dataclass, field

import config
import context as ctx_mod
import llm as llm_mod
from tools import Tools


def _fetch_result_window(text: str) -> str:
    """FetchToolResult 的返回：原文太长时只给「命中窗口」而不是全文。

    理由：这个工具存在的意义是补回被压缩掉的那一小段细节，
    如果把 3 万字原文整段塞回去，等于把压缩白做了，还会把别的证据挤出预算。
    实现上按「异常行/错误行」做锚点，取锚点前后 FETCH_RESULT_WINDOW 字符。
    """
    if len(text) <= config.FETCH_RESULT_MAX_CHARS:
        return text
    anchor = -1
    for pat in ("Caused by", "Exception", "ERROR", "Error", "error"):
        anchor = text.rfind(pat)
        if anchor != -1:
            break
    if anchor == -1:
        anchor = len(text) // 2
    half = config.FETCH_RESULT_WINDOW // 2
    start = max(0, anchor - half)
    end = min(len(text), anchor + half)
    body = text[start:end]
    return (f"[原文共 {len(text)} 字，此处仅返回关键片段（第 {start}~{end} 字），"
            f"已定位到最新一处异常]\n...\n{body}\n...")


SYSTEM_PROMPT = """你是 DevPilot，一名资深的后端故障排查专家。

工作方式：
1. 先判断信息是否充分。信息不足时，用工具取证（检索文档 / 查提交 / 读日志），不要凭经验猜测。
2. 每次只发起必要的工具调用；多个彼此独立的查询可以一次并发发起。
3. 工具返回错误时，读懂错误原因再调整参数重试，不要原样重复同一调用。
4. 证据足够后停止调用工具，直接给出结论。

关于"这个问题不用查"（重要）：
- 判断"这是咨询而不是排障"只是你的推断，不是事实。只要问题涉及本系统的
  组件或机制（Redis / RabbitMQ / HikariCP / JVM / OOM / SSE / 限流 / 检索 /
  会话记忆 / 配置项），必须先调用一次 SearchInternalDoc 再决定用不用它的结果。
- 未调用任何检索工具就直接下结论、或者直接反问用户，视为排查未完成。
- 只有检索结果为空或明显不相关时，才允许写"证据不足"；此时要说明
  **你查了什么、返回了什么**（例如"检索 'heap dump 工具' 返回的前 5 条与问题无关"）。

输出要求（严格遵守）：
- 结构固定为四段：**现象** / **根因** / **证据** / **修复建议**
- "证据"必须逐条列出所依据的文档 id 或提交 hash，格式如 `[doc=xxx id=abc123]`；
  没有证据支撑的技术断言不要写，宁可写"当前证据不足，建议补充 XX 信息"。
- 不要输出与排查无关的客套话。"""


@dataclass
class RunResult:
    answer: str
    steps: int = 0
    trace: list[dict] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    degraded: bool = False
    degrade_reason: str = ""
    tool_calls_total: int = 0
    tool_calls_failed: int = 0
    duplicate_calls: int = 0      # 被去重拦截的重复调用次数


def _args_hash(name: str, args: dict) -> str:
    try:
        norm = json.dumps(args, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        norm = str(args)
    return hashlib.md5(f"{name}|{norm}".encode()).hexdigest()[:12]


class Agent:
    def __init__(self, tools: Tools, max_steps: int = config.MAX_STEPS):
        self.tools = tools
        self.max_steps = max_steps

    # ---------------- 主循环 ----------------
    def run(self, user_input: str, history: list[dict] | None = None,
            on_text=None, trace_id: str | None = None) -> RunResult:
        trace_id = trace_id or uuid.uuid4().hex[:12]
        started = time.perf_counter()
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history or [])
        messages.append({"role": "user", "content": user_input})

        result = RunResult(answer="", trace=[])
        recent_hashes: list[str] = []
        seen_calls: dict[str, str] = {}   # (tool|args_hash) -> 已经回填过的结果摘要
        tail_text = ""   # 最后一步模型已经吐出的文本，超时/异常时作为部分结论

        for step in range(1, self.max_steps + 1):
            result.steps = step
            if (time.perf_counter() - started) * 1000 > config.TASK_TIMEOUT * 1000:
                result.degraded, result.degrade_reason = True, "task_timeout"
                result.answer = tail_text or "任务超时，已返回已获取的证据，未能给出完整结论。"
                break

            # 只在最后一步或不需要工具时才流式输出给用户；中间步骤的思考不流出去
            step_text_parts: list[str] = []
            try:
                resp = llm_mod.chat(
                    messages,
                    tools=self.tools.schemas(),
                    on_text=step_text_parts.append if on_text else None,
                    stream=True,
                )
            except Exception as exc:  # noqa: BLE001
                result.degraded, result.degrade_reason = True, f"llm_error:{exc}"
                result.answer = tail_text or f"模型调用失败：{exc}"
                break

            result.prompt_tokens += resp.prompt_tokens
            result.completion_tokens += resp.completion_tokens
            step_text = "".join(step_text_parts) or resp.content
            if step_text:
                tail_text = step_text

            # ---- 终止条件 1：模型不再发起工具调用 = 收敛 ----
            if not resp.tool_calls:
                result.answer = resp.content or step_text
                result.trace.append({
                    "step": step, "type": "final", "latency_ms": resp.latency_ms,
                    "prompt_tokens": resp.prompt_tokens, "completion_tokens": resp.completion_tokens,
                    "finish_reason": resp.finish_reason,
                })
                break

            # ---- 死循环检测：滑窗内相同 tool call 重复过多次 ----
            hashes = [_args_hash(tc.name, tc.parsed or {}) for tc in resp.tool_calls]
            repeated = False
            for h in hashes:
                if recent_hashes and recent_hashes[-1] == h:
                    recent_hashes.append(h)
                else:
                    recent_hashes = [h]
                if len(recent_hashes) >= config.LOOP_WINDOW:
                    repeated = True
                    break

            messages.append(llm_mod.build_assistant_message(resp))

            if repeated:
                # 终止条件 2：不是硬失败，先注入提示要求改变策略，给模型一次机会
                self._append_tool_msgs(
                    messages, resp, [("你已连续多次发起完全相同的工具调用且未获得新信息。"
                                      "请停止重复，基于现有证据直接给出结论；若证据不足，请说明还缺什么。")] * len(resp.tool_calls),
                    result, step, resp,
                )
                recent_hashes = []
                if step == self.max_steps:
                    result.degraded, result.degrade_reason = True, "loop_detected"
                continue

            # ---- 工具执行（并发 + 单工具超时 + 失败回填） ----
            tool_msgs = self._execute_tools(resp.tool_calls, seen_calls, result)
            self._append_tool_msgs(messages, resp, [m[1] for m in tool_msgs], result, step, resp,
                                   meta=[m[2] for m in tool_msgs])

            # 上下文预算控制：工具结果总量超预算时，把较早的轮次压成更短的占位
            self._enforce_budget(messages)

        else:
            # 终止条件 3：步数耗尽
            result.degraded, result.degrade_reason = True, "max_steps"
            result.answer = tail_text or "已达到最大推理步数，未能收敛。以下是基于现有证据的初步判断，建议人工复核。"

        result.latency_ms = int((time.perf_counter() - started) * 1000)
        _check_grounding(result)
        _dump_trace(trace_id, user_input, result)
        return result

    # ---------------- 工具执行 ----------------
    def _execute_tools(self, tool_calls, seen_calls: dict[str, str] | None = None,
                       result: RunResult | None = None) -> list[tuple]:
        """执行工具。同一 (工具名, 参数) 只真正执行一次，第二次起直接回填提示。

        为什么要这个硬约束：实测发现模型会「换个措辞再问一遍」同一件事
        （比如 SearchInternalDoc 连调 3 次，query 只差一两个字），
        每次都要付一整轮的 prompt token，但没有任何新信息。
        纯靠 prompt 说「不要重复调用」没有用 —— 模型会把它理解成建议而不是规则。
        """
        seen = seen_calls if seen_calls is not None else {}

        def one(tc) -> tuple:
            t0 = time.perf_counter()
            if tc.parsed is None:
                return (tc.id, f"错误：你输出的 arguments 不是合法 JSON（原文：{tc.arguments[:200]}）。"
                               f"请重新以严格 JSON 格式调用该工具。", {"error": "bad_json"})

            # ---- 去重：完全相同的调用只执行一次 ----
            key = f"{tc.name}|{_args_hash(tc.name, tc.parsed)}"
            if key in seen:
                if result is not None:
                    result.duplicate_calls += 1
                return (tc.id,
                        f"[跳过] 你用完全相同的参数已经调用过 {tc.name}，上面已有结果，这里不再重复执行。"
                        f"要拿新信息请换参数（比如换关键词）或换工具；若已够用，请直接给出结论。",
                        {"duplicate": True, "tool": tc.name, "args": tc.parsed,
                         "latency_ms": 0})

            content = self.tools.call(tc.name, tc.parsed)
            handle = None
            # FetchToolResult 的语义就是「取原文」，绝不能对它再跑一次压缩链 ——
            # 否则取回来还是同一份摘要，这一轮工具调用纯属浪费（早期版本的真实 bug）。
            no_compact = tc.name == "FetchToolResult"
            if not content.startswith("错误"):
                if no_compact:
                    content = _fetch_result_window(content)
                    meta = {"compaction": "none", "orig_chars": len(content),
                            "new_chars": len(content), "handle": None}
                else:
                    handle = self.tools.store.put(tc.name, content)
                    comp = ctx_mod.compact_tool_result(content, query=json.dumps(tc.parsed, ensure_ascii=False))
                    content = comp["content"] + f"\n\n[完整原文 handle={handle}，需要时用 FetchToolResult 取回]"
                    meta = {"compaction": comp["method"], "orig_chars": comp["original_chars"],
                            "new_chars": comp["compacted_chars"], "handle": handle}
            else:
                meta = {"error": content[:200]}
            meta.update({"tool": tc.name, "args": tc.parsed,
                         "latency_ms": int((time.perf_counter() - t0) * 1000)})
            # 只有真正执行过（不管成功还是报错）的调用才登记，避免把「跳过」也当成一次调用
            seen[key] = f"{tc.name} 已调用，结果已在上文"
            return (tc.id, content, meta)

        out: list[tuple] = []
        with ThreadPoolExecutor(max_workers=config.MAX_PARALLEL_TOOLS) as pool:
            futures = [(tc, pool.submit(one, tc)) for tc in tool_calls]
            for tc, fut in futures:
                try:
                    out.append(fut.result(timeout=config.PER_TOOL_TIMEOUT))
                except FutTimeout:
                    out.append((tc.id, f"错误：工具 {tc.name} 执行超过 {config.PER_TOOL_TIMEOUT}s 超时。"
                                       f"请缩小查询范围后重试，或改用其他工具。",
                                {"error": "timeout", "tool": tc.name}))
                except Exception as exc:  # noqa: BLE001
                    out.append((tc.id, f"错误：工具 {tc.name} 异常 {type(exc).__name__}: {exc}",
                                {"error": str(exc), "tool": tc.name}))
        return out

    def _append_tool_msgs(self, messages, resp, contents, result: RunResult, step: int, resp_meta,
                          meta: list[dict] | None = None) -> None:
        """按 tool_call_id 严格对齐回填。顺序错会直接 400。"""
        meta = meta or [{} for _ in contents]
        for tc, content, m in zip(resp.tool_calls, contents, meta):
            messages.append(llm_mod.make_tool_message(tc.id, content))
            result.tool_calls_total += 1
            if m.get("error"):
                result.tool_calls_failed += 1
        # 失败/压缩过程打到 stderr，方便演示时讲"自我纠错"和"上下文压缩"
        for tc, m in zip(resp.tool_calls, meta):
            if m.get("duplicate"):
                print(f"  [step {step}] ⊘ {tc.name} 重复调用已跳过", file=sys.stderr, flush=True)
            elif m.get("error"):
                print(f"  [step {step}] ✗ {tc.name} 失败: {m['error']}", file=sys.stderr, flush=True)
            elif m.get("compaction") and m["compaction"] != "none":
                print(f"  [step {step}] ↓ {tc.name} 压缩 {m['orig_chars']}→{m['new_chars']} 字 "
                      f"({m['compaction']})", file=sys.stderr, flush=True)
            else:
                print(f"  [step {step}] ✓ {tc.name} ({m.get('latency_ms', 0)}ms)", file=sys.stderr, flush=True)
        result.trace.append({
            "step": step, "type": "tool_calls", "latency_ms": resp_meta.latency_ms,
            "prompt_tokens": resp_meta.prompt_tokens, "completion_tokens": resp_meta.completion_tokens,
            "calls": [{"tool": tc.name, **m} for tc, m in zip(resp.tool_calls, meta)],
        })

    @staticmethod
    def _enforce_budget(messages: list[dict]) -> None:
        """工具结果总字符超预算时，按「保留价值」淘汰，而不是简单地从最早的开始扔。

        排错场景里第 1 步的日志是所有后续推理的锚点，扔了它后面的检索全部失去意义，
        所以淘汰顺序按以下优先级从低到高：
          1. 已经过期占位的（跳过）
          2. 多余的 FetchToolResult 原文（同一份原文往往取了不止一次，最该先扔）
          3. 较早的其它工具结果
          4. 最新的那一条永远不扔（模型马上要用）
        """
        expired = ("[已过期] 该工具结果因上下文预算被移出。"
                   "如果结论依赖它，请重新调用对应工具（FetchErrorLog / SearchInternalDoc / QueryGitCommit）取证，"
                   "不要基于'看不到内容'直接下结论。")

        def idx_of() -> list[int]:
            return [i for i, m in enumerate(messages) if m.get("role") == "tool"]

        tool_idx = idx_of()
        total = sum(len(messages[i]["content"]) for i in tool_idx)
        if total <= config.BUDGET_TOOL_RESULTS:
            return

        # 候选淘汰顺序：先 FetchToolResult 原文（取回的大块），再从早到晚
        def evict_order() -> list[int]:
            live = [i for i in idx_of() if not messages[i]["content"].startswith("[已过期]")]
            if len(live) <= 1:
                return []
            protect = live[-1]                      # 最新一条不淘汰
            rest = [i for i in live if i != protect]
            # FetchToolResult 的调用参数里带 handle，命中则优先淘汰
            rest.sort(key=lambda i: (
                0 if "FetchToolResult" in messages[i - 1].get("content", "") else 1,
                i,
            ))
            return rest

        for i in evict_order():
            if total <= config.BUDGET_TOOL_RESULTS:
                break
            old = messages[i]["content"]
            if old.startswith("[已过期]"):
                continue
            messages[i]["content"] = expired
            total -= len(old) - len(expired)


def _check_grounding(result: RunResult) -> None:
    """答案里没有任何证据引用时，标记为 degraded（可溯源校验）。

    为什么不拦截答案：有些问题（闲聊、澄清）本来就不需要证据，硬拦截会误伤。
    打标记是更工程化的做法 —— 上层拿到 degraded=True 就知道"这段结论没有出处"，
    可以决定是提示用户「建议人工复核」还是走人工审批。

    实测触发场景：doc_only 有一次只用了 1191 个 prompt token、0 次工具调用就直接作答，
    答案本身读起来通顺，但完全来自模型参数记忆，属于潜在的幻觉。
    回归脚本里「证据引用消失」这条硬判断就是为了抓这种情况。

    豁免规则：显式声明「证据不足 / 信息缺失」的答案不算幻觉，不该打标。
    否则"日志文件不存在"这种合理结论会被误判成降级。

    ⚠️ 但豁免本身留了一个洞（实测发现）：
    模型可以在**一次工具都不调**的情况下直接写"证据不足"，然后凭参数记忆作答。
    这句话完全合法地绕过了上面所有判断，而 25 条评测里这种"0 工具调用"出现了 9 次，
    其中 8 条语料里明明有答案、检索 top1 就是正确段落。
    所以先判 no_tool_call，再判 no_citation —— 逃避路径必须单独关掉。
    """
    if result.degraded:
        return
    # 0 次工具调用 = 根本没有取证，无论答案写了什么都是未完成排查
    if result.tool_calls_total == 0:
        result.degraded = True
        result.degrade_reason = "no_tool_call"
        return
    ans = result.answer
    if "[doc=" in ans or "commit=" in ans.lower():
        return
    # 查过了、检索结果确实没有证据，并主动承认 = 诚实，不是幻觉
    if "证据不足" in ans or "信息缺失" in ans or "无法确定" in ans:
        return
    result.degraded = True
    result.degrade_reason = "no_citation"


def _dump_trace(trace_id: str, user_input: str, result: RunResult) -> None:
    d = config.TRACE_DIR / trace_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "trace.json").write_text(json.dumps({
        "trace_id": trace_id,
        "input": user_input,
        "answer": result.answer,
        "steps": result.steps,
        "degraded": result.degraded,
        "degrade_reason": result.degrade_reason,
        "latency_ms": result.latency_ms,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "tool_calls_total": result.tool_calls_total,
        "tool_calls_failed": result.tool_calls_failed,
        "duplicate_calls": result.duplicate_calls,
        "spans": result.trace,
    }, ensure_ascii=False, indent=2), encoding="utf-8")