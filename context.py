"""上下文工程：工具结果进 prompt 前的压缩链。

五级策略：
1. 结构化优先：堆栈/日志用正则抽 Exception + Caused by 链 + 前 N 帧，噪声行丢弃（零成本）
2. 保留首尾：中间省略，因为错误详情和最终堆栈通常在尾部
3. 相关性筛选：按行与 query 的重合度排序取 top（便宜的近似，不引入幻觉）
4. LLM 摘要兜底：未自动触发，只留接口和注释
5. 全局预算：由 agent 层统一控制

原文另存：压缩后的摘要进 prompt，原文写进 traces/ 目录，
需要时模型可以调 FetchToolResult(handle) 取回 —— 避免压缩丢信息变成死路。
"""
from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

import config

EXC_LINE = re.compile(
    r"(Exception|Error|Caused by|at\s+[\w.$]+\(|^\s*\.\.\.\s+\d+\s+more|"
    r"panicked|FATAL|ERROR|WARN|OOM|timeout|refused|Traceback)",
    re.IGNORECASE,
)
FRAME_LINE = re.compile(r"^\s*at\s+[\w.$]+\(|^\s+File \"", re.IGNORECASE)
LOG_TS = re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}")


def compact_tool_result(text: str, query: str, max_chars: int = config.TOOL_RESULT_MAX_CHARS) -> dict:
    """返回 {content, original_chars, compacted_chars, method}。"""
    original_chars = len(text)
    if original_chars <= max_chars:
        return {"content": text, "original_chars": original_chars,
                "compacted_chars": original_chars, "method": "none"}

    # 1. 结构化抽取：异常行 + 异常续行 + 前 20 帧栈 + Caused by 链
    #
    # 注意「异常续行」：Java 堆栈里 `Caused by: com.xxx.InvalidFormatException:` 之后
    # 往往紧跟一行真正的描述（如 "Cannot deserialize value of type ..."），
    # 这一行不匹配任何异常/帧正则，早期版本会被直接丢掉 —— 导致压缩后模型误判"证据不足"。
    lines = text.splitlines()
    keep: list[str] = []
    frame_count = 0
    has_stack = False
    continuation_left = 0          # 异常行之后允许跟随的描述行数
    for ln in lines:
        # 注意：栈帧行也含 "at xxx(...)"，会被 EXC_LINE 命中。
        # 如果直接走异常分支，frame_count 会被每条栈帧重置为 0，
        # 「前 20 帧」的上限就形同虚设（单测 test_frames_capped_at_20 抓到的真 bug）。
        if EXC_LINE.search(ln) and not FRAME_LINE.search(ln):
            keep.append(ln)
            frame_count = 0
            has_stack = True
            continuation_left = 3
        elif continuation_left > 0 and ln.strip() and not FRAME_LINE.search(ln):
            keep.append(ln)          # 续行：异常描述、at [Source: ...] 定位等
            continuation_left -= 1
        elif has_stack and FRAME_LINE.search(ln) and frame_count < 20:
            keep.append(ln)
            frame_count += 1
            continuation_left = 0
        else:
            continuation_left = 0
    structured = "\n".join(keep).strip()
    if structured:
        structured = f"[已按堆栈结构抽取，原 {original_chars} 字]\n{structured}"
        if len(structured) <= max_chars:
            return {"content": structured, "original_chars": original_chars,
                    "compacted_chars": len(structured), "method": "structure"}

    # 2. + 3. 保留首尾 + 相关性筛选（按 query 词重合度给行打分）
    head = text[: max_chars // 4]
    tail = text[-max_chars // 3 :]
    middle_budget = max_chars - len(head) - len(tail) - 200
    if middle_budget > 0:
        q_tokens = set(_tok(query))
        scored = [(len(q_tokens & set(_tok(ln))), i, ln)
                  for i, ln in enumerate(lines) if LOG_TS.match(ln) or len(ln) < 400]
        scored.sort(key=lambda x: (-x[0], x[1]))
        picked = sorted({i for sc, i, _ in scored if sc > 0} or
                        {i for _, i, _ in scored[:20]})
        middle_lines = [lines[i] for i in picked]
        middle = ""
        for ln in middle_lines:
            if len(middle) + len(ln) > middle_budget:
                break
            middle += ln + "\n"
    else:
        middle = ""

    out = (f"{head}\n\n... [中间省略原 {original_chars} 字，"
           f"按关键字相关性保留约 {len(middle)} 字] ...\n{middle}\n\n{tail}")
    method = "head_tail_relevance" if not structured else "structure+head_tail"
    return {"content": out, "original_chars": original_chars,
            "compacted_chars": len(out), "method": method}


def _tok(s: str) -> list[str]:
    import re as _re
    return _re.findall(r"[a-zA-Z_][a-zA-Z0-9_.]{1,}|[\u4e00-\u9fa5]{2,}", s.lower())


class ResultStore:
    """工具结果原文存盘 + handle，模型可按需取回（第 4 节那句加分话术的落地）。"""

    def __init__(self, trace_id: str) -> None:
        self.dir = config.TRACE_DIR / trace_id / "tool_results"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index: dict[str, str] = {}

    def put(self, tool_name: str, text: str) -> str:
        handle = f"{tool_name}_{uuid.uuid4().hex[:8]}"
        p = self.dir / f"{handle}.txt"
        p.write_text(text, encoding="utf-8")
        self.index[handle] = str(p)
        return handle

    def get(self, handle: str) -> str:
        p = self.index.get(handle)
        if not p or not Path(p).exists():
            raise KeyError(f"未知 handle: {handle}")
        return Path(p).read_text(encoding="utf-8")

    def dump_index(self) -> None:
        (self.dir.parent / "tool_results_index.json").write_text(
            json.dumps(self.index, ensure_ascii=False, indent=2), encoding="utf-8"
        )