"""LLM 客户端封装。

面试考点（这两个坑是"跑过"和"看过"的分界线，代码里都有注释）：

1. 流式下 function.arguments 是增量分片，中途不是合法 JSON，不能边收边 json.loads。
2. 一轮返回多个 tool_call 时，回填给模型的 ToolMessage 顺序必须和 tool_calls 一一对应
   （按 index / tool_call_id 对齐），否则 OpenAI 兼容接口直接 400。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from openai import OpenAI

import config


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str = ""          # 原始字符串，解析失败时保留原文用于回填报错
    parsed: dict | None = None   # 解析成功后的 dict


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    raw: Any = None              # 原始 message，回填 assistant 消息用


def _client() -> OpenAI:
    if not config.DEEPSEEK_API_KEY:
        raise RuntimeError("缺少 DEEPSEEK_API_KEY，请复制 .env.example 为 .env 再填 key")
    return OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.LLM_BASE_URL)


def chat(
    messages: list[dict],
    tools: list[dict] | None = None,
    on_text: Callable[[str], None] | None = None,
    stream: bool = True,
    temperature: float = 0.2,
) -> LLMResponse:
    """一次模型调用。带流式时，文本增量回调 on_text，工具调用在结束后统一解析。

    temperature 设 0.2：排错场景要稳定复现，不需要创造性。
    """
    client = _client()
    last_err: Exception | None = None

    for attempt in range(config.LLM_RETRY + 1):
        started = time.perf_counter()
        try:
            if not stream:
                resp = client.chat.completions.create(
                    model=config.LLM_MODEL,
                    messages=messages,
                    tools=tools or None,
                    temperature=temperature,
                )
                msg = resp.choices[0].message
                usage = resp.usage
                return LLMResponse(
                    content=msg.content or "",
                    tool_calls=_parse_tool_calls(msg.tool_calls or []),
                    finish_reason=resp.choices[0].finish_reason or "",
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    raw=msg.model_dump(exclude_none=True),
                )

            # ---------- 流式 ----------
            content_parts: list[str] = []
            # index -> {"id","name","arguments"}，注意按 index 聚合而不是按顺序追加
            tc_buffer: dict[int, dict] = {}
            finish_reason = ""
            usage_info = None

            stream_resp = client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=messages,
                tools=tools or None,
                temperature=temperature,
                stream=True,
                # 关键：流式默认不返回 usage，必须显式声明，否则 token 统计永远是 0
                stream_options={"include_usage": True},
            )

            for chunk in stream_resp:
                if getattr(chunk, "usage", None):
                    usage_info = chunk.usage
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta

                if delta and delta.content:
                    content_parts.append(delta.content)
                    if on_text:
                        on_text(delta.content)

                if delta and delta.tool_calls:
                    for tc in delta.tool_calls:
                        slot = tc_buffer.setdefault(
                            tc.index, {"id": "", "name": "", "arguments": ""}
                        )
                        if tc.id:
                            slot["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                slot["name"] += tc.function.name
                            # 分片拼接：这一段是**不合法 JSON**，绝不能在这里 loads
                            if tc.function.arguments:
                                slot["arguments"] += tc.function.arguments

                if choice.finish_reason:
                    finish_reason = choice.finish_reason

            # 流结束，此时 arguments 才是完整合法的 JSON
            tool_calls: list[ToolCall] = []
            for idx in sorted(tc_buffer):
                slot = tc_buffer[idx]
                parsed = None
                if slot["arguments"]:
                    try:
                        parsed = json.loads(slot["arguments"])
                    except json.JSONDecodeError:
                        parsed = None  # 模型输出坏 JSON，交给上层把错误回填给它自我纠正
                tool_calls.append(
                    ToolCall(
                        id=slot["id"] or f"call_{idx}",
                        name=slot["name"],
                        arguments=slot["arguments"],
                        parsed=parsed,
                    )
                )

            return LLMResponse(
                content="".join(content_parts),
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                prompt_tokens=getattr(usage_info, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage_info, "completion_tokens", 0) or 0,
                latency_ms=int((time.perf_counter() - started) * 1000),
                raw=None,
            )

        except Exception as exc:  # noqa: BLE001 —— 网络/限流/超时统一重试
            last_err = exc
            if attempt >= config.LLM_RETRY:
                break
            time.sleep(2 ** attempt)  # 指数退避 1s, 2s

    raise RuntimeError(f"LLM 调用失败（已重试 {config.LLM_RETRY} 次）: {last_err}") from last_err


def _parse_tool_calls(raw_calls: list) -> list[ToolCall]:
    out: list[ToolCall] = []
    for tc in raw_calls:
        args = tc.function.arguments or ""
        parsed = None
        if args:
            try:
                parsed = json.loads(args)
            except json.JSONDecodeError:
                parsed = None
        out.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args, parsed=parsed))
    return out


def build_assistant_message(resp: LLMResponse) -> dict:
    """把模型回复转成可以回填进 messages 的 assistant 消息。

    有 tool_calls 时必须带上，否则下一轮模型看不到自己发起过调用。
    """
    msg: dict = {"role": "assistant", "content": resp.content or ""}
    if resp.tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments},
            }
            for tc in resp.tool_calls
        ]
    return msg


def make_tool_message(call_id: str, content: str) -> dict:
    """工具结果回填。失败时内容里放错误信息，而不是抛异常中断循环。"""
    return {"role": "tool", "tool_call_id": call_id, "content": content}