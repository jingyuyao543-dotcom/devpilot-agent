"""工具集。

约定：
- 统一 Tool 描述符：name / description / schema / permission_level / timeout / idempotent
- description 本身参与 prompt：参数约束和示例写进去，能降低参数格式错误率
- 所有工具只读，没有副作用工具 —— 有副作用的操作默认不进工具集
- 工具异常不外抛，统一转成文本回填给模型
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import config
import context as ctx_mod

WORKSPACE = Path(__file__).parent


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., str]
    permission_level: str = "read"     # read | write（write 需人工确认，本 demo 不提供）
    idempotent: bool = True

    def to_openai(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class Tools:
    def __init__(self, retriever, store: ctx_mod.ResultStore, user_groups: list[str] | None = None):
        self.retriever = retriever
        self.store = store
        self.user_groups = user_groups or []
        self._registry: dict[str, ToolSpec] = {}
        self._register_all()

    # ---------------- 工具实现 ----------------

    def search_internal_doc(self, query: str, top_k: int = 5) -> str:
        st: dict = {}
        hits = self.retriever.search(query, top_k=top_k, user_groups=self.user_groups, stats=st)
        if not hits:
            return "检索无结果。可能是知识库中没有相关文档，请换用更具体的关键词（如异常类名、服务名）重试，或直接依据已有证据给出结论。"
        parts = []
        for h in hits:
            parts.append(
                f"[{h['rank']}] doc={h['doc']} | section={h['header_path']} | id={h['chunk_id']}\n"
                f"{h['text']}"
            )
        # 把精排决策附在结果尾部：模型可见（知道排序可信度），trace 也可回溯。
        # 只在失败/拦截时提示，避免正常路径多占 token。
        if st.get("rerank_requested") and not st.get("rerank_applied"):
            parts.append(f"[排序说明] 精排模型对该查询置信度低（top1={st.get('rerank_top1')}），"
                         f"已退回关键词+向量的融合排序，相关性可能不如常规。")
        elif st.get("rerank_error"):
            parts.append(f"[排序说明] 精排服务不可用，本次使用融合排序：{st['rerank_error']}")
        return "\n\n".join(parts)

    def query_git_commit(self, path: str = ".", since: str = "30 days ago", limit: int = 10) -> str:
        cmd = ["git", "log", f"--since={since}", f"--max-count={limit}",
               "--pretty=format:%h | %an | %ad | %s", "--date=short", "--", path]
        try:
            out = subprocess.run(
                cmd, cwd=WORKSPACE, capture_output=True, text=True, timeout=10, encoding="utf-8", errors="ignore"
            )
        except FileNotFoundError:
            return "错误：当前环境没有安装 git，无法查询提交记录。请改用其他证据。"
        except subprocess.TimeoutExpired:
            return "错误：git 查询超时（10s）。请缩小 path 范围后重试。"
        if out.returncode != 0:
            return f"错误：git 返回码 {out.returncode}，stderr={out.stderr.strip()[:300]}"
        return out.stdout.strip() or "该路径最近没有提交记录。"

    def fetch_error_log(self, log_file: str, service: str = "") -> str:
        p = Path(log_file)
        if not p.is_absolute():
            p = WORKSPACE / log_file
        if not p.exists():
            cand = sorted((WORKSPACE / "logs").glob("*.log")) if (WORKSPACE / "logs").exists() else []
            hint = "、".join(c.name for c in cand[:5]) or "无"
            return f"错误：日志文件不存在 {log_file}。可用文件：{hint}"
        text = p.read_text(encoding="utf-8", errors="ignore")
        if service:
            lines = [ln for ln in text.splitlines() if service.lower() in ln.lower()]
            if lines:
                text = "\n".join(lines)
        return text

    def fetch_tool_result(self, handle: str) -> str:
        try:
            return self.store.get(handle)
        except KeyError as exc:
            return f"错误：{exc}"

    # ---------------- 注册 ----------------

    def _register_all(self) -> None:
        self.add(ToolSpec(
            name="SearchInternalDoc",
            description=(
                "检索团队内部文档与知识库（含故障排查手册、配置文件说明、历史工单）。"
                "返回按相关度排序的文档片段，每段带 doc/section/id。"
                "当不确定某个异常、组件或配置项的含义、或需要检索官方排障步骤时调用。"
                "query 请使用**具体关键词**（异常类名、服务名、配置项名），不要用自然语言长句。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索关键词，例如 'NullPointerException Jackson 反序列化'"},
                    "top_k": {"type": "integer", "description": "返回片段数，默认 5，最大 10", "default": 5},
                },
                "required": ["query"],
            },
            fn=self.search_internal_doc,
        ))

        self.add(ToolSpec(
            name="QueryGitCommit",
            description=(
                "查询代码仓库最近的提交记录，用于判断问题是否由近期变更引入。"
                "path 传相对路径（如 'src/service/UserService.java'），不传则为全仓库。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "相对路径，默认 '.' 表示全仓库"},
                    "since": {"type": "string", "description": "起始时间，git 语法，如 '7 days ago'", "default": "30 days ago"},
                    "limit": {"type": "integer", "description": "最多返回条数，默认 10", "default": 10},
                },
                "required": [],
            },
            fn=self.query_git_commit,
        ))

        self.add(ToolSpec(
            name="FetchErrorLog",
            description=(
                "读取原始日志文件内容。当用户提供的错误堆栈不完整、需要更多上下文时调用。"
                "log_file 传相对路径（如 'logs/app.log'），可用 service 过滤只保留包含该关键字的行。"
                "返回内容可能很长，系统会自动压缩，你只会看到关键部分。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "log_file": {"type": "string", "description": "日志文件相对路径"},
                    "service": {"type": "string", "description": "可选，按服务名/关键字过滤行"},
                },
                "required": ["log_file"],
            },
            fn=self.fetch_error_log,
        ))

        self.add(ToolSpec(
            name="FetchToolResult",
            description=(
                "取回某个工具结果被压缩掉的完整原文。"
                "当你在压缩后的结果里看到 [中间省略] 标记、且怀疑关键信息被省略时，用结果中给出的 handle 调用。"
            ),
            parameters={
                "type": "object",
                "properties": {"handle": {"type": "string", "description": "工具结果句柄，形如 FetchErrorLog_a1b2c3d4"}},
                "required": ["handle"],
            },
            fn=self.fetch_tool_result,
        ))

    def add(self, spec: ToolSpec) -> None:
        self._registry[spec.name] = spec

    def schemas(self) -> list[dict]:
        return [t.to_openai() for t in self._registry.values()]

    def names(self) -> list[str]:
        return list(self._registry)

    def call(self, name: str, args: dict) -> str:
        """统一入口：只调用白名单内工具；异常转文本，不抛给循环。"""
        spec = self._registry.get(name)
        if spec is None:
            return f"错误：不存在名为 {name} 的工具。可用工具：{', '.join(self._registry)}。请重新选择。"
        if spec.permission_level == "write":
            return f"错误：工具 {name} 涉及写操作，需要人工确认，已拒绝自动执行。"
        try:
            return spec.fn(**args)
        except TypeError as exc:
            return (
                f"错误：参数不匹配 —— {exc}。"
                f"该工具的 schema 为 {json.dumps(spec.parameters, ensure_ascii=False)}，请修正参数后重试。"
            )
        except Exception as exc:  # noqa: BLE001
            return f"错误：工具 {name} 执行异常：{type(exc).__name__}: {exc}"