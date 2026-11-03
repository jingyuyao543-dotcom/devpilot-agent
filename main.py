"""CLI 入口：python main.py "报错日志和问题" """
import sys

import config
from agent import Agent
from context import ResultStore
from rag import Retriever
from tools import Tools


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        print("示例：python main.py \"线上报 UserService 空指针，麻烦看下\"")
        return

    user_input = sys.argv[1]
    retriever = Retriever()
    retriever.load()
    store = ResultStore(trace_id="cli")
    tools = Tools(retriever, store, user_groups=["dev"])
    agent = Agent(tools)

    print(f"\n用户输入：{user_input}\n" + "=" * 60)
    streamed: list[str] = []

    def _on_text(t: str) -> None:
        streamed.append(t)
        # 只打印收敛前的最后一段；中间步骤（"我去查日志…"）不往屏幕上刷
        print(t, end="", flush=True)

    res = agent.run(user_input, on_text=_on_text)

    # 兜底：如果流式没吐出最终结论（模型最后一步没走 on_text、或中途降级），
    # 把 answer 补打出来，避免"有答案但屏幕空白"
    if not "".join(streamed).strip() or res.answer.strip() not in "".join(streamed):
        print("\n" + res.answer)

    print("\n" + "=" * 60)
    print(f"步数={res.steps}  耗时={res.latency_ms}ms  "
          f"tokens={res.prompt_tokens}+{res.completion_tokens}  "
          f"工具={res.tool_calls_total}(失败{res.tool_calls_failed})  "
          f"降级={res.degraded}{'(' + res.degrade_reason + ')' if res.degrade_reason else ''}")


if __name__ == "__main__":
    main()