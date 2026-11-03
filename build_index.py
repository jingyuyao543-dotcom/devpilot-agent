"""建索引：python build_index.py"""
import sys

import config
from rag import Retriever


def main() -> None:
    r = Retriever()
    r.build(config.CORPUS_DIR)
    # 建完立刻做 8 条查询的自检（D2 当天就要有 recall 手感，别等到 D5）
    probes = [
        "NullPointerException Jackson 反序列化",
        "连接池耗尽 HikariCP timeout",
        "Redis 限流 滑动窗口 Lua",
        "RabbitMQ 消息重复消费 幂等",
        "SSE 流式 nginx buffering",
        "OOM heap dump 排查",
        "跨域 CORS 预检失败",
        "索引 重建 embedding 模型升级",
    ]
    print("\n--- 检索自检 ---")
    for q in probes:
        hits = r.search(q, top_k=3)
        found = "命中" if hits else "无结果"
        top = hits[0] if hits else None
        desc = f"top1={top['doc']}#{top['header_path']}" if top else "-"
        print(f"[{found}] {q} -> {desc}")


if __name__ == "__main__":
    sys.exit(main())