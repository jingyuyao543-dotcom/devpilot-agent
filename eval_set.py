"""D5 检索评测集。

## 出题原则（这段是所有数字可信度的前提，面试必讲）

1. **从用户视角出题，不从文档标题出题。**
   文档叫「3.1 滑动窗口 + Lua」，用户不会搜"滑动窗口"，
   他会搜"限流在整点附近放过去两倍请求"或者直接贴一句报错日志。
   这种题才有区分度——考的是能不能跨过语义词面差。

2. **按命中难度分三档，分开统计。**
   - `lexical`：问法和文档用词高度重合，BM25 应该轻松命中
   - `semantic`：问法换了说法，字面几乎不重合，靠语义
   - `mixed`：问法里既有代码标识符又有口语描述
   只报总分是没有意义的——真正的结论在"哪一档两种方案拉开了差距"。

3. **标 `gold` 而不是标"唯一正确答案"。**
   一个问题的答案常常横跨多个 chunk，只要命中其中任意一个就算召回。

4. **写清楚容易误召回什么（`confuser`）。**
   这是这套评测集最有价值的部分：它让你能解释"为什么这题错了"，
   而不是只知道"错了"。

## 已知偏差（主动交代，被问出来就扣分）

- 题目是我看了语料写的，**存在「用答案找问题」的偏差**。
  缓解手段：语气一律用用户口吻，不照抄文档标题；并明确标注为 `dev set`。
- 24 个 chunk 的语料太小，recall@5 的理论上限是 1.0（top5 占了 21%），
  所以**看相对差异，不看绝对值**。
- 25 条样本量小，单条差异不显著；只有跨档一致的差距才当结论。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Query:
    qid: str
    query: str
    gold: tuple[str, ...]      # 任意命中其一即算召回
    kind: str                  # lexical | semantic | mixed
    confuser: str = ""         # 容易误召回的 chunk 标题，用于解释失败原因
    note: str = ""


# chunk 用「章节路径」定位，避免硬编码 hash（语料一改 hash 就全变）。
# 这里只写 header_path 的尾部（不含文档名），eval_rag.py 会在全部 chunk 里做唯一匹配并校验。
# 若某个 gold 匹配到 0 个或多个 chunk，评测脚本直接报错——宁可崩，不要静默算错分。

EVAL_SET: tuple[Query, ...] = (
    # ---------------- lexical：问法和文档用词高度重合 ----------------
    Query("L1", "HikariCP 连接池耗尽 Connection is not available 怎么排查",
          ("2. 数据库连接池耗尽 > 2.1 现象",), "lexical",
          confuser="2. HikariCP 连接池",
          note="术语直接对上，BM25 应该稳"),
    Query("L2", "Redis 限流 滑动窗口 Lua 脚本怎么实现",
          ("3. Redis 限流实现 > 3.1 滑动窗口 + Lua",), "lexical",
          note="文档标题就是答案，字面完全重合"),
    Query("L3", "RabbitMQ 消息重复消费 幂等 怎么保证",
          ("4. RabbitMQ 消息可靠消费 > 4.1 重复消费与幂等",), "lexical",
          note="注意文档里 4.2 讲重试和死信，容易抢 4.1 的位置"),
    Query("L4", "JVM OOM heap dump 用什么工具分析",
          ("6. JVM 内存溢出（OOM）排查 > 6.2 步骤",), "lexical",
          confuser="6. JVM 内存溢出（OOM）排查 > 6.1 判断类型",
          note="6.1 也有 OOM 关键词，但讲的是分类不是工具"),
    Query("L5", "换 embedding 模型之后向量索引要全量重建吗",
          ("7. 向量索引与 Embedding 升级 > 7.1 换模型必须全量重建",), "lexical",
          note="问题几乎就是小标题"),
    Query("L6", "RabbitMQ 死信队列 x-death 重试几次进死信",
          ("4. RabbitMQ 消息可靠消费 > 4.2 失败重试与死信",), "lexical",
          note="x-death 是稀有 token，BM25 应该很强"),
    Query("L7", "HikariCP max-lifetime 和 wait_timeout 的关系",
          ("2. HikariCP 连接池",), "lexical",
          confuser="2. 数据库连接池耗尽 > 2.3 排查与修复",
          note="两个 chunk 都提 HikariCP，考的是能否分清「配置」和「故障」"),
    Query("L8", "RabbitMQ 队列声明 durable 配死信 exchange 的写法",
          ("4. RabbitMQ > 4.1 队列声明",), "lexical",
          note="考代码片段定位"),

    # ---------------- semantic：换了说法，字面几乎不重合 ----------------
    Query("S1", "为什么我的限流在整点附近会突然放过两倍的请求",
          ("3. Redis 限流实现 > 3.1 滑动窗口 + Lua",), "semantic",
          note="用户描述的是「固定窗口边界双倍流量」现象，但一个专业词都没用"),
    Query("S2", "接口大面积超时 CPU 却不高，线程数一直涨",
          ("2. 数据库连接池耗尽 > 2.1 现象",), "semantic",
          note="这是连接池耗尽的典型表征，但用户不会说「连接池」"),
    Query("S3", "对话里用户开头说的限制条件，聊到后面模型就忘了",
          ("3. Redis > 3.2 会话记忆存储",), "semantic",
          confuser="3. Redis 限流实现 > 3.2 Token 额度限流",
          note="这题两极分化最明显：BM25 会被「Redis」和「3.2」带偏到限流那节"),
    Query("S4", "前端等了很久一个字都没收到，最后一次性全吐出来",
          ("5. SSE 流式响应中断 > 5.1 常见原因",), "semantic",
          note="描述缓冲（buffering）现象，不含技术名词"),
    Query("S5", "内存里对象越来越多但是又查不出来哪个没释放",
          ("6. JVM 内存溢出（OOM）排查 > 6.2 步骤",), "semantic",
          confuser="2. 数据库连接池耗尽 > 2.2 常见根因",
          note="「泄漏」两个字两边都有，但用户这里说的是内存泄漏"),
    Query("S6", "用户关闭页面以后后台还在跑还一直扣钱",
          ("5. SSE 流式响应中断 > 5.1 常见原因",), "semantic",
          note="对应「客户端断开未取消下游」。**警告：5.1 是一个超大 chunk，包含 4 个子问题**"),
    Query("S7", "怎么在只按次数收费之外控制这个成本",
          ("3. Redis 限流实现 > 3.2 Token 额度限流",), "semantic",
          confuser="3. Redis > 3.1 关键配置",
          note="用户说的是「成本」，文档说的是「token 计费」"),
    Query("S8", "两个搜索分数一个是 20 多一个是 0.8 怎么合并",
          ("8. 混合检索与 RRF 融合 > 8.1 为什么不加权求和",), "semantic",
          note="描述的是「分数不可比」问题，但没说 RRF"),

    # ---------------- mixed：代码标识符 + 口语 ----------------
    Query("M1", "线上一直报 Communications link failure 但数据库明明是好的",
          ("2. HikariCP 连接池", "5. 常见错误码对照"), "mixed",
          note="报了错误码但加了「数据库是好的」这个误导信息"),
    Query("M2", "日志里 ConcurrentModificationException 和 PrefixLogger 一起出现是什么情况",
          (), "mixed",
          note="【对照项】语料里根本没有这个异常。考的是**不相关时能不能不硬凑**，"
               "gold 为空，评测按「不该召回高置信答案」单独统计，不计入 recall"),
    Query("M3", "为什么两边都写了超时结果前端还是拿到半截",
          ("5. SSE 流式响应中断 > 5.1 常见原因", "1. 服务基础配置 > 1.2 超时矩阵"), "mixed",
          note="横跨两个文档，任意一个都算对，但完整答案要两个"),
    Query("M4", "OOM 之后怎么确认是内存泄漏还是确实就是对象太多",
          ("6. JVM 内存溢出（OOM）排查 > 6.2 步骤",), "mixed",
          note="用户把 6.1 的判断和 6.2 的步骤合成一个问法"),
    Query("M5", "LEAK_DETECTION_THRESHOLD 开了但没打印任何东西",
          ("2. 数据库连接池耗尽 > 2.3 排查与修复",), "mixed",
          confuser="2. HikariCP 连接池",
          note="配置文档里有这个参数，故障手册里讲了它怎么用；考「配置在 A、用法在 B」"),
    Query("M6", "413 上传超了但是 client_max_body_size 改过还是没用",
          ("5. 常见错误码对照",), "mixed",
          note="错误码 + 配置项混在一起，很可能被拆到两个 chunk"),
    Query("M7", "429 返回了但是不知道剩余额度还剩多少",
          ("3. Redis 限流实现 > 3.2 Token 额度限流",), "mixed",
          note="文档明确写了「超限时返回 429 并附剩余额度」"),
    Query("M8", "SSE 一直断 是不是网关把连接杀了",
          ("5. SSE 流式响应中断 > 5.1 常见原因",), "mixed",
          note="用户猜了原因（网关），文档里确实有「空闲连接被杀」但不是唯一原因"),
    Query("M9", "多态反序列化 type 字段对不上为什么没报错而是空指针",
          ("1. NullPointerException 排查 > 1.1 典型成因",), "mixed",
          note="对应第 2 条成因 @JsonSubTypes，考「NPE 和 Jackson 的因果链」"),
)

KINDS = ("lexical", "semantic", "mixed")


def stats() -> dict:
    by_kind = {k: 0 for k in KINDS}
    for q in EVAL_SET:
        by_kind[q.kind] = by_kind.get(q.kind, 0) + 1
    return {
        "total": len(EVAL_SET),
        "scored": sum(1 for q in EVAL_SET if q.gold),   # gold 为空的是对照项，不进 recall
        "by_kind": by_kind,
    }


if __name__ == "__main__":
    s = stats()
    print(f"评测集共 {s['total']} 条，其中参与 recall 统计的 {s['scored']} 条")
    for k in KINDS:
        print(f"  {k:9s} {s['by_kind'][k]} 条")
    print("\n对照项（gold 为空，用于检测'不相关时是否硬凑'）：")
    for q in EVAL_SET:
        if not q.gold:
            print(f"  {q.qid}: {q.query}")