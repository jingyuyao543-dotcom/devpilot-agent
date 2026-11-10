"""上下文压缩的单测：不调模型、不花钱、0.1 秒跑完。

为什么必须有这个文件（面试可讲）：
  压缩逻辑是「代码保证」还是「模型这次恰好看懂了」，两回事。
  本项目真实踩过坑：C1 那次正则把 `Caused by:` 之后的描述行当噪声丢了，
  模型因此误判"证据不足"。如果当时有这条单测，回归时会立刻红。

  另一个价值：它把"压缩不能丢什么"这件事写成了可执行的规格，
  以后改正则，跑一下就知道有没有把关键行丢掉。

运行：python -m pytest test_compact.py -v
   或：python test_compact.py            （不装 pytest 也能跑）
"""
from __future__ import annotations

import sys

from context import compact_tool_result

# ---------------- 测试夹具：一段真实形态的 Java 堆栈 ----------------

NOISE = "\n".join(f"2026-11-03 10:00:{i:02d}.123 [http-nio-8080-exec-1] INFO  "
                  f"c.example.service.OrderService - 查询订单 orderNo={i}" for i in range(50))

JAVA_STACK = f"""{NOISE}
2026-11-03 10:05:00.001 [http-nio-8080-exec-8] ERROR c.example.web.GlobalExceptionHandler - 未捕获异常 traceId=7fa9a2ce
java.lang.NullPointerException: null
	at com.example.service.UserService.getUser(UserService.java:88)
	at com.example.web.UserController.query(UserController.java:41)
	at org.springframework.web.method.support.InvocableHandlerMethod.doInvoke(InvocableHandlerMethod.java:207)
	at org.springframework.web.servlet.DispatcherServlet.doDispatch(DispatcherServlet.java:1067)
	at javax.servlet.http.HttpServlet.service(HttpServlet.java:764)
	at org.apache.catalina.core.ApplicationFilterChain.internalDoFilter(ApplicationFilterChain.java:231)
	at org.apache.catalina.core.ApplicationFilterChain.doFilter(ApplicationFilterChain.java:166)
	at org.apache.tomcat.util.net.SocketProcessorBase.run(SocketProcessorBase.java:49)
	at java.lang.Thread.run(Thread.java:834)
Caused by: com.fasterxml.jackson.databind.exc.InvalidFormatException: Cannot deserialize value of type `java.util.ArrayList<java.lang.String>` from Object value (token `JsonToken.START_OBJECT`)
 at [Source: (String)"{{"code":0,"data":{{"price":null,"tags":{{"a":1}}}}}}"; line: 1, column: 62]
	at com.fasterxml.jackson.databind.DeserializationContext.handleUnexpectedToken(DeserializationContext.java:3991)
	at com.fasterxml.jackson.databind.deser.std.CollectionDeserializer.handleNonArray(CollectionDeserializer.java:364)
	at com.example.client.ProductClient.parseDetail(ProductClient.java:233)
	at com.example.service.OrderService.createOrder(OrderService.java:157)
	... 20 more
{NOISE}"""

# 关键断言点：这些字符串一旦被压缩丢掉，模型就会误判
MUST_KEEP = [
    "NullPointerException",
    "Caused by",
    "InvalidFormatException",
    "Cannot deserialize value of type",       # ← C1 事故里被丢掉的续行
    "java.util.ArrayList<java.lang.String>",  # ← 续行里的关键类型
    "line: 1, column: 62",                    # ← 续行里的定位信息
    "ProductClient.parseDetail",              # ← 真正的出错点
    "getUser(UserService.java:88)",
]

MUST_DROP = ["INFO  c.example.service.OrderService - 查询订单"]


# ---------------- 测试 ----------------

def test_short_text_not_compacted():
    """短文本原样返回，不浪费一次结构抽取。"""
    short = "ERROR: connection refused at 10.0.0.1:6379"
    r = compact_tool_result(short, query="redis 连不上", max_chars=6000)
    assert r["method"] == "none"
    assert r["content"] == short
    assert r["compacted_chars"] == r["original_chars"] == len(short)


def test_structure_keeps_exception_and_continuation():
    """核心用例：异常行 + 续行 + 栈帧必须保留（C1 事故的回归）。"""
    r = compact_tool_result(JAVA_STACK, query="UserService 空指针", max_chars=6000)
    assert r["method"] == "structure", f"应走结构抽取，实际走的是 {r['method']}"
    content = r["content"]
    for needle in MUST_KEEP:
        assert needle in content, f"压缩后丢失了关键信息：{needle!r}"
    # 噪声行按「占比」判断，不要求完全消失：
    # 压缩结果里可能包含 head 分支截取的开头片段，那里必然带几条噪声。
    # 真正要保证的是「噪声没有成规模地留在结果里」。
    noise_lines = sum(1 for ln in content.splitlines() if "INFO  c.example.service" in ln)
    total_lines = max(1, len(content.splitlines()))
    assert noise_lines / total_lines < 0.15, \
        f"噪声行占比 {noise_lines}/{total_lines}，压缩没起到作用"


def test_structure_reduces_size():
    """压缩比：完整堆栈至少压到 1/5 以下。"""
    r = compact_tool_result(JAVA_STACK, query="UserService 空指针", max_chars=6000)
    assert r["original_chars"] > 10000, "夹具太短，测不出压缩效果"
    ratio = r["original_chars"] / r["compacted_chars"]
    assert ratio >= 5, f"压缩比只有 {ratio:.1f}x，太低了"


def test_frames_capped_at_20():
    """单条异常链最多保留 20 个栈帧，防止超长栈把预算吃光。

    注意 max_chars 必须小于原文长度，否则 compact_tool_result 会直接短路返回
    （method="none"），根本走不到抽取逻辑 —— 第一版测试就是这么写错的。
    """
    lines = ["java.lang.RuntimeException: boom"]
    lines += [f"\tat com.example.a{i}.b(File{i}.java:{i})" for i in range(100)]
    text = "\n".join(lines)
    assert len(text) > 2500, "夹具太短，压缩不会触发"
    r = compact_tool_result(text, query="boom", max_chars=2500)
    assert r["method"] == "structure", f"应走结构抽取，实际 {r['method']}"
    kept_frames = sum(1 for ln in r["content"].splitlines() if ln.strip().startswith("at "))
    assert kept_frames == 20, f"应保留 20 帧，实际 {kept_frames}"


def test_c_more_marker_kept():
    """`... 20 more` 是截断标记，必须保留，否则模型以为栈就这么多。"""
    r = compact_tool_result(JAVA_STACK, query="空指针", max_chars=6000)
    assert "... 20 more" in r["content"]


def test_no_exception_falls_back():
    """没有任何异常行时，不能返回空串，必须降级到首尾+相关性策略。"""
    text = "普通日志\n" * 3000
    r = compact_tool_result(text, query="随便", max_chars=2000)
    assert r["content"].strip(), "压缩结果不能为空"
    assert r["method"] in ("head_tail_relevance", "structure+head_tail")
    assert r["compacted_chars"] <= 2000 * 1.5, "没压到目标大小附近"


def test_traceback_style_kept():
    """Python traceback 也要能抽出来，不能只认 Java。"""
    py = "\n".join([
        "normal line " * 50,
        "Traceback (most recent call last):",
        '  File "app.py", line 42, in handler',
        "    return fetch(url)",
        "ConnectionError: HTTPConnectionPool(host='api.deepseek.com', port=443): Max retries exceeded",
        "normal line " * 50,
    ])
    r = compact_tool_result(py, query="deepseek 连不上", max_chars=3000)
    assert "Traceback" in r["content"]
    assert "ConnectionError" in r["content"]


def test_idempotent_on_compact_input():
    """已经压过的文本再压一次，不应继续劣化（防止压缩链被重复调用）。"""
    r1 = compact_tool_result(JAVA_STACK, query="空指针", max_chars=6000)
    r2 = compact_tool_result(r1["content"], query="空指针", max_chars=6000)
    # 第二次因为已经小于 max_chars，应该原样返回
    assert r2["method"] == "none"
    assert r2["content"] == r1["content"]


# ---------------- 简易 runner（不依赖 pytest） ----------------

def _run_all() -> int:
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ✓ {name}")
        except AssertionError as exc:
            failed += 1
            print(f"  ✗ {name}\n      {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ✗ {name} 异常 {type(exc).__name__}: {exc}")

    if failed:
        print(f"\n失败 {failed} 个（共 {len(tests)} 个）")
    else:
        print(f"\n全部通过（共 {len(tests)} 个）")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())