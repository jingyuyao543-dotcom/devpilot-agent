"""生成一段接近真实的大日志，用于 D4 的上下文压缩实验。

用法：python make_log.py [行数] [--mix]
默认 3000 行，约 2 万字符；--mix 会混入真实日志里的大段无关噪声
（健康检查、慢 SQL、心跳），用来验证结构化抽取能不能把它们丢掉。
"""
import random
import sys
from pathlib import Path

HEADER = """2026-11-02 03:14:11.871  INFO 1 --- [http-nio-8801-exec-7] c.d.order.OrderController : 下单请求接收 orderId=98231
2026-11-02 03:14:11.902  INFO 1 --- [http-nio-8801-exec-7] c.d.order.OrderService : 开始校验库存 sku=S10023 qty=2
2026-11-02 03:14:12.114 DEBUG 1 --- [http-nio-8801-exec-7] c.d.order.OrderService : 查询商品详情 sku=S10023
2026-11-02 03:14:12.210  INFO 1 --- [http-nio-8801-exec-7] c.d.product.ProductClient : 调用商品中心耗时 96ms
2026-11-02 03:14:13.802 ERROR 1 --- [http-nio-8801-exec-7] c.d.order.OrderService : 订单创建失败，开始回滚
2026-11-02 03:14:13.803 ERROR 1 --- [http-nio-8801-exec-7] c.d.order.OrderService : 异常详情:
java.lang.NullPointerException: Cannot invoke "com.dev.product.dto.SkuDetail.getPrice()" because "skuDetail" is null
\tat com.dev.order.OrderService.calcAmount(OrderService.java:187)
\tat com.dev.order.OrderService.createOrder(OrderService.java:112)
\tat com.dev.order.OrderController.submit(OrderController.java:64)
\tat org.springframework.cglib.proxy.MethodProxy.invoke(MethodProxy.java:218)
\tat org.springframework.aop.framework.CglibAopProxy$CglibMethodInvocation.invokeJoinpoint(CglibAopProxy.java:793)
\tat org.springframework.transaction.interceptor.TransactionInterceptor.invoke(TransactionInterceptor.java:119)
\tat com.dev.common.log.LogAspect.around(LogAspect.java:41)
\tat com.dev.order.OrderController.submit(OrderController.java:58)
Caused by: com.fasterxml.jackson.databind.exc.InvalidFormatException: 
Cannot deserialize value of type `java.util.ArrayList<java.lang.String>` from Object value (token `JsonToken.START_OBJECT`)
 at [Source: (String)"{"code":0,"data":{"price":null,"tags":{"a":1}}}"; line: 1, column: 62]
\tat com.fasterxml.jackson.databind.DeserializationContext.reportInputMismatch(DeserializationContext.java:1745)
\tat com.dev.product.ProductClient.parseDetail(ProductClient.java:233)
\tat com.dev.product.ProductClient.getSku(ProductClient.java:141)
\tat com.dev.order.OrderService.calcAmount(OrderService.java:183)
\tat com.dev.order.OrderService.createOrder(OrderService.java:112)
... 14 more
"""

NOISE = [
    "DEBUG 1 --- [health-check-{}] c.d.common.HealthIndicator : 健康检查通过，组件 {} 状态 UP",
    " INFO 1 --- [schedul-1] c.d.job.CleanJob : 清理过期日志 {} 条，耗时 {}ms",
    "DEBUG 1 --- [http-nio-8801-exec-{}] o.a.catalina.webresources.JarWarResourceSet : 加载资源 /static/app.js",
    " INFO 1 --- [http-nio-8801-exec-{}] c.d.cache.CacheService : 缓存命中 key=product:detail:{}",
    "DEBUG 1 --- [redis-pool-{}] c.d.redis.LettuceConnection : 从连接池获取连接成功",
    " INFO 1 --- [async-{}] c.d.notify.NotifySender : 推送消息到用户 {} 成功",
]

TAILS = [
    "2026-11-02 03:14:14.001  WARN 1 --- [http-nio-8801-exec-7] c.d.common.RetryAdvisor : 触发重试，attempt=2",
    "2026-11-02 03:14:15.442 ERROR 1 --- [http-nio-8801-exec-7] c.d.common.GlobalExceptionHandler : 请求处理异常，返回 500，traceId=7f3a9c2e",
]


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    noise_ratio = 0.85 if "--mix" in sys.argv else 0.35
    random.seed(42)

    out = [HEADER]
    for i in range(n):
        ts = f"2026-11-02 03:{(14 + i // 600) % 60:02d}:{i % 60:02d}.{i % 1000:03d}"
        if random.random() < noise_ratio:
            tpl = random.choice(NOISE)
            body = tpl.format(i % 50, random.randint(1, 999))
        else:
            body = f" INFO 1 --- [http-nio-8801-exec-7] c.d.order.OrderService : 处理订单明细 item={i}"
        out.append(f"{ts} {body}")
    out.extend(TAILS)

    text = "\n".join(out)
    p = Path(__file__).parent / "logs" / "app.log"
    p.parent.mkdir(exist_ok=True)
    p.write_text(text, encoding="utf-8")
    print(f"已生成 {p}：{len(out)} 行，{len(text)} 字")


if __name__ == "__main__":
    main()