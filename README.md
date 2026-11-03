# DevPilot Demo —— D1~D3 骨架

面向上线排查场景的最小 Agent 内核。**这是骨架，不是完整项目**：Java 网关、MQ 异步入库、
Rerank、FastAPI/SSE 端点留到 D6，D1~D3 的目标是"能跑、有数字、能讲清每个设计决策"。

## 1. 装环境（5 分钟）

```powershell
cd D:\Project\deepseek\项目\devpilot-demo
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Python 3.10+ 即可。`chromadb` 装不上不影响跑通：把 `.env` 里 `RAG_BM25_ONLY=true`，
代码会走纯 BM25 降级，除向量检索外的全部功能（含 Agent 循环、压缩、trace）都可验证。
这一点本身在面试里就是加分项——"检索层做了降级，向量库不可用时自动回退 BM25"。

## 2. 配 key

```powershell
copy .env.example .env
# 编辑 .env，填 DEEPSEEK_API_KEY
```

DeepSeek 官方 API 本身就是 OpenAI 兼容协议，配 `LLM_BASE_URL=https://api.deepseek.com` 即可。
**注意：DeepSeek 目前没有 embedding 接口**，所以 embedding 需要另一个兼容 OpenAI 协议的服务
（示例里用硅基流动的 `BAAI/bge-m3`，注册有免费额度）。没有的话就开纯 BM25 模式。

## 3. 三步跑通

```powershell
python make_log.py 3000 --mix        # 生成约 2 万字的大日志（D4 用）
python build_index.py                # 建索引 + 8 条查询自检
python main.py "线上报 UserService 空指针，日志在 logs/app.log，帮忙定位"
```

`main.py` 正常输出长这样（末尾一行是你要记进 numbers.md 的数字）：

```
步数=3  耗时=18420ms  tokens=8210+1420  工具=4(失败0)  降级=False
```

## 4. 目录说明

| 文件 | 作用 | 面试考点 |
| --- | --- | --- |
| `llm.py` | LLM 客户端，流式 + tool_call 分片聚合 | 分片中途非法 JSON；`include_usage` 才能拿到 token |
| `agent.py` | 手写 ReAct 循环 | 三种终止、并发工具、失败回填、死循环检测、两级超时 |
| `context.py` | 工具结果压缩链 + 原文另存 | 五级压缩策略、handle 按需取回 |
| `rag.py` | 切块 + BM25 + 向量 + RRF | 结构切块、RRF 不自造权重、ACL 过滤 |
| `tools.py` | 工具注册与调用 | 工具描述即 prompt、只读白名单、异常转文本 |

## 5. 跑完必须做的事（不然白跑）

1. **打开 `traces/<id>/trace.json`，逐 span 看一遍**。确认 `prompt_tokens` 不是 0
   （是 0 说明 `stream_options` 没生效）；确认 `compaction` 那行的 `orig_chars → new_chars`。
2. **把数字抄进 `numbers.md`**，包括失败那次的数字。D7 全靠它。
3. **故意制造一次失败**：把 `main.py` 里提问改成指向一个不存在的日志文件，
   看模型会不会自我纠正换文件（这就是"工具失败回填"的现场证据，面试要讲这段）。
4. **故意触发死循环**：把 `config.LOOP_WINDOW` 改成 2，看提示注入是否让模型改变策略。

## 6. 已知的取舍（面试主动交代，别说成完美）

- 切块按**字符**数而非 token 数，是为了省掉 tokenizer 依赖；生产应按 token 切。
- `_enforce_budget` 用的是最朴素的"挪掉最早的"，没有做摘要压缩——这是 D6 的活。
- `QueryGitCommit` 在 demo 仓库里查的是本目录的 git，没有真实业务仓库，属演示性质。
- ACL 用文件名后缀 `__groupA,groupB.md` 模拟，生产应从文档元数据/权限系统同步。
- 没有做 prompt injection 的防御（D6 补数据与指令分区 + 输出侧引用校验）。

## 7. 下一步（D4~D7）

- D4：`make_log.py 3000 --mix` 跑一遍，记录压缩前后的字符数，验证关键行（NPE 那行、
  `Caused by` 那行、最终 `traceId`）在压缩后是否还在——**这是你最有说服力的一段讲述素材**。
- D5：写 `eval.py` + 25 条标注，做三组对比（纯向量 / 混合 / +rerank）和 MAX_STEPS 5 vs 8。
- D6：`api.py` 加 FastAPI + SSE；Redis 记忆与 token 额度。
- D7：README 换成正式版（架构图 + 时序图 + 对比表），写应答稿。