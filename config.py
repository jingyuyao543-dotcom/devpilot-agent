"""全局配置。

设计要点：所有可调参数集中在这里，面试时"你参数怎么定的"能一句话指到文件。
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")   # 按文件位置找 .env，不受 cwd 影响

ROOT = Path(__file__).parent
CORPUS_DIR = ROOT / "corpus"
INDEX_DIR = ROOT / ".index"
TRACE_DIR = ROOT / "traces"

# ---------- LLM ----------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")

# ---------- Embedding（DeepSeek 无 embedding，用兼容协议的外部服务） ----------
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
RAG_BM25_ONLY = os.getenv("RAG_BM25_ONLY", "false").lower() == "true"

# ---------- Agent 循环 ----------
MAX_STEPS = 8                 # 5/8/12 三档要自己测，默认 8
MAX_PARALLEL_TOOLS = 3        # 同一轮多个 tool_call 的并发上限
PER_TOOL_TIMEOUT = 15.0       # 单个工具超时（秒）
TASK_TIMEOUT = 90.0           # 整个任务超时（秒），触发后返回部分结论
TOOL_RETRY = 2                # 工具自身瞬时错误的内部重试次数（不含模型自我纠错）
LOOP_WINDOW = 3               # 滑窗内相同 tool call 出现次数达到该值判定死循环
LLM_RETRY = 2                 # 模型调用失败的重试次数（指数退避）

# ---------- 上下文预算 ----------
# 单位说明（面试会问，别答错）：
#   代码里按「字符数」计算，因为中英混排没有免费的 tokenizer；
#   换算是经验值：中文 1 字 ≈ 1 token，英文 4 字符 ≈ 1 token，代码/日志约 3 字符 ≈ 1 token。
#   DeepSeek-chat 上下文 64K token，按最保守的中文 1:1 估算，工具结果给 12000 字 ≈ 12K token，安全。
BUDGET_SYSTEM = 1000
BUDGET_MEMORY = 2000
BUDGET_TOOL_RESULTS = 12000       # 工具结果总字符预算（不是单条）
TOOL_RESULT_MAX_CHARS = 6000      # 单条工具结果进 prompt 的硬上限

# ---------- FetchToolResult 原文召回 ----------
FETCH_RESULT_MAX_CHARS = 20000    # 模型主动要原文时放宽上限（本来就是为了看重细节）
FETCH_RESULT_WINDOW = 2500        # 超长时返回「命中窗口」而非全文，避免把压缩白做

# ---------- 检索 ----------
CHUNK_SIZE = 700              # 字符数（不是 token，简化计算；面试时说清楚）
CHUNK_OVERLAP = 120
RETRIEVE_TOP_K = 50           # 两路各召回多少，进 RRF 融合
FINAL_TOP_K = 5
RRF_K = 60                    # RRF 平滑常数，标准值

# ---------- Rerank（cross-encoder 精排） ----------
# 为什么不直接全用 rerank 的结果：实测它会把 4 道原本对的题排错。
# 原因是 reranker "没把握"时不会报错，照样给一个顺序（分数整体塌到 0.01 量级）。
# 所以加一道闸：只有 top1 分数够高才采信，否则保留 RRF 原序。这是决策，不是调参。
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() == "true"
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_CANDIDATES = 20        # 送进 reranker 的候选数（全语料才 24，取 20 已覆盖绝大多数）
RERANK_MIN_SCORE = 0.10       # top1 低于此分视为"没把握"，退回 RRF 原序
RERANK_TIMEOUT = 8.0          # 精排是同步阻塞在检索路径上，超时就走降级

for _d in (CORPUS_DIR, INDEX_DIR, TRACE_DIR):
    _d.mkdir(parents=True, exist_ok=True)