"""检索层：切块 + BM25 + 向量 + RRF 融合 + rerank 精排。

几个实现上的取舍：
- 切块按标题层级而不是定长 512 token，块前缀拼 header_path。
- 混合检索用 RRF 融合排名，不对两路分数加权求和（量纲不可比）。
- metadata ACL 过滤在检索层做，不依赖 prompt 约束模型。
- 没有 embedding key 时降级为纯 BM25，保证代码在没有向量服务时也能跑。
- rerank 带阈值闸门：reranker 没把握时分数整体塌到 0.01 量级却照样给顺序，
  直接采信会用噪声排序覆盖本来正确的 RRF 结果（评测里坏了 4 条）。
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

import config

# search() 的形参也叫 rerank，会遮蔽模块级函数，所以这里起个别名再引用
rerank_query = None   # 在 rerank() 定义之后绑定（见文件下方）

HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    chunk_id: str
    doc: str
    header_path: str
    text: str
    allowed_groups: list[str] = field(default_factory=list)  # ACL：空表示公开

    def embedding_text(self) -> str:
        """拼 header_path 前缀再 embed，改善"这段属于哪一节"的语义缺失。"""
        return f"{self.doc} > {self.header_path}\n{self.text}" if self.header_path else f"{self.doc}\n{self.text}"


# ---------------- 切块 ----------------

def chunk_markdown(path: Path, allowed_groups: list[str] | None = None) -> list[Chunk]:
    """按 Markdown 标题层级递归切，超长小节再按段落滑窗切并保留 overlap。"""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    doc_name = path.stem
    chunks: list[Chunk] = []

    stack: list[tuple[int, str]] = []   # (level, title)
    buf: list[str] = []

    def flush() -> None:
        text = "\n".join(buf).strip()
        if not text:
            return
        header_path = " > ".join(t for _, t in stack)
        for piece in _split_long(text):
            cid = hashlib.md5(f"{doc_name}|{header_path}|{piece[:80]}".encode()).hexdigest()[:12]
            chunks.append(Chunk(cid, doc_name, header_path, piece, allowed_groups or []))
        buf.clear()

    for line in lines:
        m = HEADER_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, m.group(2).strip()))
        else:
            buf.append(line)
    flush()
    return chunks


def _split_long(text: str) -> list[str]:
    """超长段落按段落边界滑窗切，overlap 由 config 控制。"""
    size, overlap = config.CHUNK_SIZE, config.CHUNK_OVERLAP
    if len(text) <= size:
        return [text]
    out: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind("\n", start + size // 2, end)  # 尽量在段落边界断开
            if cut > 0:
                end = cut
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [p for p in out if p]


# ---------------- 索引构建 ----------------

class Retriever:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self._bm25 = None
        self._tokenized: list[list[str]] = []
        self._vectors = None          # numpy float32 [n, dim]，L2 归一化后存
        self._vec_dim = 0
        self.embedding_enabled = False

    # ---- 建索引 ----
    def build(self, corpus_dir: Path) -> None:
        from rank_bm25 import BM25Okapi

        self.chunks = []
        for p in sorted(corpus_dir.glob("**/*.md")):
            groups = _groups_from_filename(p)
            self.chunks.extend(chunk_markdown(p, groups))
        if not self.chunks:
            raise RuntimeError(f"{corpus_dir} 下没有 .md 文档，先放几篇真实文档进去")

        self._tokenized = [tokenize(c.embedding_text()) for c in self.chunks]
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None

        if not config.RAG_BM25_ONLY and config.EMBEDDING_API_KEY:
            try:
                self._build_vector()
                self.embedding_enabled = True
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] 向量索引构建失败，降级为纯 BM25：{exc}")
        else:
            print("[info] 未配置 EMBEDDING_API_KEY 或 RAG_BM25_ONLY=true，走纯 BM25。")

        meta = {
            "chunks": len(self.chunks),
            "embedding_enabled": self.embedding_enabled,
            "embedding_model": config.EMBEDDING_MODEL if self.embedding_enabled else None,
        }
        (config.INDEX_DIR / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (config.INDEX_DIR / "chunks.json").write_text(
            json.dumps([asdict(c) for c in self.chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[ok] 建索引完成：{len(self.chunks)} 个 chunk，向量={self.embedding_enabled}")

    def _build_vector(self) -> None:
        """调 embedding API 拿到全部 chunk 的向量，存成本地 .npy。

        为什么不用 chromadb：24 个 chunk 用 ANN 是杀鸡用牛刀，暴力余弦 3ms 就能算完，
        而且能避开 chromadb 版本间 embedding function 参数名不一致的坑。
        向量库真正需要 ANN 的临界点大概在十万级以上。
        """
        import numpy as np

        vecs = embed_texts([c.embedding_text() for c in self.chunks])
        self._vectors = _l2_normalize(np.asarray(vecs, dtype=np.float32))
        self._vec_dim = self._vectors.shape[1]
        np.save(config.INDEX_DIR / "vectors.npy", self._vectors)
        (config.INDEX_DIR / "vector_ids.json").write_text(
            json.dumps([c.chunk_id for c in self.chunks], ensure_ascii=False), encoding="utf-8"
        )

    # ---- 加载已有索引（CLI/评测脚本复用，避免重复调 embedding） ----
    def load(self) -> None:
        from rank_bm25 import BM25Okapi

        chunks_file = config.INDEX_DIR / "chunks.json"
        if not chunks_file.exists():
            raise RuntimeError("索引不存在，先跑 python build_index.py")
        self.chunks = [Chunk(**d) for d in json.loads(chunks_file.read_text(encoding="utf-8"))]
        self._tokenized = [tokenize(c.embedding_text()) for c in self.chunks]
        self._bm25 = BM25Okapi(self._tokenized)

        meta = json.loads((config.INDEX_DIR / "meta.json").read_text(encoding="utf-8"))
        if meta.get("embedding_enabled"):
            vec_file = config.INDEX_DIR / "vectors.npy"
            ids_file = config.INDEX_DIR / "vector_ids.json"
            if not vec_file.exists() or not ids_file.exists():
                raise RuntimeError("向量索引文件缺失，先重新跑 python build_index.py")
            import numpy as np

            self._vectors = np.load(vec_file)
            self._vec_dim = self._vectors.shape[1]
            saved_ids = json.loads(ids_file.read_text(encoding="utf-8"))
            if saved_ids != [c.chunk_id for c in self.chunks]:
                raise RuntimeError(
                    "向量索引与 chunks.json 顺序不一致（语料变了？），重新跑 build_index.py"
                )
            self.embedding_enabled = True

    # ---- 检索 ----
    def search(
        self,
        query: str,
        top_k: int = config.FINAL_TOP_K,
        user_groups: list[str] | None = None,
        mode: str = "hybrid",          # hybrid | bm25 | vector，用于对比实验
        rerank: bool | None = None,    # None = 跟随 config.RERANK_ENABLED
        stats: dict | None = None,     # 传入则回填 rerank 决策与耗时，供评测/调试读取
    ) -> list[dict]:
        user_groups = user_groups or []
        allowed_ids = {
            c.chunk_id for c in self.chunks
            if not c.allowed_groups or set(c.allowed_groups) & set(user_groups)
        }
        if not allowed_ids:
            return []

        bm25_rank: list[str] = []
        if mode in ("hybrid", "bm25") and self._bm25 is not None:
            scores = self._bm25.get_scores(tokenize(query))
            order = sorted(range(len(scores)), key=lambda i: -scores[i])
            bm25_rank = [self.chunks[i].chunk_id for i in order if self.chunks[i].chunk_id in allowed_ids]

        vec_rank: list[str] = []
        if mode in ("hybrid", "vector") and self._vectors is not None:
            order = self._vector_order(query)
            vec_rank = [self.chunks[i].chunk_id for i in order if self.chunks[i].chunk_id in allowed_ids]

        if mode == "bm25":
            fused = bm25_rank
        elif mode == "vector":
            if self._vectors is None:
                # 显式报错而不是返回空列表：空列表会被评测脚本当成"向量检索 0 分"，
                # 得出完全错误的结论。宁可崩，也不要静默出一个假数字。
                raise RuntimeError("请求了 vector 检索但向量索引未启用（无 EMBEDDING_API_KEY 或未建索引）")
            fused = vec_rank
        elif self._vectors is None:
            # hybrid 但向量不可用 → 自动降级为纯 BM25（trace 里会标明）
            fused = bm25_rank
        else:
            fused = self._rrf([r[: config.RETRIEVE_TOP_K] for r in (bm25_rank, vec_rank)])

        by_id = {c.chunk_id: c for c in self.chunks}
        idx_of = {c.chunk_id: i for i, c in enumerate(self.chunks)}

        # ---- rerank 闸门 ----
        # 只精排"融合后的头部候选"，不是全量：精排收益集中在头部，
        # 而延迟和候选数成正比（每个 [query, doc] 都要过一次模型）。
        use_rerank = config.RERANK_ENABLED if rerank is None else rerank
        apply_rerank = False
        rerank_top1 = 0.0
        rerank_ms = 0.0
        if use_rerank and mode in ("hybrid", "vector") and len(fused) > 1:
            cand = fused[: config.RERANK_CANDIDATES]
            try:
                t0 = time.perf_counter()
                scores = rerank_query(query, [by_id[c].text for c in cand])
                rerank_ms = (time.perf_counter() - t0) * 1000
                order = sorted(range(len(cand)), key=lambda i: -scores[i])
                rerank_top1 = scores[order[0]] if order else 0.0
                # 闸门：top1 分数够高才采信。低分说明 reranker 自己也没把握，
                # 它仍会给一个顺序，但那是噪声排序——此时保留 RRF 原序更安全。
                if rerank_top1 >= config.RERANK_MIN_SCORE:
                    fused = [cand[i] for i in order] + fused[len(cand):]
                    apply_rerank = True
            except Exception as e:                      # 网络/限流/超时：降级，不中断检索
                rerank_ms = -1.0
                if stats is not None:
                    stats["rerank_error"] = f"{type(e).__name__}: {e}"
        if stats is not None:
            stats.update({
                "rerank_requested": use_rerank,
                "rerank_applied": apply_rerank,
                "rerank_top1": round(rerank_top1, 4),
                "rerank_ms": round(rerank_ms, 1),
            })

        # 只为展示/调试算一次原始分数：BM25 是稀疏打分（可到 20+），余弦恒在 [-1,1]，
        # 两者不可比——这正是融合用 RRF 的原因，这里的分数绝不参与融合。
        bm25_scores = self._bm25.get_scores(tokenize(query)) if mode in ("hybrid", "bm25") and self._bm25 else None
        q_vec = None
        if mode in ("hybrid", "vector") and self._vectors is not None:
            import numpy as np          # 局部导入：numpy 只在向量路径需要
            q_vec = _l2_normalize(np.asarray(embed_texts([query]), dtype=np.float32))[0]

        out = []
        for rank, cid in enumerate(fused[:top_k]):
            c = by_id[cid]
            item = {
                "chunk_id": c.chunk_id, "doc": c.doc, "header_path": c.header_path,
                "text": c.text, "rank": rank + 1,
            }
            if rank == 0:
                item["scores"] = {}
                if bm25_scores is not None:
                    item["scores"]["bm25"] = round(float(bm25_scores[idx_of[cid]]), 3)
                if q_vec is not None:
                    item["scores"]["cosine"] = round(float(self._vectors[idx_of[cid]] @ q_vec), 4)
            out.append(item)
        return out

    def _vector_order(self, query: str) -> list[int]:
        """暴力余弦检索：向量已 L2 归一化，点积即余弦相似度。返回按分数降序的 chunk 下标。"""
        import numpy as np

        q = _l2_normalize(np.asarray(embed_texts([query]), dtype=np.float32))[0]
        scores = self._vectors @ q
        return list(np.argsort(-scores))

    @staticmethod
    def _rrf(rank_lists: list[list[str]], k: int = config.RRF_K) -> list[str]:
        """Reciprocal Rank Fusion：只用排名，不用原始分数，避免两路分数不可比的问题。"""
        agg: dict[str, float] = {}
        for rl in rank_lists:
            for rank, cid in enumerate(rl, start=1):
                agg[cid] = agg.get(cid, 0.0) + 1.0 / (k + rank)
        return sorted(agg, key=lambda x: -agg[x])


# ---------------- Rerank（cross-encoder 精排） ----------------

def rerank(query: str, docs: list[str], timeout: float | None = None) -> list[float]:
    """对 [query, 每个 doc] 成对打分，返回与 docs 同序的分数列表。

    和 embedding 检索的本质区别：
    - 双塔（bi-encoder）：query 和 doc 各自编码，最后算余弦。快，但两者从没见过面。
    - 交叉编码（cross-encoder）：query 和 doc 拼在一起过模型，能看到"这个词在这句话里"的交互。
      慢（每个候选都要过一次模型），但准得多——实测同一对内容，bge-m3 的余弦差只有 0.08，
      而 reranker 给出 0.93 vs 0.01。

    失败不吞异常：rerank 挂了就当它没开，让上层显式决定降级，别悄悄返回原序假装成功。
    """
    if not docs:
        return []
    if not config.EMBEDDING_API_KEY:
        raise RuntimeError("EMBEDDING_API_KEY 未配置，无法调用 rerank")

    body = {"model": config.RERANK_MODEL, "query": query, "documents": docs}
    req = urllib.request.Request(
        config.EMBEDDING_BASE_URL.rstrip("/") + "/rerank",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {config.EMBEDDING_API_KEY}",
                 "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout or config.RERANK_TIMEOUT) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    # 接口返回的是 [{index, relevance_score}]，可能不带 top_n 时是全量；
    # 按 index 回填，保证与入参 docs 同序 —— 顺序错位会导致"分数配错文档"，静默且致命。
    scores = [0.0] * len(docs)
    for r in data.get("results", []):
        i = int(r["index"])
        if 0 <= i < len(docs):
            scores[i] = float(r["relevance_score"])
    return scores


rerank_query = rerank


# ---------------- 分词 ----------------

_STOP = set("的 了 是 在 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好 自己 这".split())
_jieba = None


def tokenize(text: str) -> list[str]:
    """中文分词，英文/标识符按词切，统一小写。

    三档降级，顺序经过评测校准（别凭直觉改）：jieba > 二元组 > 整句。
    「整句」那一档是最坏的：正则 `[\\u4e00-\\u9fa5]{2,}` 会把一句中文切成一整个 token，
    BM25 全库零命中，所有分都是 0，排序退化成「按文档 id」——不报错，但已经失效了。
    这是实测过的失败路径：未安装 jieba 时语义档 recall@5 只有 0.125，根因就是它。
    """
    global _jieba
    if _jieba is None:
        try:
            import jieba
            jieba.setLogLevel(60)
            _jieba = jieba
        except ImportError:
            _jieba = False
    raw = text                       # 驼峰边界必须先看原文，lower 之后 CamelCase 就没了
    lowered = text.lower()

    if _jieba:
        words = [w.strip() for w in _jieba.lcut(lowered)]
        # jieba 不认标识符，驼峰/下划线拆分单独补一遍
        for ident in re.findall(r"[A-Za-z][A-Za-z0-9_.\-]*", raw):
            words.append(ident.lower())
            words.extend(_split_identifier(ident))
    else:
        # 英文/标识符整词保留；连续中文取字二元组，保证「限流」「超时」这类词能拆出来
        words = []
        for m in re.finditer(r"[a-z_][a-z0-9_.\-]{1,}|[\u4e00-\u9fa5]+", lowered):
            tok = m.group(0)
            if "\u4e00" <= tok[0] <= "\u9fa5":
                bigrams = [tok[i:i + 2] for i in range(len(tok) - 1)]
                words.extend(bigrams if bigrams else [tok])
            else:
                words.append(tok)
        # 驼峰边界要用原文，lower 过的整句拆不出 CamelCase
        for ident in re.findall(r"[A-Za-z][A-Za-z0-9_.\-]*", raw):
            words.extend(_split_identifier(ident))

    return [w for w in words if w and w not in _STOP and not w.isspace()]


def _split_identifier(tok: str) -> list[str]:
    """把 `leak_detection_threshold` / `leakDetectionThreshold` / `a.b.c` 拆成原子词。

    评测里 M5 是唯一一条 BM25 漏召回的题，根因就是这个：
    语料写 `leakDetectionThreshold`，用户查 `LEAK_DETECTION_THRESHOLD`，
    lower 之后是两个不同的 token，查询最重的那一票直接投空，全库 0 分。
    同一个配置项在代码、配置、日志、提问里会有四种拼法，分词层不统一就必挂。

    做法：原词保留（精确匹配仍有价值）+ 追加按 `_ - .` 与驼峰边界拆出的原子词。
    """
    if not re.search(r"[_\-.]|[a-z][A-Z]", tok):
        return []
    parts = [p for p in re.split(r"[_\-.\s]+", tok) if p]
    out: list[str] = []
    for p in parts:
        out.extend(x.lower() for x in re.findall(r"[A-Z]?[a-z]+|\d+|[A-Z]+(?![a-z])", p))
    # 只有拆出 2 个以上原子词时才回填，避免把单词本身重复计入两次
    return out if len(out) > 1 or (out and out[0] != tok) else []


def _embedding_function():
    """已废弃：改用 `embed_texts()` 直接调 API + numpy 暴力检索。

    保留这个函数名只为兼容旧引用，调用会直接报错，避免有人以为它还在生效。
    """
    raise RuntimeError(
        "chromadb 路径已移除，请使用 embed_texts()（直接调 OpenAI 兼容 embedding 接口 + numpy 余弦）"
    )


def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """调 OpenAI 兼容的 /embeddings 接口。兼容硅基流动、OpenAI、智谱等。

    失败不吞异常：向量检索要不要降级是**调用方**的决定，
    这里悄悄返回零向量会让"向量检索结果异常"变成一个极难发现的坑。
    """
    from openai import OpenAI

    if not config.EMBEDDING_API_KEY:
        raise RuntimeError("EMBEDDING_API_KEY 未配置，无法做向量检索")

    client = OpenAI(api_key=config.EMBEDDING_API_KEY, base_url=config.EMBEDDING_BASE_URL)
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=config.EMBEDDING_MODEL, input=batch)
        # 按 index 排序，不依赖返回顺序（部分网关会乱序返回）
        out.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
    return out


def _l2_normalize(arr):
    norm = (arr ** 2).sum(axis=1, keepdims=True) ** 0.5
    return arr / (norm + 1e-12)


def _groups_from_filename(p: Path) -> list[str]:
    """文件名形如 `foo__group-a,group-b.md` 时解析 ACL 组，否则视为公开文档。"""
    m = re.search(r"__(.+)\.md$", p.name)
    if not m:
        return []
    return [g for g in m.group(1).split(",") if g]