"""检索层：切块 + BM25 + 向量 + RRF 融合。

面试要点：
- 切块按结构（标题层级）而不是定长 512 token，块前缀拼 header_path。
- 混合检索用 RRF 融合排名，不给两路分数加权求和（分数不可比）。
- metadata ACL 过滤在检索层做，不靠 prompt 约束模型。
- 没有 embedding key 时自动降级为纯 BM25，保证代码随时能跑。
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import config

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
        self._collection = None
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
        import chromadb

        client = chromadb.PersistentClient(path=str(config.INDEX_DIR / "chroma"))
        try:
            client.delete_collection("kb")
        except Exception:  # noqa: BLE001
            pass
        embed_fn = _embedding_function()
        self._collection = client.get_or_create_collection(
            name="kb", embedding_function=embed_fn, metadata={"hnsw:space": "cosine"}
        )
        B = 64
        for i in range(0, len(self.chunks), B):
            batch = self.chunks[i : i + B]
            self._collection.add(
                ids=[c.chunk_id for c in batch],
                documents=[c.embedding_text() for c in batch],
                metadatas=[
                    {"doc": c.doc, "header_path": c.header_path,
                     "groups": ",".join(c.allowed_groups)}
                    for c in batch
                ],
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
            import chromadb

            client = chromadb.PersistentClient(path=str(config.INDEX_DIR / "chroma"))
            embed_fn = _embedding_function()
            self._collection = client.get_or_create_collection(name="kb", embedding_function=embed_fn)
            self.embedding_enabled = True

    # ---- 检索 ----
    def search(
        self,
        query: str,
        top_k: int = config.FINAL_TOP_K,
        user_groups: list[str] | None = None,
        mode: str = "hybrid",          # hybrid | bm25 | vector，用于 D5 的对比实验
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
        if mode in ("hybrid", "vector") and self._collection is not None:
            res = self._collection.query(query_texts=[query], n_results=min(config.RETRIEVE_TOP_K * 2, max(len(self.chunks), 1)))
            for cid in (res.get("ids") or [[]])[0]:
                if cid in allowed_ids:
                    vec_rank.append(cid)

        if mode == "bm25":
            fused = bm25_rank
        elif mode == "vector":
            fused = vec_rank
        elif self._collection is None:
            # hybrid 但向量库不可用 → 自动降级为纯 BM25
            fused = bm25_rank
        else:
            fused = self._rrf([r[: config.RETRIEVE_TOP_K] for r in (bm25_rank, vec_rank)])

        by_id = {c.chunk_id: c for c in self.chunks}
        out = []
        for rank, cid in enumerate(fused[:top_k]):
            c = by_id[cid]
            out.append({
                "chunk_id": c.chunk_id, "doc": c.doc, "header_path": c.header_path,
                "text": c.text, "rank": rank + 1,
            })
        return out

    @staticmethod
    def _rrf(rank_lists: list[list[str]], k: int = config.RRF_K) -> list[str]:
        """Reciprocal Rank Fusion：只用排名，不用原始分数，避免两路分数不可比的问题。"""
        agg: dict[str, float] = {}
        for rl in rank_lists:
            for rank, cid in enumerate(rl, start=1):
                agg[cid] = agg.get(cid, 0.0) + 1.0 / (k + rank)
        return sorted(agg, key=lambda x: -agg[x])


# ---------------- 分词 ----------------

_STOP = set("的 了 是 在 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好 自己 这".split())
_jieba = None


def tokenize(text: str) -> list[str]:
    """中文用 jieba 分词，英文/标识符按词切，统一小写。串类名要能整体匹配到。"""
    global _jieba
    if _jieba is None:
        try:
            import jieba
            jieba.setLogLevel(60)
            _jieba = jieba
        except ImportError:
            _jieba = False
    lowered = text.lower()
    words: list[str] = []
    if _jieba:
        words = [w.strip() for w in _jieba.lcut(lowered)]
    else:
        words = re.findall(r"[a-z_][a-z0-9_.]{1,}|[\u4e00-\u9fa5]{2,}", lowered)
    return [w for w in words if w and w not in _STOP and not w.isspace()]


def _embedding_function():
    """构造 chromadb 的 OpenAI 兼容 embedding function。

    不同 chromadb 版本这个函数的参数名不一致（api_base / api_base_url / api_base_url），
    这里按签名探测，避免用户升级版本后直接崩在初始化。
    """
    import inspect

    from chromadb.utils import embedding_functions

    fn_cls = embedding_functions.OpenAIEmbeddingFunction
    params = inspect.signature(fn_cls.__init__).parameters
    kwargs = {"api_key": config.EMBEDDING_API_KEY, "model_name": config.EMBEDDING_MODEL}

    if "api_base" in params:
        kwargs["api_base"] = config.EMBEDDING_BASE_URL
    elif "api_base_url" in params:
        kwargs["api_base_url"] = config.EMBEDDING_BASE_URL
    elif "base_url" in params:
        kwargs["base_url"] = config.EMBEDDING_BASE_URL
    else:
        raise RuntimeError(
            f"当前 chromadb 版本的 OpenAIEmbeddingFunction 不支持自定义 base_url，"
            f"可用参数：{list(params)}。请升级 chromadb 或改用 RAG_BM25_ONLY=true。"
        )
    return fn_cls(**kwargs)


def _groups_from_filename(p: Path) -> list[str]:
    """文件名形如 `foo__group-a,group-b.md` 时解析 ACL 组，否则视为公开文档。"""
    m = re.search(r"__(.+)\.md$", p.name)
    if not m:
        return []
    return [g for g in m.group(1).split(",") if g]