from __future__ import annotations

import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, Optional, Sequence

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import common

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
DEFAULT_SIM_THRESHOLD = 0.92


def canonical_json(alert: dict) -> str:
    return json.dumps(alert, sort_keys=True, ensure_ascii=False)


def alert_md5(alert: dict) -> str:
    return hashlib.md5(canonical_json(alert).encode("utf-8")).hexdigest()


def md5_dedup(alerts: list[dict]) -> tuple[list[dict], list[dict]]:
    seen: set[str] = set()
    kept: list[dict] = []
    dropped: list[dict] = []
    for a in alerts:
        h = alert_md5(a)
        if h in seen:
            dropped.append(a)
        else:
            seen.add(h)
            kept.append(a)
    return kept, dropped


@dataclass
class EmbeddingBackend:
    name: str
    encode: Optional[Callable[[Sequence[str]], object]] = None


_backend_cache: dict[str, EmbeddingBackend] = {}


def _char_ngram_jaccard(a: str, b: str, n: int = 2) -> float:

    def grams(s: str) -> set[str]:
        s = s.strip()
        if len(s) < n:
            return {s} if s else set()
        return {s[i : i + n] for i in range(len(s) - n + 1)}

    ga, gb = grams(a), grams(b)
    if not ga and not gb:
        return 1.0
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def _lexical_similarity(a: str, b: str) -> float:
    return max(_char_ngram_jaccard(a, b), SequenceMatcher(None, a, b).ratio())


def load_embedding_backend(
    model_name: str = DEFAULT_MODEL_NAME, force_fallback: bool = False
) -> EmbeddingBackend:
    cache_key = "__fallback__" if force_fallback else model_name
    cached = _backend_cache.get(cache_key)
    if cached is not None:
        return cached

    if not force_fallback:
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name)

            def _encode(texts: Sequence[str]):
                return model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)

            backend = EmbeddingBackend(name="bge-real", encode=_encode)
            logger.info("语义去重后端：真实 BGE 模型加载成功 (%s)", model_name)
            _backend_cache[cache_key] = backend
            return backend
        except Exception as exc:
            logger.warning(
                "BGE 模型加载失败（%s: %s），降级为词法相似度兜底 —— 仅供离线开发使用，"
                "使用语义去重请安装 sentence-transformers，并确认当前环境可以加载 %s",
                type(exc).__name__,
                exc,
                model_name,
            )

    backend = EmbeddingBackend(name="lexical-fallback", encode=None)
    _backend_cache[cache_key] = backend
    return backend


def semantic_dedup(
    alerts: list[dict],
    threshold: float = DEFAULT_SIM_THRESHOLD,
    text_fn: Callable[[dict], str] = common.alert_text,
    backend: Optional[EmbeddingBackend] = None,
) -> tuple[list[dict], list[dict]]:
    backend = backend or load_embedding_backend()
    texts = [text_fn(a) for a in alerts]

    kept: list[dict] = []
    dropped: list[dict] = []

    if backend.name == "bge-real" and backend.encode is not None:
        import numpy as np

        vectors = np.asarray(backend.encode(texts))
        kept_vec_idx: list[int] = []
        for i, a in enumerate(alerts):
            is_dup = any(float(np.dot(vectors[i], vectors[j])) > threshold for j in kept_vec_idx)
            if is_dup:
                dropped.append(a)
            else:
                kept_vec_idx.append(i)
                kept.append(a)
        return kept, dropped

    kept_texts: list[str] = []
    for a, t in zip(alerts, texts):
        is_dup = any(_lexical_similarity(t, kt) > threshold for kt in kept_texts)
        if is_dup:
            dropped.append(a)
        else:
            kept.append(a)
            kept_texts.append(t)
    return kept, dropped


def dedup_pipeline(
    alerts: list[dict],
    threshold: float = DEFAULT_SIM_THRESHOLD,
    text_fn: Callable[[dict], str] = common.alert_text,
    backend: Optional[EmbeddingBackend] = None,
) -> dict:
    after_md5, dropped_md5 = md5_dedup(alerts)
    backend = backend or load_embedding_backend()
    kept, dropped_semantic = semantic_dedup(after_md5, threshold=threshold, text_fn=text_fn, backend=backend)
    return {
        "kept": kept,
        "dropped_md5_exact": dropped_md5,
        "dropped_semantic": dropped_semantic,
        "backend_used": backend.name,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="对告警做精确去重和语义去重")
    parser.add_argument("--in-dir", required=True, help="输入目录，读取其中的 seed_*.json")
    parser.add_argument("--out-dir", required=True, help="输出目录，写入幸存的 seed_*.json")
    parser.add_argument("--threshold", type=float, default=DEFAULT_SIM_THRESHOLD)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    backend = load_embedding_backend()
    logger.info("embedding backend in this environment: %s", backend.name)

    pairs = common.read_alerts_dir(args.in_dir)
    logger.info("读取 %d 条种子告警自 %s", len(pairs), args.in_dir)

    alerts = [a for _, a in pairs]
    result = dedup_pipeline(alerts, threshold=args.threshold, backend=backend)

    kept_ids = {id(a) for a in result["kept"]}
    kept_pairs = [(fname, a) for fname, a in pairs if id(a) in kept_ids]
    common.write_alerts_dir(kept_pairs, args.out_dir)

    logger.info(
        "去重完成：保留 %d 条，砍掉 MD5 精确重复 %d 条、语义重复 %d 条（后端=%s）-> 写入 %s",
        len(kept_pairs),
        len(result["dropped_md5_exact"]),
        len(result["dropped_semantic"]),
        result["backend_used"],
        args.out_dir,
    )


if __name__ == "__main__":
    main()
