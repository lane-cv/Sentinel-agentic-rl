from __future__ import annotations

import logging
import time
from typing import Any, Optional

from agent import config

log = logging.getLogger("aiops.memory")

_model = None
_collection = None
_connected = False


def _resolve_model_path() -> str:
    import os

    ms_cache = os.path.expanduser(
        os.path.join("~/.cache/modelscope", config.EMBEDDING_MODEL)
    )
    if os.path.isdir(ms_cache):
        return ms_cache
    return config.EMBEDDING_MODEL


def _get_model():
    global _model
    if _model is not None:
        return _model
    try:
        from sentence_transformers import SentenceTransformer

        path = _resolve_model_path()
        log.info("首次使用，正在加载 embedding 模型 %s", path)
        _model = SentenceTransformer(path)
    except Exception as exc:
        log.warning("embedding 模型不可用，RAG 降级关闭: %s", exc)
        _model = None
    return _model


def embed(text: str, *, is_query: bool = False) -> Optional[list[float]]:
    model = _get_model()
    if model is None:
        return None
    try:
        payload = ("为这个句子生成表示以用于检索相关文章：" + text) if is_query else text
        vec = model.encode(payload, normalize_embeddings=True)
        return vec.tolist()
    except Exception as exc:
        log.warning("embed 失败: %s", exc)
        return None
