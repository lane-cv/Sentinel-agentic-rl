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


def _ensure_collection():
    global _collection, _connected
    if _collection is not None:
        return _collection
    try:
        from pymilvus import (
            Collection,
            CollectionSchema,
            DataType,
            FieldSchema,
            connections,
            utility,
        )

        if not _connected:
            uri = config.MILVUS_URI
            connections.connect(alias="default", uri=uri)
            _connected = True

        name = config.MILVUS_COLLECTION
        if utility.has_collection(name):
            _collection = Collection(name)
            _collection.load()
            return _collection

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=config.EMBEDDING_DIM),
            FieldSchema(name="fingerprint", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="alertname", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="suspect_service", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="kind", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="route", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="remediation_detail", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="ts", dtype=DataType.DOUBLE),
        ]
        schema = CollectionSchema(fields, description="AIOps 历史诊断工单")
        _collection = Collection(name, schema)
        _collection.create_index(
            field_name="vector",
            index_params={"index_type": "HNSW", "metric_type": "IP",
                          "params": {"M": 8, "efConstruction": 64}},
        )
        _collection.load()
        log.info("已创建 Milvus 集合 %s (dim=%d)", name, config.EMBEDDING_DIM)
        return _collection
    except Exception as exc:
        log.warning("Milvus 不可用，RAG 降级关闭: %s", exc)
        _collection = None
        return None


def _ticket_text(diag: dict) -> str:
    parts = [
        diag.get("summary", ""),
        diag.get("remediation_detail", ""),
        "\n".join(diag.get("evidence", []) or []),
    ]
    return "\n".join(p for p in parts if p).strip()


def _truncate(s: str, n: int) -> str:
    return s[:n] if s else ""


def store_ticket(report: dict[str, Any]) -> None:
    if not config.RAG_ENABLED:
        return
    diag = report.get("diagnosis")
    if not diag:
        return
    try:
        vec = embed(_ticket_text(diag))
        if vec is None:
            return
        col = _ensure_collection()
        if col is None:
            return
        alert = report.get("alert", {}) or {}
        meta = report.get("meta", {}) or {}
        col.insert([{
            "vector": vec,
            "fingerprint": _truncate(meta.get("fingerprint", ""), 64),
            "alertname": _truncate(str(alert.get("alertname", "")), 256),
            "suspect_service": _truncate(diag.get("suspect_service", ""), 256),
            "kind": _truncate(diag.get("kind", ""), 64),
            "route": _truncate(report.get("route", ""), 64),
            "summary": _truncate(diag.get("summary", ""), 4096),
            "remediation_detail": _truncate(diag.get("remediation_detail", ""), 4096),
            "ts": time.time(),
        }])
        col.flush()
        log.info("已存工单: service=%s kind=%s route=%s",
                 diag.get("suspect_service"), diag.get("kind"), report.get("route"))
    except Exception as exc:
        log.warning("store_ticket 失败（已忽略）: %s", exc)


def search_tickets(query: str, top_k: Optional[int] = None) -> list[dict]:
    if not config.RAG_ENABLED:
        return []
    k = top_k or config.RAG_TOP_K
    try:
        vec = embed(query, is_query=True)
        if vec is None:
            return []
        col = _ensure_collection()
        if col is None:
            return []
        hits = col.search(
            data=[vec],
            anns_field="vector",
            param={"metric_type": "IP", "params": {"ef": 64}},
            limit=k,
            output_fields=["fingerprint", "alertname", "suspect_service", "kind",
                           "route", "summary", "remediation_detail", "ts"],
        )
        out: list[dict] = []
        for hit in hits[0]:
            e = hit.entity
            out.append({
                "score": round(float(hit.distance), 4),
                "alertname": e.get("alertname"),
                "suspect_service": e.get("suspect_service"),
                "kind": e.get("kind"),
                "route": e.get("route"),
                "summary": e.get("summary"),
                "remediation_detail": e.get("remediation_detail"),
                "ts": e.get("ts"),
            })
        return out
    except Exception as exc:
        log.warning("search_tickets 失败（已忽略）: %s", exc)
        return []
