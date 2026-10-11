#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verl_adapter.rag_bridge import _load_search_fn

DEFAULT_PORT = 18765
_PORT_ENV = "AIOPS_RAG_SERVICE_PORT"

_search_fn: Callable[[dict], Any] | None = None
_search_lock = threading.Lock()


def _init_search_fn() -> None:
    global _search_fn
    _search_fn = _load_search_fn()
    if not callable(_search_fn):
        raise RuntimeError("rag_service: _load_search_fn returned a non-callable")
    try:
        asyncio.run(_search_fn({"query": "服务启动预热探测"}))
    except Exception as exc:
        print(f"[rag_service] warm-up query failed (will retry lazily on first real request): {exc}", flush=True)


class _RagHandler(BaseHTTPRequestHandler):

    def _respond_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/healthz":
            self._respond_json(200, {"ok": True})
        else:
            self._respond_json(404, {"ok": False, "error": f"unknown path {self.path!r}"})

    def do_POST(self) -> None:
        if self.path != "/search":
            self._respond_json(404, {"is_error": True, "content": [{"type": "text", "text": f"unknown path {self.path!r}"}]})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            args = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            if not isinstance(args, dict):
                raise ValueError(f"body must be a JSON object, got {type(args).__name__}")
        except (json.JSONDecodeError, ValueError) as exc:
            self._respond_json(400, {"is_error": True, "content": [{"type": "text", "text": f"bad request body: {exc}"}]})
            return
        if _search_fn is None:
            self._respond_json(500, {"is_error": True, "content": [{"type": "text", "text": "rag_service: search fn not initialized"}]})
            return
        try:
            with _search_lock:
                result = asyncio.run(_search_fn(args))
        except Exception as exc:
            result = {"is_error": True, "content": [{"type": "text", "text": f"rag_service search failed: {type(exc).__name__}: {exc}"}]}
        self._respond_json(200, result)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[rag_service] {self.address_string()} {fmt % args}", flush=True)


def main() -> None:
    port = int(os.environ.get(_PORT_ENV) or DEFAULT_PORT)
    _init_search_fn()
    server = ThreadingHTTPServer(("127.0.0.1", port), _RagHandler)
    print(f"[rag_service] listening on 127.0.0.1:{port} (env {_PORT_ENV} overrides; model warm, ready)", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
