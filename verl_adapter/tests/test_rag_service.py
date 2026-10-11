from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from typing import Any

import pytest

from verl_adapter import rag_service


def test_init_search_fn_actually_calls_the_search_path_once(monkeypatch: pytest.MonkeyPatch):
    calls: list[dict] = []

    async def fake_search(args: dict) -> dict:
        calls.append(args)
        return {"content": []}

    monkeypatch.setattr(rag_service, "_load_search_fn", lambda: fake_search)
    rag_service._init_search_fn()
    assert len(calls) == 1
    assert calls[0].get("query")


def test_init_search_fn_warmup_failure_does_not_block_startup(monkeypatch: pytest.MonkeyPatch):
    async def boom(args: dict) -> dict:
        raise RuntimeError("milvus not up yet")

    monkeypatch.setattr(rag_service, "_load_search_fn", lambda: boom)
    rag_service._init_search_fn()
    assert rag_service._search_fn is boom


@pytest.fixture()
def live_service(monkeypatch: pytest.MonkeyPatch):
    calls: list[dict] = []

    async def fake_search(args: dict) -> dict:
        calls.append(args)
        if args.get("query") == "__boom__":
            raise RuntimeError("milvus down (simulated)")
        return {"content": [{"type": "text", "text": f"hits for {args.get('query', '')!r}"}]}

    monkeypatch.setattr(rag_service, "_search_fn", fake_search)
    server = rag_service.ThreadingHTTPServer(("127.0.0.1", 0), rag_service._RagHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", calls
    server.shutdown()
    thread.join(timeout=5)


def _post(url: str, body: bytes, content_type: str = "application/json") -> tuple[int, dict | str]:
    req = urllib.request.Request(
        f"{url}/search", data=body, headers={"Content-Type": content_type}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def test_search_returns_mcp_shaped_json(live_service):
    url, calls = live_service
    status, body = _post(url, json.dumps({"query": "kafka 积压"}).encode())
    assert status == 200
    assert body == {"content": [{"type": "text", "text": "hits for 'kafka 积压'"}]}
    assert calls == [{"query": "kafka 积压"}]


def test_app_level_failure_is_200_with_is_error_body(live_service):
    url, _ = live_service
    status, body = _post(url, json.dumps({"query": "__boom__"}).encode())
    assert status == 200
    assert body["is_error"] is True
    assert "milvus down" in body["content"][0]["text"]


def test_bad_request_body_is_400(live_service):
    url, _ = live_service
    status, body = _post(url, b"not json at all")
    assert status == 400
    assert body["is_error"] is True


def test_non_object_body_is_400(live_service):
    url, _ = live_service
    status, body = _post(url, json.dumps(["a", "list"]).encode())
    assert status == 400
    assert body["is_error"] is True


def test_healthz(live_service):
    url, _ = live_service
    with urllib.request.urlopen(f"{url}/healthz", timeout=10) as resp:
        assert resp.status == 200
        assert json.loads(resp.read().decode("utf-8")) == {"ok": True}


def test_unknown_path_is_404(live_service):
    url, _ = live_service
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"{url}/nope", timeout=10)
    assert exc_info.value.code == 404


def test_concurrent_requests_serialize_through_the_lock(
    live_service, monkeypatch: pytest.MonkeyPatch
):
    url, _ = live_service
    inside = 0
    max_inside = 0

    async def counting_search(args: dict) -> dict:
        nonlocal inside, max_inside
        inside += 1
        max_inside = max(max_inside, inside)
        import asyncio

        await asyncio.sleep(0.05)
        inside -= 1
        return {"content": [{"type": "text", "text": "ok"}]}

    monkeypatch.setattr(rag_service, "_search_fn", counting_search)
    threads = [
        threading.Thread(target=lambda: _post(url, json.dumps({"query": f"q{i}"}).encode()))
        for i in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert max_inside == 1
