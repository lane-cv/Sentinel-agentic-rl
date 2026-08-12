from __future__ import annotations

import recommendation_server as rs


def test_returns_recommendations():
    out = rs.get_recommendations(["PRODUCT-0"], max_results=3)
    assert isinstance(out, list)
    assert len(out) <= 3
    assert "PRODUCT-0" not in out


def test_memory_is_bounded():
    start = rs.seen_count()
    for i in range(5000):
        rs.get_recommendations([f"PRODUCT-{i % 20}"])
    growth = rs.seen_count() - start
    assert growth < 1000, (
        f"memory appears unbounded: tracking structure grew by {growth} "
        f"entries over 5000 requests (expected a bounded cap)"
    )
