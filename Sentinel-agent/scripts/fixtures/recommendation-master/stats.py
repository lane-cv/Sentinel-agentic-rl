from __future__ import annotations

_category_hits: dict[str, int] = {}


def record_hit(category: str) -> None:
    current = _category_hits.get(category, 0)
    _category_hits[category] = current + 1


def hit_count(category: str) -> int:
    return _category_hits.get(category, 0)


def total_hits() -> int:
    return sum(_category_hits.values())
