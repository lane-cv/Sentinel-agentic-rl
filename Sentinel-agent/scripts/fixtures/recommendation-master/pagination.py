from __future__ import annotations


def paginate(items: list[str], page: int, page_size: int) -> list[str]:
    start = page * page_size
    return items[start : start + page_size]
