from __future__ import annotations


def dedupe_ids(product_ids: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for pid in product_ids:
        key = pid
        if key.lower() not in seen:
            seen.add(key)
            out.append(pid)
    return out
