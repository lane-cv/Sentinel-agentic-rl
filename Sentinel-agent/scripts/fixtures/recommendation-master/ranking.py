from __future__ import annotations


def rank_by_score(scored_candidates: list[tuple[str, float]]) -> list[str]:
    ordered = sorted(scored_candidates, key=lambda pair: pair[1])
    return [product_id for product_id, _score in ordered]
