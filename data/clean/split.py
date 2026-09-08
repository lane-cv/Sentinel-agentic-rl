from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Callable

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import common

DEFAULT_HELD_OUT_RATIO = 0.2


def stratify(alerts: list[dict], key_fn: Callable[[dict], str] = common.stratum_key) -> dict[str, list[int]]:
    buckets: dict[str, list[int]] = {}
    for idx, a in enumerate(alerts):
        k = key_fn(a)
        buckets.setdefault(k, []).append(idx)
    return buckets


def stratified_split(
    alerts: list[dict],
    held_out_ratio: float = DEFAULT_HELD_OUT_RATIO,
    seed: int = 42,
    key_fn: Callable[[dict], str] = common.stratum_key,
) -> dict[str, list[dict]]:
    if not 0.0 <= held_out_ratio <= 1.0:
        raise ValueError(f"held_out_ratio must be in [0, 1], got {held_out_ratio}")

    buckets = stratify(alerts, key_fn=key_fn)
    rng = random.Random(seed)

    held_out_idx: list[int] = []
    train_idx: list[int] = []
    for k in sorted(buckets):
        idxs = list(buckets[k])
        rng.shuffle(idxs)
        n_held = round(len(idxs) * held_out_ratio)
        held_out_idx.extend(idxs[:n_held])
        train_idx.extend(idxs[n_held:])

    held_out = [alerts[i] for i in held_out_idx]
    train_pool = [alerts[i] for i in train_idx]

    assert len(held_out) + len(train_pool) == len(alerts), "held_out + train_pool 数量应等于输入总数"
    assert len(set(held_out_idx) | set(train_idx)) == len(alerts), "不应该有下标被漏掉"
    assert len(set(held_out_idx) & set(train_idx)) == 0, "不应该有下标被分到两边"

    return {"held_out": held_out, "train_pool": train_pool}


def strata_report(alerts: list[dict], key_fn: Callable[[dict], str] = common.stratum_key) -> dict[str, int]:
    buckets = stratify(alerts, key_fn=key_fn)
    return {k: len(v) for k, v in sorted(buckets.items())}


def main() -> None:
    import argparse
    import logging

    parser = argparse.ArgumentParser(description="按告警分层切分训练集和留出集")
    parser.add_argument("--in-dir", required=True, help="输入目录，读取其中的 seed_*.json")
    parser.add_argument("--out-dir", required=True, help="输出根目录，会在其下建 train/ 和 held_out/")
    parser.add_argument("--held-out-ratio", type=float, default=DEFAULT_HELD_OUT_RATIO)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    pairs = common.read_alerts_dir(args.in_dir)
    logger.info("读取 %d 条种子告警自 %s", len(pairs), args.in_dir)

    alerts = [a for _, a in pairs]
    logger.info("分层统计：%s", strata_report(alerts))

    result = stratified_split(alerts, held_out_ratio=args.held_out_ratio, seed=args.seed)

    out_dir = Path(args.out_dir)
    for split_name, out_key in (("train", "train_pool"), ("held_out", "held_out")):
        wanted_ids = {id(a) for a in result[out_key]}
        split_pairs = [(fname, a) for fname, a in pairs if id(a) in wanted_ids]
        common.write_alerts_dir(split_pairs, out_dir / split_name)
        logger.info("%s: %d 条 -> %s", split_name, len(split_pairs), out_dir / split_name)


if __name__ == "__main__":
    main()
