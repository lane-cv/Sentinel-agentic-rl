from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from data.cold_start.prefix_split import DEFAULT_TRAJ_DIR, split_directory

REFERENCE_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "prefix_split_reference.json"


def run(traj_dir: Path = DEFAULT_TRAJ_DIR) -> dict[str, Any]:
    n_trajectories = len(list(Path(traj_dir).glob("*.json")))
    if n_trajectories == 0:
        raise RuntimeError(f"no trajectory files found under {traj_dir}")

    subsamples = split_directory(traj_dir)
    prefix_split_total = len(subsamples)
    prefix_split_converge = sum(1 for s in subsamples if s["target_type"] == "diagnosis")
    prefix_split_continue = prefix_split_total - prefix_split_converge
    prefix_split_ratio = round(prefix_split_converge / prefix_split_total, 4) if prefix_split_total else None

    no_split_total = n_trajectories
    no_split_converge = 0
    no_split_continue = no_split_total
    no_split_ratio = round(no_split_converge / no_split_total, 4) if no_split_total else None

    return {
        "n_trajectories": n_trajectories,
        "avg_steps_per_trajectory": round(prefix_split_total / n_trajectories, 3),
        "prefix_split": {
            "subsample_count": prefix_split_total,
            "converge_examples": prefix_split_converge,
            "continue_examples": prefix_split_continue,
            "termination_example_ratio": prefix_split_ratio,
        },
        "no_split": {
            "subsample_count": no_split_total,
            "converge_examples": no_split_converge,
            "continue_examples": no_split_continue,
            "termination_example_ratio": no_split_ratio,
        },
        "subsample_count_delta": prefix_split_total - no_split_total,
        "termination_example_ratio_delta": (
            round(prefix_split_ratio - no_split_ratio, 4)
            if prefix_split_ratio is not None and no_split_ratio is not None
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Count training examples with and without prefix splitting")
    parser.add_argument("--show-reference", action="store_true", help="also print the published comparison figures and their source")
    args = parser.parse_args()
    result = run()
    print("[no_prefix_split] real computation over data/cold_start/trajectories/mock/*.json:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.show_reference:
        print("\n[no_prefix_split] Published comparison figures:")
        print(REFERENCE_PATH.read_text(encoding="utf-8").rstrip())
    print(
        f"\n[no_prefix_split] prefix-split gives {result['prefix_split']['converge_examples']} "
        f"standalone, fully-weighted 'converge' example(s) out of "
        f"{result['prefix_split']['subsample_count']} total "
        f"(ratio={result['prefix_split']['termination_example_ratio']}); "
        f"no-split gives {result['no_split']['converge_examples']} out of "
        f"{result['no_split']['subsample_count']} (ratio={result['no_split']['termination_example_ratio']})."
    )


if __name__ == "__main__":
    main()
