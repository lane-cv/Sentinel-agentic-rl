from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval.model_endpoints import (
    OpusReferencePlaceholderEndpoint,
    PartiallyGoodEndpoint,
    PerfectDiagnosisEndpoint,
    SFTGRPOEndpoint,
    ZeroShotLikeEndpoint,
)
from eval.run_eval import DEFAULT_REPORTS_DIR, build_report, load_eval_set, run_endpoint

DOC_TIER_ENDPOINTS: tuple[type, ...] = (
    ZeroShotLikeEndpoint,
    PartiallyGoodEndpoint,
    SFTGRPOEndpoint,
    OpusReferencePlaceholderEndpoint,
)

USER_LEVEL_METRIC_KEYS: tuple[str, ...] = ("route_accuracy", "stop_loss_success_rate")

SYSTEM_INTERNAL_METRIC_KEYS: tuple[str, ...] = (
    "root_cause_field_accuracy_overall",
    "evidence_traceability_rate",
    "avg_tool_call_rounds",
    "termination_decision_accuracy",
)


def run_comparison(include_perfect: bool = False) -> dict[str, Any]:
    cases = load_eval_set()
    endpoint_classes = list(DOC_TIER_ENDPOINTS)
    if include_perfect:
        endpoint_classes.append(PerfectDiagnosisEndpoint)

    reports: dict[str, dict[str, Any]] = {}
    for cls in endpoint_classes:
        endpoint = cls()
        trajectories = run_endpoint(endpoint, cases)
        reports[endpoint.name] = build_report(endpoint.name, cases, trajectories)

    def _row(keys: tuple[str, ...]) -> dict[str, dict[str, Any]]:
        return {key: {name: reports[name]["metrics"].get(key) for name in reports} for key in keys}

    comparison_table = {
        "user_level": _row(USER_LEVEL_METRIC_KEYS),
        "system_internal": _row(SYSTEM_INTERNAL_METRIC_KEYS),
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tiers": [cls().name for cls in endpoint_classes],
        "n_cases": len(cases),
        "reports": reports,
        "comparison_table": comparison_table,
        "evaluation_mode": "scripted_mock",
        "caveat": "All tiers are scripted and read the case labels. The comparison checks the scoring pipeline.",
    }


def _print_table(comparison: dict[str, Any]) -> None:
    tiers = comparison["tiers"]
    print(f"[run_baseline_comparison] scripted mock | n_cases={comparison['n_cases']}  tiers={tiers}")
    for section_name, section in comparison["comparison_table"].items():
        print(f"\n[{section_name}]")
        header = "  " + "metric".ljust(34) + "".join(t.ljust(24) for t in tiers)
        print(header)
        for metric_key, per_tier in section.items():
            row = "  " + metric_key.ljust(34) + "".join(str(per_tier.get(t)).ljust(24) for t in tiers)
            print(row)


def _main(args: argparse.Namespace) -> None:
    comparison = run_comparison(include_perfect=args.include_perfect)
    _print_table(comparison)

    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_path = Path(args.out) if args.out else reports_dir / f"baseline_comparison_{ts}.json"
    out_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n[run_baseline_comparison] wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare scripted baselines on labeled cases")
    parser.add_argument("--include-perfect", action="store_true", help="also run the 100%%-by-construction sanity double")
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--out", help="explicit output path")
    args = parser.parse_args()
    _main(args)


if __name__ == "__main__":
    main()
