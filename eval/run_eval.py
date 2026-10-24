from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from eval import metrics
from eval.model_endpoints import ENDPOINT_REGISTRY, DiagnosisEndpoint, build_endpoint
from reward.trajectory import Trajectory

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DEFAULT_TRAJ_DIR = REPO_ROOT / "data" / "cold_start" / "trajectories" / "mock"
DEFAULT_GT_DIR = REPO_ROOT / "data" / "cold_start" / "ground_truth"
DEFAULT_REPORTS_DIR = HERE / "reports"

_PASSTHROUGH_GT_FIELDS: tuple[str, ...] = (
    "suspect_service",
    "remediation_type",
    "kind",
    "suspect_repo",
    "suspect_commit_hint",
    "suspect_file_hint",
)


@dataclass
class EvalCase:

    case_id: str
    alert: dict[str, Any]
    ground_truth: dict[str, Any]


def load_eval_set(traj_dir: Path = DEFAULT_TRAJ_DIR, gt_dir: Path = DEFAULT_GT_DIR) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for traj_path in sorted(Path(traj_dir).glob("*.json")):
        case_id = traj_path.stem
        gt_path = Path(gt_dir) / f"{case_id}.json"
        if not gt_path.exists():
            continue
        traj_raw = json.loads(traj_path.read_text(encoding="utf-8"))
        gt_raw = json.loads(gt_path.read_text(encoding="utf-8"))

        ground_truth: dict[str, Any] = {}
        for field_name in _PASSTHROUGH_GT_FIELDS:
            if gt_raw.get(field_name) is not None:
                ground_truth[field_name] = gt_raw[field_name]
        route = gt_raw.get("expect_route", gt_raw.get("route"))
        if route is not None:
            ground_truth["route"] = route
        ground_truth["expected_step_count"] = len(traj_raw.get("steps") or [])

        cases.append(EvalCase(case_id=case_id, alert=traj_raw["alert"], ground_truth=ground_truth))
    return cases


def check_case(traj: Trajectory) -> dict[str, Any]:
    gt = traj.ground_truth
    diag = traj.final_diagnosis
    checks: dict[str, bool] = {}
    if gt.get("kind") is not None:
        checks["kind"] = diag.get("kind") == gt["kind"]
    if gt.get("suspect_service") is not None:
        checks["suspect_service"] = (diag.get("suspect_service") or "").lower() == gt["suspect_service"].lower()
    if gt.get("remediation_type") is not None:
        checks["remediation_type"] = diag.get("remediation_type") == gt["remediation_type"]
    if gt.get("route") is not None:
        checks["route"] = traj.route == gt["route"]
    return {"checks": checks, "passed": all(checks.values()) if checks else False}


def run_endpoint(endpoint: DiagnosisEndpoint, cases: list[EvalCase]) -> list[Trajectory]:
    return [endpoint.diagnose(case.alert, case.ground_truth) for case in cases]


def build_report(endpoint_name: str, cases: list[EvalCase], trajectories: list[Trajectory]) -> dict[str, Any]:
    per_case: list[dict[str, Any]] = []
    for case, traj in zip(cases, trajectories):
        result = check_case(traj)
        per_case.append(
            {
                "case_id": case.case_id,
                "ground_truth": case.ground_truth,
                "route": traj.route,
                "final_diagnosis": {
                    k: traj.final_diagnosis.get(k)
                    for k in ("kind", "suspect_service", "remediation_type", "confidence", "evidence")
                },
                "num_steps": len(traj.steps),
                "checks": result["checks"],
                "passed": result["passed"],
            }
        )
    n_passed = sum(1 for row in per_case if row["passed"])
    return {
        "endpoint": endpoint_name,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_cases": len(cases),
        "n_passed": n_passed,
        "pass_rate": round(n_passed / len(cases), 4) if cases else None,
        "per_case": per_case,
        "metrics": metrics.compute_all(trajectories),
        "evaluation_mode": "scripted_mock",
        "caveat": (
            "The scripted endpoint reads the case labels. These metrics validate the scoring logic."
        ),
    }


def run(endpoint_key: str, traj_dir: Path = DEFAULT_TRAJ_DIR, gt_dir: Path = DEFAULT_GT_DIR) -> dict[str, Any]:
    endpoint = build_endpoint(endpoint_key)
    cases = load_eval_set(traj_dir, gt_dir)
    trajectories = run_endpoint(endpoint, cases)
    return build_report(endpoint.name, cases, trajectories)


def _main(args: argparse.Namespace) -> None:
    report = run(args.endpoint)

    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_path = Path(args.out) if args.out else reports_dir / f"run_eval_{args.endpoint}_{ts}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[run_eval] scripted mock | endpoint={args.endpoint}  {report['n_passed']}/{report['n_cases']} passed "
          f"(pass_rate={report['pass_rate']})")
    for k, v in report["metrics"].items():
        print(f"  {k:34s} = {v}")
    print(f"[run_eval] wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check scoring with a scripted endpoint and labeled cases")
    parser.add_argument("--endpoint", choices=sorted(ENDPOINT_REGISTRY), default="perfect")
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--out", help="explicit output path (default: <reports-dir>/run_eval_<endpoint>_<ts>.json)")
    args = parser.parse_args()
    _main(args)


if __name__ == "__main__":
    main()
