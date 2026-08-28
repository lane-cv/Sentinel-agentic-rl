from __future__ import annotations

import argparse
import glob
import json
import statistics
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORTS_DIR = REPO_ROOT / "reports"


def _git_head() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except FileNotFoundError:
        return None
    return None


def _load_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    return rows


def _safe_div(numer: float, denom: float) -> float | None:
    if denom <= 0:
        return None
    return round(numer / denom, 4)


def _relative_source(p: Path) -> str:
    resolved = p.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return resolved.name


def _p50(values: list[float]) -> float | None:
    if not values:
        return None
    return round(statistics.median(values), 3)


def _aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {
            "rows": 0,
            "scenarios_total": 0,
            "runs_per_scenario_avg": 0,
            "effectiveness": {},
        }

    per_scenario: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        per_scenario[r["scenario"]].append(r)

    scenarios_total = len(per_scenario)
    runs_per_scenario_avg = round(len(rows) / scenarios_total, 2)

    service_acc = _safe_div(sum(1 for r in rows if r["service_ok"]), len(rows))
    kind_acc = _safe_div(sum(1 for r in rows if r["kind_ok"]), len(rows))
    route_acc = _safe_div(sum(1 for r in rows if r["route_ok"]), len(rows))

    fix_rows = [r for r in rows if r["triggered_code_fix"]]
    fix_success = _safe_div(
        sum(1 for r in fix_rows if r.get("fix_verified")),
        len(fix_rows),
    )

    loc_rows = [r for r in rows if r.get("changed_files_hit") is not None]
    code_location_acc = _safe_div(
        sum(1 for r in loc_rows if r["changed_files_hit"]),
        len(loc_rows),
    )

    latencies = [float(r["total_latency_s"]) for r in rows if r.get("total_latency_s")]
    mttr_s_p50 = _p50(latencies)
    mttr_min_p50 = round(mttr_s_p50 / 60.0, 3) if mttr_s_p50 is not None else None

    verified_rows = [r for r in fix_rows if r.get("fix_verified")]
    pr_rate = _safe_div(
        sum(1 for r in verified_rows if r.get("pr_created")),
        len(verified_rows),
    )

    return {
        "rows": len(rows),
        "scenarios_total": scenarios_total,
        "runs_per_scenario_avg": runs_per_scenario_avg,
        "effectiveness": {
            "service_accuracy": service_acc,
            "kind_accuracy": kind_acc,
            "route_accuracy": route_acc,
            "fix_success_rate": fix_success,
            "code_location_accuracy": code_location_acc,
            "mttr_seconds_p50": mttr_s_p50,
            "mttr_minutes_p50": mttr_min_p50,
            "pr_submission_rate": pr_rate,
        },
        "denominators": {
            "service_accuracy": len(rows),
            "kind_accuracy": len(rows),
            "route_accuracy": len(rows),
            "fix_success_rate": len(fix_rows),
            "code_location_accuracy": len(loc_rows),
            "mttr_p50": len(latencies),
            "pr_submission_rate": len(verified_rows),
        },
        "per_scenario_runs": {name: len(rs) for name, rs in per_scenario.items()},
    }


def build_report(paths: list[Path]) -> dict:
    rows = _load_rows(paths)
    agg = _aggregate(rows)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": _git_head(),
        "sources": [_relative_source(p) for p in paths],
        **agg,
        "baseline_reference": {
            "human_mttr_minutes_low": 30,
            "human_mttr_minutes_high": 60,
            "source": "Google SRE Book / DORA Report 行业中位数",
            "note": "引用行业公开均值做对照，非本项目内做过的人工对照实验。",
        },
    }


def write_report(out: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _resolve_paths(reports_dir: Path, explicit_input: str | None) -> tuple[list[Path], Path]:
    if explicit_input:
        p = Path(explicit_input)
        return [p], p.parent

    flat = sorted(reports_dir.glob("multi_run_*.jsonl"))
    if flat:
        return flat, reports_dir

    run_dirs = sorted(reports_dir.glob("run_*")) or sorted(reports_dir.glob("*/run_*"))
    if run_dirs:
        latest = run_dirs[-1]
        return sorted(latest.glob("multi_run_*.jsonl")), latest

    return [], reports_dir


def _main(args: argparse.Namespace) -> None:
    reports_dir = Path(args.reports_dir)
    paths, default_out_dir = _resolve_paths(reports_dir, args.input)

    if not paths:
        raise SystemExit(
            f"[aggregate] no multi_run_*.jsonl found under {reports_dir} "
            "(直接的文件或 run_* 子目录都没有); run `python3 -m eval.multi_run` first."
        )

    print(f"[aggregate] loading {len(paths)} file(s):")
    for p in paths:
        print(f"  - {p}")

    out = build_report(paths)

    out_path = Path(args.output) if args.output else default_out_dir / "metrics.json"
    write_report(out, out_path)
    print(f"\n[aggregate] wrote {out_path}")

    eff = out["effectiveness"]
    print("[aggregate] effectiveness summary:")
    for k, v in eff.items():
        print(f"  {k:28s} = {v}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Sentinel agent effectiveness metrics")
    parser.add_argument("--input", help="single multi_run_*.jsonl file (default: glob all in reports-dir)")
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR), help="reports directory")
    parser.add_argument("--output", help="output path (default: <reports-dir>/metrics.json)")
    args = parser.parse_args()
    _main(args)


if __name__ == "__main__":
    main()
