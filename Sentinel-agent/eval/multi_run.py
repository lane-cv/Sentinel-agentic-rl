from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from agent import config
from agent.integrations import ingest
from agent.integrations.ingest import load_alert
from agent.run import run
from eval import metrics
from eval.aggregate import build_report, write_report

EXPECTED_PATH = Path(__file__).parent / "expected.json"


def _model_slug() -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", config.MODEL)


def _reset_dedup_cache() -> None:
    ingest._SEEN.clear()


async def _run_once(alert_path: str) -> dict:
    alert = load_alert(alert_path)
    return await run(alert)


async def _run_scenario(name: str, exp: dict, exp_all: dict, default_runs: int) -> list[dict]:
    effective_runs = int(exp.get("runs", default_runs))
    depends_on = exp.get("depends_on")
    rows: list[dict] = []

    for i in range(effective_runs):
        _reset_dedup_cache()

        if depends_on:
            dep_exp = exp_all[depends_on]
            print(f"[multi_run] {name} run#{i}: seeding dependency {depends_on} ({dep_exp['alert']})")
            _ = await _run_once(dep_exp["alert"])
            report = await _run_once(exp["alert"])
        else:
            report = await _run_once(exp["alert"])

        row = metrics.extract(name, exp, report)
        row["run_index"] = i
        row["ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        rows.append(row)

        mark = (
            f"route={row['route']}  service_ok={row['service_ok']}  "
            f"kind_ok={row['kind_ok']}  route_ok={row['route_ok']}  "
            f"total_latency_s={row['total_latency_s']}"
        )
        print(f"[multi_run] {name} run#{i}: {mark}")

    return rows


async def _main(args: argparse.Namespace) -> None:
    expected = json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))
    names = args.only or list(expected.keys())

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    date_str = os.getenv("AIOPS_EVAL_TAG") or datetime.now().strftime("%Y%m%d")
    model_dir = Path(args.reports_dir) / f"{_model_slug()}_{date_str}"
    run_dir = model_dir / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / f"multi_run_{ts}.jsonl"

    t_start = time.perf_counter()
    total_rows = 0
    with out_path.open("w", encoding="utf-8") as f:
        for name in names:
            if name not in expected:
                print(f"[multi_run] WARN: {name} not in expected.json, skipped")
                continue
            exp = expected[name]
            print(f"\n[multi_run] === {name} (runs={exp.get('runs', args.runs)}) ===")
            rows = await _run_scenario(name, exp, expected, args.runs)
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
            total_rows += len(rows)

    elapsed = time.perf_counter() - t_start
    print(f"\n[multi_run] wrote {total_rows} rows to {out_path} in {elapsed:.1f}s")

    if total_rows:
        run_metrics_path = run_dir / "metrics.json"
        write_report(build_report([out_path]), run_metrics_path)
        print(f"[multi_run] wrote metrics to {run_metrics_path}")

        model_jsonl = sorted(model_dir.glob("run_*/multi_run_*.jsonl"))
        model_metrics_path = model_dir / "metrics.json"
        write_report(build_report(model_jsonl), model_metrics_path)
        print(f"[multi_run] wrote combined metrics to {model_metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sentinel agent multi-run effectiveness eval")
    parser.add_argument("--runs", type=int, default=3, help="default runs per scenario (可被 expected.json 中的 runs 字段覆盖)")
    parser.add_argument("--only", nargs="*", help="subset of scenario names, e.g. --only s2 s4")
    parser.add_argument("--reports-dir", default="reports", help="output directory for jsonl")
    args = parser.parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
