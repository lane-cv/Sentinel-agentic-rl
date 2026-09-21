from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prune_steps
import rejection_sample as rs

GT_DIR = HERE / "ground_truth"
RAW_ROOT = HERE / "trajectories_v9_raw"
GOVERNED = HERE / "trajectories_governed_v9"
AUG_MAP = HERE.parent.parent / "data" / "seeds" / "generated" / "_v9_augmentation_map.json"
STATE = HERE / "_collection_v9_state.json"
FINAL_DIR = HERE / "trajectories_v9_final"
STATS_OUT = HERE / "_v9_dataset_stats.md"

FAMILY_QUOTA = {"B1a": 6, "B1b": 3, "A1": 5, "D1": 3, "C1": 6}

POLLUTION_PATTERNS = [
    "eval/expected.json", "eval/multi_run", "multi_run/", "multi_run目录", "reports/", "alerts/s1", "alerts/s2",
    "alerts/s3", "alerts/s4", "alerts/s5", "alerts/s6", "alerts/s7", "alerts/s8", "alerts/s9",
    "s10_ranking", "s11_dedupe", "s12_race", "s7_hybrid", "s8_fix_fail", "s9_adv",
    "s_lowconf", "s5_dup", "s6_info", "expect_route", "suspect_service_contains", "kind_any_of",
]


def best_accepted() -> dict[str, tuple[int, Path]]:
    state = json.loads(STATE.read_text())
    best: dict[str, tuple[int, Path]] = {}
    for r in state["runs"]:
        stem = r["stem"]
        if stem in best or r.get("verdict") != "ACCEPTED":
            continue
        p = RAW_ROOT / f"round{r['round']}" / f"{stem}.json"
        if p.exists():
            best[stem] = (r["round"], p)
    return best


def pollution_scan(traj: dict[str, Any]) -> list[str]:
    hits = []
    for i, s in enumerate(traj.get("steps") or [], 1):
        blob = json.dumps({"cmd": s.get("tool_input"), "obs": s.get("observation")}, ensure_ascii=False)
        for pat in POLLUTION_PATTERNS:
            if pat in blob:
                hits.append(f"step{i}:{pat}")
    return hits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-gates", action="store_true")
    args = parser.parse_args()

    aug = json.loads(AUG_MAP.read_text())
    accepted = best_accepted()
    print(f"[assemble] 新种子验收通过 {len(accepted)}/47")

    GOVERNED.mkdir(parents=True, exist_ok=True)
    newly_governed = []
    for stem, (rnd, path) in sorted(accepted.items()):
        traj = json.loads(path.read_text())
        gt = json.loads((GT_DIR / f"{stem}.json").read_text())
        pruned = prune_steps.prune_trajectory(traj)
        verdict = rs.check_trajectory(pruned, gt)
        if not verdict["keep"]:
            print(f"[assemble][warn] {stem} 修剪后验收失败（跳过入库）: {verdict['reasons']}")
            continue
        out = GOVERNED / f"{stem}.json"
        out.write_text(json.dumps(pruned, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        newly_governed.append(stem)
    print(f"[assemble] C2 修剪后入库 {len(newly_governed)} 条新轨迹（含 meta.pruning）")

    if FINAL_DIR.exists():
        shutil.rmtree(FINAL_DIR)
    FINAL_DIR.mkdir(parents=True)
    family_of: dict[str, str] = {}
    for f in sorted(GOVERNED.glob("*.json")):
        shutil.copy(f, FINAL_DIR / f.name)
        stem = f.stem
        if stem in aug:
            family_of[stem] = aug[stem]["family"]
        else:
            family_of[stem] = "v8_base"
    n_total = len(list(FINAL_DIR.glob("*.json")))
    print(f"[assemble] trajectories_v9_final: {n_total} 条（基座 28 + 新 {len(newly_governed)}）")

    gates: dict[str, Any] = {}
    all_trajs = {f.stem: json.loads(f.read_text()) for f in FINAL_DIR.glob("*.json")}

    poll_hits = []
    for stem, traj in all_trajs.items():
        hits = pollution_scan(traj)
        if hits:
            poll_hits.append((stem, hits[:3]))
    gates["污染扫描"] = {"pass": not poll_hits, "detail": poll_hits if poll_hits else "0 命中"}

    kind_fail = []
    for stem, traj in all_trajs.items():
        gt_path = GT_DIR / f"{stem}.json"
        if not gt_path.exists():
            continue
        v = rs.check_trajectory(traj, json.loads(gt_path.read_text()))
        if not v["keep"]:
            kind_fail.append((stem, v["reasons"][:2]))
    gates["全量重验收(含kind)"] = {"pass": not kind_fail, "detail": kind_fail if kind_fail else f"{len(all_trajs)}/{len(all_trajs)} 全过"}

    malformed = []
    for stem, traj in all_trajs.items():
        for i, s in enumerate(traj.get("steps") or [], 1):
            if prune_steps.is_malformed_step(s):
                malformed.append((stem, i))
    gates["malformed目标=0"] = {"pass": not malformed, "detail": malformed if malformed else "0"}

    fam_cnt = Counter(family_of.values())
    quota_detail = {f: f"{fam_cnt.get(f, 0)}/{q}" for f, q in FAMILY_QUOTA.items()}
    quota_pass = all(fam_cnt.get(f, 0) >= q for f, q in FAMILY_QUOTA.items())
    lowconf_targets = fam_cnt.get("B1b", 0) + sum(
        1 for s, f in family_of.items()
        if f == "A1"
        and json.loads((GT_DIR / f"{s}.json").read_text()).get("expect_route") == "feishu_low_confidence"
    )
    quota_detail["B1b+A1低置信目标"] = f"{lowconf_targets}/5"
    gates["家族配额"] = {"pass": quota_pass, "detail": quota_detail}

    step_counts = {s: len(t.get("steps") or []) for s, t in all_trajs.items()}
    mean_steps = sum(step_counts.values()) / len(step_counts)
    gates["步数均值≤12"] = {"pass": mean_steps <= 12, "detail": f"{mean_steps:.2f}"}

    state = json.loads(STATE.read_text())
    ran_stems = {r["stem"] for r in state["runs"]}
    acc_stems = {r["stem"] for r in state["runs"] if r.get("verdict") == "ACCEPTED"}
    rate = len(acc_stems) / len(ran_stems) if ran_stems else 0
    gates["总验收率≥55%"] = {"pass": rate >= 0.55, "detail": f"{len(acc_stems)}/{len(ran_stems)} = {rate*100:.0f}%"}

    print("\n===== 六道数据门（§8.1） =====")
    all_pass = True
    for name, g in gates.items():
        mark = "PASS" if g["pass"] else "FAIL"
        if not g["pass"]:
            all_pass = False
        print(f"  [{mark}] {name}: {g['detail']}")

    subsamples_path = HERE / "prefix_subsamples_v9.json"
    subprocess.run(
        [sys.executable, str(HERE / "prefix_split.py"), "--traj-dir", str(FINAL_DIR), "--out", str(subsamples_path)],
        check=True,
    )
    jsonl_path = HERE / "aiops_cold_start_sft_v9.jsonl"
    subprocess.run(
        [sys.executable, str(HERE / "to_llamafactory_format.py"), "--in", str(subsamples_path),
         "--out", str(jsonl_path), "--dataset-info-out", str(HERE / "dataset_info_snippet_v9.json")],
        check=True,
    )

    fam_traj = Counter(family_of.values())
    step_dist = sorted(step_counts.values())
    rem_types = Counter(t["final_diagnosis"].get("remediation_type") for t in all_trajs.values())
    routes = Counter(t.get("route") for t in all_trajs.values())
    kinds = Counter(t["final_diagnosis"].get("kind") for t in all_trajs.values())
    confs = [t["final_diagnosis"].get("confidence") for t in all_trajs.values()]
    subs = json.loads(subsamples_path.read_text())

    lines = []
    ap = lines.append
    ap("# v9 数据集统计（Phase C 组装产物）")
    ap("")
    ap(f"- 生成时间：{__import__('datetime').datetime.now(timezone.utc).isoformat(timespec='seconds')}" if False else "- 轨迹总数：%d（v8 治理基座 28 + v9 新增 %d）" % (n_total, len(newly_governed)))
    ap("- 家族分布（轨迹数）：" + json.dumps(dict(sorted(fam_traj.items())), ensure_ascii=False))
    ap("- remediation_type：" + json.dumps(dict(rem_types), ensure_ascii=False))
    ap("- route：" + json.dumps({str(k): v for k, v in routes.items()}, ensure_ascii=False))
    ap("- kind：" + json.dumps(dict(kinds), ensure_ascii=False))
    ap("- confidence：min=%.2f p50=%.2f max=%.2f" % (min(confs), sorted(confs)[len(confs)//2], max(confs)))
    ap("- 步数：mean=%.2f min=%d p50=%d max=%d（门：均值≤12）" % (mean_steps, step_dist[0], step_dist[len(step_dist)//2], step_dist[-1]))
    ap("- 前缀子样本：%d 条（%d tool_call + %d diagnosis）" % (
        len(subs), sum(1 for s in subs if s["target_type"] == "tool_call"),
        sum(1 for s in subs if s["target_type"] == "diagnosis")))
    ap("- 最终 jsonl：%s（%d 行）" % (jsonl_path.name, sum(1 for _ in jsonl_path.open())))
    ap("")
    ap("## 六道数据门")
    for name, g in gates.items():
        ap("- [%s] %s：%s" % ("PASS" if g["pass"] else "FAIL", name, json.dumps(g["detail"], ensure_ascii=False, default=str)[:400]))
    ap("")
    ap("## v9 新增轨迹明细（%d 条，按家族）" % len(newly_governed))
    for stem in sorted(newly_governed):
        t = all_trajs[stem]
        m = t.get("meta", {}).get("pruning", {})
        ap("- `%s` [%s] steps %d→%d conf=%.2f kind=%s route=%s" % (
            stem, family_of.get(stem, "?"), m.get("steps_before", "?"), m.get("steps_after", len(t["steps"])),
            t["final_diagnosis"].get("confidence", 0), t["final_diagnosis"].get("kind"), t.get("route")))
    from datetime import timezone
    STATS_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[assemble] 统计写入 {STATS_OUT}")
    print(f"[assemble] 数据门整体：{'ALL PASS' if all_pass else 'HAS FAILURES（见上）'}")


if __name__ == "__main__":
    main()
