from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_TRAJ_DIR = HERE / "trajectories" / "mock"
DEFAULT_GT_DIR = HERE / "ground_truth"

_IDENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.%-]*")

_CODE_FIX_DEPENDENT_ROUTES = {
    "code_fix_pr",
    "feishu_fix_unverified",
    "auto_remediated_and_code_fix_pr",
    "online_op_and_code_fix_pr",
}


def _significant_tokens(text: str) -> set[str]:
    return {t.lower() for t in _IDENT_RE.findall(text or "") if len(t) >= 2}


def is_evidence_traceable(evidence: str, steps: list[dict[str, Any]]) -> bool:
    ev_tokens = _significant_tokens(evidence)
    if not ev_tokens:
        stripped = evidence.strip()
        return bool(stripped) and any(stripped in (s.get("observation", "") or "") for s in steps)
    need = min(2, len(ev_tokens))
    for step in steps:
        haystack = f"{step.get('observation', '')} {json.dumps(step.get('tool_input', {}), ensure_ascii=False)}"
        hay_tokens = _significant_tokens(haystack)
        if len(ev_tokens & hay_tokens) >= need:
            return True
    return False


def evidence_traceability_ok(diagnosis: dict[str, Any], steps: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    bad = [ev for ev in diagnosis.get("evidence", []) if not is_evidence_traceable(ev, steps)]
    return (len(bad) == 0, bad)


def executed_actions_match_ledger(steps: list[dict[str, Any]], executed_actions: list[dict[str, Any]]) -> bool:
    ledger = {
        (s.get("hook_action"), s.get("hook_target"), (s.get("tool_input") or {}).get("command", ""))
        for s in steps
        if s.get("hook_decision") == "allow"
    }
    claimed = {(a.get("action"), a.get("target"), a.get("command", "")) for a in executed_actions}
    return ledger == claimed


def suspect_service_matches(actual: str, expected: str) -> bool:
    return bool(expected) and expected.strip().lower() in (actual or "").strip().lower()


def expect_kind_set(gt: dict[str, Any]) -> list[str] | None:
    any_of = gt.get("expect_kind_any_of")
    if isinstance(any_of, list) and any_of:
        return [str(k) for k in any_of]
    single = gt.get("expect_kind")
    if isinstance(single, str) and single:
        return [single]
    return None


def _confidence_in_range(diag: dict[str, Any], rng: Any) -> tuple[bool, str]:
    lo, hi = float(rng[0]), float(rng[1])
    conf = diag.get("confidence")
    ok = isinstance(conf, (int, float)) and lo <= float(conf) <= hi
    return ok, f"confidence={conf!r} 不在期望区间 [{lo}, {hi}]"


def _suspect_file_hits(diag: dict[str, Any], files: list[str]) -> tuple[bool, str]:
    hint = (diag.get("suspect_file_hint") or "")
    hit = next((f for f in files if f.lower() in hint.lower()), None)
    return hit is not None, f"suspect_file_hint={hint!r} 未命中 expect_suspect_file_any_of={files!r}"


def _steps_within_max(traj: dict[str, Any], max_steps: int) -> tuple[bool, str]:

    def _malformed(step: dict[str, Any]) -> bool:
        return "__unparsedToolInput" in json.dumps(step.get("tool_input") or {}, ensure_ascii=False)

    n = len([s for s in (traj.get("steps") or []) if not _malformed(s)])
    return n <= max_steps, f"步数 {n} 超过 expect_max_steps={max_steps}"


def _negative_claim_present(diag: dict[str, Any], claim: dict[str, Any]) -> tuple[bool, str]:
    markers = [str(m) for m in (claim.get("markers") or [])]
    obj = claim.get("object")
    text = " ".join(
        [str(diag.get("summary") or "")] + [str(e) for e in (diag.get("evidence") or [])]
    )
    low = text.lower()
    hit_marker = next((m for m in markers if m.lower() in low), None)
    if hit_marker is None:
        return False, f"summary/evidence 缺少否定性/如实措辞（markers={markers!r} 一个都没出现）"
    if obj and str(obj).lower() not in low:
        return False, f"否定性措辞存在（{hit_marker!r}）但未提及期望对象 {obj!r}"
    return True, ""


def _evidence_groups_covered(diag: dict[str, Any], groups: list[list[str]]) -> tuple[bool, str]:
    evidences = [str(e) for e in (diag.get("evidence") or [])]
    low_ev = [e.lower() for e in evidences]
    missing = []
    for gi, group in enumerate(groups):
        tokens = [str(t) for t in group]
        if not any(any(t.lower() in ev for t in tokens) for ev in low_ev):
            missing.append((gi, tokens))
    if missing:
        return False, f"evidence 未覆盖的信源组: {missing!r}"
    return True, ""


def _detail_contains_any(diag: dict[str, Any], needles: list[str]) -> tuple[bool, str]:
    detail = str(diag.get("remediation_detail") or "")
    hit = next((n for n in needles if n.lower() in detail.lower()), None)
    return hit is not None, f"remediation_detail 未包含 expect_detail_contains_any={needles!r} 中任何一个"


def check_trajectory(traj: dict[str, Any], gt: dict[str, Any]) -> dict[str, Any]:
    diag = traj.get("final_diagnosis") or {}
    steps = traj.get("steps") or []
    reasons: list[str] = []

    kind_any_of = expect_kind_set(gt)
    if kind_any_of is not None:
        actual_kind = (diag.get("kind") or "").strip()
        if actual_kind not in kind_any_of:
            reasons.append(
                f"kind 不在期望集合: got={actual_kind!r} expect_any_of={kind_any_of!r}"
            )

    remediation_any_of = gt.get("expect_remediation_any_of")
    if isinstance(remediation_any_of, list) and remediation_any_of:
        if diag.get("remediation_type") not in remediation_any_of:
            reasons.append(
                f"remediation_type 不一致: got={diag.get('remediation_type')!r}"
                f" expect_any_of={remediation_any_of!r}"
            )
    elif diag.get("remediation_type") != gt.get("remediation_type"):
        reasons.append(
            f"remediation_type 不一致: got={diag.get('remediation_type')!r} expect={gt.get('remediation_type')!r}"
        )

    conf_range = gt.get("expect_confidence_range")
    if isinstance(conf_range, list) and len(conf_range) == 2:
        ok, why = _confidence_in_range(diag, conf_range)
        if not ok:
            reasons.append(f"confidence 越界: {why}")

    max_steps = gt.get("expect_max_steps")
    if isinstance(max_steps, int):
        ok, why = _steps_within_max(traj, max_steps)
        if not ok:
            reasons.append(why)

    suspect_files = gt.get("expect_suspect_file_any_of")
    if isinstance(suspect_files, list) and suspect_files:
        ok, why = _suspect_file_hits(diag, [str(f) for f in suspect_files])
        if not ok:
            reasons.append(why)

    negative_claim = gt.get("expect_negative_claim")
    if isinstance(negative_claim, dict) and negative_claim.get("markers"):
        ok, why = _negative_claim_present(diag, negative_claim)
        if not ok:
            reasons.append(f"缺少否定性/如实陈述: {why}")

    evidence_groups = gt.get("expect_evidence_groups")
    if isinstance(evidence_groups, list) and evidence_groups:
        ok, why = _evidence_groups_covered(diag, evidence_groups)
        if not ok:
            reasons.append(f"evidence 信源覆盖不足: {why}")

    detail_any = gt.get("expect_detail_contains_any")
    if isinstance(detail_any, list) and detail_any:
        ok, why = _detail_contains_any(diag, [str(n) for n in detail_any])
        if not ok:
            reasons.append(why)

    if not suspect_service_matches(diag.get("suspect_service", ""), gt.get("suspect_service", "")):
        reasons.append(
            f"suspect_service 不一致: got={diag.get('suspect_service')!r} expect_contains={gt.get('suspect_service')!r}"
        )

    expect_route = gt.get("expect_route")
    route_any_of = gt.get("expect_route_any_of")
    actual_route = traj.get("route")
    code_fix_dependent = (
        expect_route in _CODE_FIX_DEPENDENT_ROUTES
        or (isinstance(route_any_of, list) and any(r in _CODE_FIX_DEPENDENT_ROUTES for r in route_any_of))
    )
    if code_fix_dependent and actual_route is None:
        if gt.get("remediation_type") == "online_op":
            expect_also_code_fix = gt.get("expect_also_code_fix")
            if diag.get("also_code_fix") != expect_also_code_fix:
                reasons.append(
                    f"also_code_fix 判断错误: got={diag.get('also_code_fix')!r} expect={expect_also_code_fix!r}"
                )
    elif isinstance(route_any_of, list) and route_any_of and expect_route is None:
        if actual_route not in route_any_of:
            reasons.append(f"route 不一致: got={actual_route!r} expect_any_of={route_any_of!r}")
    elif actual_route != expect_route:
        reasons.append(f"route 不一致: got={actual_route!r} expect={expect_route!r}")

    ok, bad_evidence = evidence_traceability_ok(diag, steps)
    if not ok:
        reasons.append(f"evidence 无法追溯到工具观测出处: {bad_evidence!r}")

    if not executed_actions_match_ledger(steps, diag.get("executed_actions") or []):
        reasons.append("executed_actions 与轨迹里 hook 放行台账不一致")

    return {"keep": len(reasons) == 0, "reasons": reasons}


def run(traj_dir: Path, gt_dir: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for f in sorted(Path(traj_dir).glob("*.json")):
        traj = json.loads(f.read_text(encoding="utf-8"))
        alert_id = traj.get("alert_id") or f.stem
        gt_path = Path(gt_dir) / f"{alert_id}.json"
        if not gt_path.exists():
            results[alert_id] = {"keep": False, "reasons": [f"未找到人工标注 ground truth: {gt_path}"]}
            continue
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        results[alert_id] = check_trajectory(traj, gt)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="对冷启轨迹跑拒绝采样过滤")
    parser.add_argument("--traj-dir", type=Path, default=DEFAULT_TRAJ_DIR, help="轨迹 JSON 所在目录")
    parser.add_argument("--gt-dir", type=Path, default=DEFAULT_GT_DIR, help="人工标注 ground truth 所在目录")
    parser.add_argument("--copy-kept-to", type=Path, default=None, help="可选：把通过的轨迹复制到这个目录")
    args = parser.parse_args()

    results = run(args.traj_dir, args.gt_dir)
    print(json.dumps(results, ensure_ascii=False, indent=2))

    kept = [k for k, v in results.items() if v["keep"]]
    rejected = [k for k, v in results.items() if not v["keep"]]
    total = len(results)
    print(f"\n通过 {len(kept)}/{total}: {kept}")
    print(f"拒绝 {len(rejected)}/{total}: {rejected}")

    if args.copy_kept_to:
        import shutil

        args.copy_kept_to.mkdir(parents=True, exist_ok=True)
        for alert_id in kept:
            src = Path(args.traj_dir) / f"{alert_id}.json"
            shutil.copy(src, args.copy_kept_to / src.name)
        print(f"已把 {len(kept)} 条通过的轨迹复制到 {args.copy_kept_to}")


if __name__ == "__main__":
    main()
