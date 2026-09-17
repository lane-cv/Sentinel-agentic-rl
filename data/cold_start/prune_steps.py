from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import rejection_sample as rs
from signals import classify_signal_types

HERE = Path(__file__).resolve().parent
DEFAULT_TRAJ_DIR = HERE / "trajectories_accepted_v8"
DEFAULT_OUT_DIR = HERE / "trajectories_governed_v9"

_NOISE_TOOL_NAMES = {"TaskStop", "KillShell", "KillBash", "Task"}

_FIND_ROOT_RE = re.compile(r"(?:^|[\s;&|])find\s+/")

_NO_OUTPUT_OBS_RE = re.compile(r"^\((?:Bash completed with no output|No content)\)$|^Exit code \d+$|^---$")

_EMPTY_RESULT_OBS_RE = re.compile(r'"(?:data|result|traces)"\s*:\s*\[\s*\]|"(?:total|errors)"\s*:\s*0\b')

_NOISE_TOKEN_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}t\d{2}"
    r"|^\d{2}:\d{2}:\d{2}"
    r"|^\d+(\.\d+)?z$"
    r"|^[0-9a-f]{12,}$"
    r"|^\d{1,3}(\.\d{1,3}){3}$"
)


def _is_no_info_observation(observation: str) -> bool:
    obs = (observation or "").strip()
    if not obs:
        return True
    return bool(_NO_OUTPUT_OBS_RE.match(obs) or _EMPTY_RESULT_OBS_RE.search(obs))


def _obs_facts(observation: str) -> set[str]:
    if _is_no_info_observation(observation):
        return set()
    return {
        t
        for t in rs._significant_tokens(observation)
        if not _NOISE_TOKEN_RE.match(t)
    }


def is_malformed_step(step: dict[str, Any]) -> bool:
    return "__unparsedToolInput" in json.dumps(step.get("tool_input") or {}, ensure_ascii=False)


def is_noise_step(step: dict[str, Any]) -> bool:
    if step.get("tool_name") in _NOISE_TOOL_NAMES:
        return True
    cmd = str((step.get("tool_input") or {}).get("command", "") or "")
    return bool(_FIND_ROOT_RE.search(" " + cmd.strip() + " ;"))


def _strip_cd_prefix(cmd: str) -> str:
    m = re.match(r"^\s*cd\s+\S+\s*(&&|;)\s*", cmd)
    return cmd[m.end():] if m else cmd


def _bash_skeleton(cmd: str) -> str:
    c = _strip_cd_prefix(cmd.strip())

    m = re.search(r"docker\s+logs\s+([A-Za-z0-9_.-]+)", c)
    if m:
        return f"docker-logs:{m.group(1)}"

    m = re.search(r"docker\s+(?:ps|stats|inspect)\b", c)
    if m:
        f = re.search(r"name=([A-Za-z0-9_.-]+)", c)
        if f:
            return f"docker-status:{f.group(1)}"
        toks = [t for t in c.split() if not t.startswith("-")]
        verb_idx = next((i for i, t in enumerate(toks) if t in ("ps", "stats", "inspect")), -1)
        container = toks[verb_idx + 1] if 0 <= verb_idx + 1 < len(toks) else "all"
        if container.startswith(("{{", "'{{", '"{{')):
            container = "all"
        return f"docker-status:{container}"

    m = re.search(r"ofrep/v1/evaluate/flags/([A-Za-z0-9_.-]+)", c)
    if m:
        return f"ofrep:{m.group(1)}"

    if ":16686" in c:
        m = re.search(r":16686(/\S+)", c)
        return f"jaeger:{m.group(1) if m else '*'}"

    if ":9090" in c:
        m = re.search(r"query=([^'&\s]+)", c)
        return f"prom:{m.group(1) if m else '*'}"

    return "cmd:" + re.sub(r"\s+", " ", c)


def _prune_candidates(steps: list[dict[str, Any]]) -> list[tuple[int, str]]:
    candidates: list[tuple[int, str]] = []
    seen: dict[str, tuple[set[str], set[str]]] = {}
    skeleton_has_data: dict[str, bool] = {}
    for step in steps:
        if step.get("tool_name") != "Bash":
            continue
        sk = _bash_skeleton(str((step.get("tool_input") or {}).get("command", "") or ""))
        if _obs_facts(step.get("observation") or ""):
            skeleton_has_data[sk] = True

    for i, step in enumerate(steps):
        if step.get("hook_decision") == "allow":
            continue
        if step.get("tool_name") == "StructuredOutput" and not is_malformed_step(step):
            continue

        if is_malformed_step(step):
            candidates.append((i, "malformed-structured-output"))
            continue
        if is_noise_step(step):
            candidates.append((i, "noise"))
            continue
        if step.get("is_repeat_no_new_info"):
            candidates.append((i, "marked-repeat"))
            continue

        if step.get("tool_name") == "Bash":
            cmd = str((step.get("tool_input") or {}).get("command", "") or "")
            skeleton = _bash_skeleton(cmd)
            sigs = set(classify_signal_types("Bash", step.get("tool_input") or {}, step.get("observation") or ""))
            obs_tokens = _obs_facts(step.get("observation") or "")
            if not sigs and len(obs_tokens) < 2:
                candidates.append((i, "no-signal-no-fact"))
                continue
            prev_sigs, prev_tokens = seen.get(skeleton, (set(), set()))
            if skeleton in seen and sigs <= prev_sigs and obs_tokens <= prev_tokens:
                candidates.append((i, f"skeleton-repeat:{skeleton}"))
                continue
            if not obs_tokens and skeleton_has_data.get(skeleton):
                candidates.append((i, f"failed-query-retry:{skeleton}"))
                continue
            if skeleton not in seen:
                seen[skeleton] = (sigs, obs_tokens)
            else:
                seen[skeleton] = (prev_sigs | sigs, prev_tokens | obs_tokens)

    return candidates


def verify_pruned(traj: dict[str, Any], pruned_steps: list[dict[str, Any]]) -> list[str]:
    reasons: list[str] = []
    diag = traj.get("final_diagnosis") or {}
    ok, bad = rs.evidence_traceability_ok(diag, pruned_steps)
    if not ok:
        reasons.append(f"evidence 无法追溯: {bad!r}")
    if not rs.executed_actions_match_ledger(pruned_steps, diag.get("executed_actions") or []):
        reasons.append("executed_actions 台账不一致")
    return reasons


def prune_trajectory(traj: dict[str, Any]) -> dict[str, Any]:
    steps = traj.get("steps") or []
    candidates = dict(_prune_candidates(steps))

    kept_idx = [i for i in range(len(steps)) if i not in candidates]

    def _steps_with(restored: list[int]) -> list[dict[str, Any]]:
        return [steps[i] for i in sorted(kept_idx + restored)]

    to_restore: list[int] = []
    if candidates and verify_pruned(traj, _steps_with(to_restore)):
        for cand_idx in sorted(candidates, reverse=True):
            to_restore.append(cand_idx)
            if not verify_pruned(traj, _steps_with(to_restore)):
                break

    final_idx = sorted(kept_idx + to_restore)
    removed_idx = sorted(set(candidates) - set(to_restore))
    removed_records = []
    for i in removed_idx:
        removed_records.append(
            {
                "step_index_1based": i + 1,
                "tool_name": steps[i].get("tool_name"),
                "reason": candidates[i],
                "command_snippet": str((steps[i].get("tool_input") or {}).get("command", ""))[:160],
            }
        )

    out = {k: v for k, v in traj.items() if k != "steps"}
    out["steps"] = [steps[i] for i in final_idx]
    meta = dict(traj.get("meta") or {})
    meta["pruning"] = {
        "steps_before": len(steps),
        "steps_after": len(final_idx),
        "removed": removed_records,
        "restored_by_verification": [
            {"step_index_1based": i + 1, "reason": candidates[i]} for i in sorted(to_restore)
        ],
    }
    out["meta"] = meta
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="对冷启轨迹做步级修剪（C2）")
    parser.add_argument("--traj-dir", type=Path, default=DEFAULT_TRAJ_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stats = []
    for f in sorted(args.traj_dir.glob("*.json")):
        traj = json.loads(f.read_text(encoding="utf-8"))
        pruned = prune_trajectory(traj)
        (args.out_dir / f.name).write_text(
            json.dumps(pruned, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        m = pruned["meta"]["pruning"]
        stats.append(
            {
                "alert_id": traj.get("alert_id") or f.stem,
                "steps_before": m["steps_before"],
                "steps_after": m["steps_after"],
                "removed": len(m["removed"]),
                "restored_by_verification": len(m["restored_by_verification"]),
            }
        )

    total_before = sum(s["steps_before"] for s in stats)
    total_after = sum(s["steps_after"] for s in stats)
    total_removed = sum(s["removed"] for s in stats)
    total_restored = sum(s["restored_by_verification"] for s in stats)
    print(f"[prune_steps] {len(stats)} 条轨迹: {total_before} 步 -> {total_after} 步"
          f"（修剪 {total_removed} 步，安全网回退 {total_restored} 步）")
    if stats:
        print(f"[prune_steps] 均值 {total_before / len(stats):.2f} -> {total_after / len(stats):.2f} 步/条")
    print(f"[prune_steps] 写入 {args.out_dir}")


if __name__ == "__main__":
    main()
