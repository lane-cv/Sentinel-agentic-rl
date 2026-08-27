from __future__ import annotations

from typing import Any


def _service_ok(exp: dict, diag: dict) -> bool:
    suspect = (diag.get("suspect_service", "") or "").lower()
    if "suspect_service_contains_any_of" in exp:
        return any(n.lower() in suspect for n in exp["suspect_service_contains_any_of"])
    needle = exp.get("suspect_service_contains")
    if not needle:
        return True
    return needle.lower() in suspect


def _kind_ok(exp: dict, diag: dict) -> bool:
    if "kind" in exp:
        return diag.get("kind") == exp["kind"]
    if "kind_any_of" in exp:
        return diag.get("kind") in exp["kind_any_of"]
    return True


def _route_ok(exp: dict, route: str | None) -> bool:
    if "expect_route" in exp:
        return route == exp["expect_route"]
    if "expect_route_any_of" in exp:
        return route in exp["expect_route_any_of"]
    return True


def _changed_files_hit(exp: dict, fix: dict | None) -> bool | None:
    gold = exp.get("expect_changed_files_any_of")
    if not gold:
        return None
    if not fix:
        return False
    changed = fix.get("changed_files") or []
    if not changed:
        return False
    gold_set = {g.lower() for g in gold}
    for cf in changed:
        cf_l = (cf or "").lower()
        for g in gold_set:
            if g in cf_l or cf_l.endswith(g) or g.endswith(cf_l):
                return True
    return False


def extract(scenario: str, exp: dict, report: dict) -> dict[str, Any]:
    diag = report.get("diagnosis") or {}
    fix = report.get("fix")
    meta = report.get("meta") or {}
    fix_meta = meta.get("fix_meta") or {}

    diag_latency = float(meta.get("latency_s") or 0.0)
    fix_latency = float(fix_meta.get("latency_s")) if fix_meta.get("latency_s") is not None else None
    total_latency = diag_latency + (fix_latency or 0.0)

    triggered_code_fix = fix is not None
    fix_verified: bool | None = None
    pr_created: bool | None = None
    if triggered_code_fix:
        fix_verified = bool(fix.get("verified"))
        pr_created = bool(fix.get("pr_url"))

    return {
        "scenario": scenario,
        "route": report.get("route"),
        "service_ok": _service_ok(exp, diag),
        "kind_ok": _kind_ok(exp, diag),
        "route_ok": _route_ok(exp, report.get("route")),
        "triggered_code_fix": triggered_code_fix,
        "fix_verified": fix_verified,
        "changed_files_hit": _changed_files_hit(exp, fix),
        "diag_latency_s": round(diag_latency, 3),
        "fix_latency_s": round(fix_latency, 3) if fix_latency is not None else None,
        "total_latency_s": round(total_latency, 3),
        "pr_created": pr_created,
    }
