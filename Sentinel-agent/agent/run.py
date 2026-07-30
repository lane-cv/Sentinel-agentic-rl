from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from agent import config
from agent.agents.code_fix import code_fix
from agent.agents.diagnose import diagnose
from agent.core.schema import Diagnosis, FixResult
from agent.integrations import memory
from agent.integrations.feishu import send_feishu_card
from agent.integrations.ingest import alert_fingerprint, load_alert, seen_recently


def _report(
    alert: dict,
    diagnosis: Diagnosis | None,
    *,
    route: str,
    meta: dict[str, Any],
    fix: FixResult | None = None,
    feishu: dict | None = None,
    routing_note: str | None = None,
) -> dict[str, Any]:
    return {
        "alert": alert,
        "route": route,
        "routing_note": routing_note,
        "diagnosis": diagnosis.model_dump(by_alias=True) if diagnosis else None,
        "fix": fix.model_dump() if fix else None,
        "feishu": feishu,
        "meta": meta,
    }


def _auto_note(diag: Diagnosis) -> str:
    acts = "；".join(f"{a.action}→{a.target}" for a in diag.executed_actions)
    return f"已自动执行低风险止血处置（{acts}），本卡片仅知会人工复核。"


async def run(alert: dict) -> dict[str, Any]:
    report = await _run_inner(alert)
    memory.store_ticket(report)
    return report


async def _run_inner(alert: dict) -> dict[str, Any]:
    fp = alert_fingerprint(alert)
    if seen_recently(fp):
        return _report(alert, None, route="skipped_duplicate", meta={"fingerprint": fp})

    dr = await diagnose(alert)
    diag: Diagnosis = dr["diagnosis"]
    meta = dr["meta"]
    meta["fingerprint"] = fp

    if diag.confidence < config.CONFIDENCE_THRESHOLD:
        note = f"置信度 {diag.confidence:.2f} < 阈值 {config.CONFIDENCE_THRESHOLD}，降级人工处理"
        feishu = send_feishu_card(diag, routing_note=note)
        return _report(
            alert, diag, route="feishu_low_confidence", meta=meta,
            feishu=feishu, routing_note=note,
        )

    if diag.remediation_type == "online_op":
        note = ""
        auto_done = bool(diag.executed_actions)
        if auto_done:
            note = _auto_note(diag)
            feishu = send_feishu_card(diag, routing_note=note)
        else:
            feishu = send_feishu_card(diag)

        if diag.also_code_fix:
            fr = await code_fix(diag.model_dump(by_alias=True))
            fix2: FixResult = fr["fix"]
            meta["fix_meta"] = fr["meta"]
            if fix2.verified:
                route = "auto_remediated_and_code_fix_pr" if auto_done else "online_op_and_code_fix_pr"
                note = "混合根因：" + ("已自动止血，" if auto_done else "已发卡提示人工止血，") + "并触发修复 Agent 提 PR 根治。"
            else:
                route = "auto_remediated" if auto_done else "feishu_online_op"
                note = _auto_note(diag) if auto_done else note
            return _report(
                alert, diag, route=route, meta=meta,
                fix=fix2, feishu=feishu, routing_note=note,
            )
        if auto_done:
            return _report(alert, diag, route="auto_remediated", meta=meta,
                           feishu=feishu, routing_note=note)
        return _report(alert, diag, route="feishu_online_op", meta=meta, feishu=feishu)

    if diag.remediation_type == "code_fix":
        fr = await code_fix(diag.model_dump(by_alias=True))
        fix: FixResult = fr["fix"]
        meta["fix_meta"] = fr["meta"]
        if not fix.verified:
            note = "已定位疑似代码根因，但自动修复未通过 build/test 验证，需人工介入。"
            feishu = send_feishu_card(diag, routing_note=note)
            return _report(
                alert, diag, route="feishu_fix_unverified", meta=meta,
                fix=fix, feishu=feishu, routing_note=note,
            )
        return _report(alert, diag, route="code_fix_pr", meta=meta, fix=fix)

    return _report(alert, diag, route="info_only", meta=meta)


def _print_report(report: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print(f"走向 ROUTE: {report['route']}")
    if report.get("routing_note"):
        print(f"路由说明:   {report['routing_note']}")
    diag = report.get("diagnosis")
    if diag:
        print(f"诊断摘要:   {diag['summary']}")
        print(f"疑似服务:   {diag['suspect_service']}  根因类型: {diag['kind']}")
        print(f"置信度:     {diag['confidence']}")
        print(f"处置建议:   {diag['remediation_type']} — {diag['remediation_detail']}")
        acts = diag.get("executed_actions") or []
        if acts:
            print("已自动处置:")
            for a in acts:
                print(f"  - {a['action']} → {a['target']}  ({a['command']})")
    fix = report.get("fix")
    if fix:
        print(f"修复结果:   verified={fix['verified']} pr_url={fix.get('pr_url')} "
              f"degraded={fix.get('degraded')}")
    m = report.get("meta", {})
    print(f"成本:       usage={m.get('usage')} cost_usd={m.get('cost_usd')} "
          f"attempts={m.get('attempts')} degraded={m.get('degraded')}")
    print("=" * 60 + "\n")


async def _main_async(args: argparse.Namespace) -> None:
    alert = load_alert(args.alert)
    report = await run(alert)
    _print_report(report)
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[报告已写入 {args.json_out}]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sentinel agent 故障诊断与修复")
    parser.add_argument("--alert", required=True, help="告警 JSON 路径（alerts/sN.json）")
    parser.add_argument("--json-out", help="可选：把完整报告 JSON 写到该路径")
    args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
