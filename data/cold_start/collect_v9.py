from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import collect_trajectories as ct
import fault_injection as fi
import rejection_sample as rs

REPO_ROOT = HERE.parent.parent
SEEDS_DIR = REPO_ROOT / "data" / "seeds" / "generated"
GT_DIR = HERE / "ground_truth"
AUG_MAP_PATH = SEEDS_DIR / "_v9_augmentation_map.json"
DEFAULT_OUT_DIR = HERE / "trajectories_v9_raw"
PROGRESS_MD = HERE / "_collection_v9_progress.md"
STATE_JSON = HERE / "_collection_v9_state.json"
AIOPS_AGENT_ROOT = REPO_ROOT / "Sentinel-agent"
RESET_SH = AIOPS_AGENT_ROOT / "scripts" / "reset.sh"

_ALL_FLAGD_FLAGS = [
    "productCatalogFailure", "recommendationCacheFailure", "adManualGc", "adHighCpu",
    "adFailure", "kafkaQueueProblems", "cartFailure", "paymentFailure",
    "paymentUnreachable", "loadGeneratorFloodHomepage", "imageSlowLoad",
]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ofrep_variant(flag: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["curl", "-s", "-m", "5", "-X", "POST",
             f"http://localhost:8016/ofrep/v1/evaluate/flags/{flag}",
             "-H", "Content-Type: application/json", "-d", "{}"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        return json.loads(out).get("variant")
    except Exception:
        return None


def _all_flags_off() -> list[str]:
    return [f for f in _ALL_FLAGD_FLAGS if _ofrep_variant(f) not in ("off", None)]


def _load_state() -> dict[str, Any]:
    if STATE_JSON.exists():
        return json.loads(STATE_JSON.read_text(encoding="utf-8"))
    return {"runs": [], "total_cost_usd": 0.0}


def _save_state(state: dict[str, Any]) -> None:
    STATE_JSON.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _rewrite_progress_md(state)


def _rewrite_progress_md(state: dict[str, Any]) -> None:
    lines: list[str] = []
    ap = lines.append
    ap("# v9 数据增强采集进度（Phase C）")
    ap("")
    ap(f"- 更新时间：{_now()}（全部时间 UTC）")
    ap(f"- 教师模型：`{state.get('teacher_model', '?')}`；后端：`{state.get('llm_base_url', '?')}`")
    ap(f"- 累计花费：**${state.get('total_cost_usd', 0.0):.2f}**（预算上限 $60，$50 停机汇报）")
    ap(f"- 采集轮次统计：见下表（round 列）")
    ap("")
    ap("| 种子 | 家族 | 轮次 | 步数 | degraded | 验收 | 拒绝原因 | cost_usd | 累计$ | 时间 |")
    ap("|---|---|---|---|---|---|---|---|---|---|")
    cum = 0.0
    for r in state["runs"]:
        cum += r.get("cost_usd") or 0.0
        ap(
            f"| `{r['stem']}` | {r.get('family', '?')} | {r.get('round', '?')} "
            f"| {r.get('steps', '?')} | {r.get('degraded') or '-'} "
            f"| {r.get('verdict', '?')} | {str(r.get('reject_reasons') or '')[:120].replace('|', '/')} "
            f"| {r.get('cost_usd') or 0:.4f} | {cum:.2f} | {r.get('ts', '')} |"
        )
    ap("")
    accepted = [r for r in state["runs"] if r.get("verdict") == "ACCEPTED"]
    rejected = [r for r in state["runs"] if r.get("verdict") == "REJECTED"]
    ap(f"**已验收通过 {len(accepted)} 条 / 已拒绝 {len(rejected)} 条**（未跑的不计）")
    PROGRESS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def preflight() -> dict[str, Any]:
    cfg = fi.discover_flagd_config()
    checks: dict[str, Any] = {"flagd_config": str(cfg) if cfg else None, "ok": True, "errors": []}

    import os
    model = os.environ.get("AIOPS_MODEL", "")
    base = os.environ.get("AIOPS_LLM_BASE_URL", "")
    checks["aiops_model"] = model
    checks["aiops_llm_base_url"] = base
    if not model or "qwen" in model.lower():
        checks["errors"].append(
            f"AIOPS_MODEL={model!r} 不是 Opus 教师——请 export AIOPS_MODEL=<真实 Opus 模型名> 再跑"
        )
    if not base or "localhost:8000" in base or "127.0.0.1:8000" in base:
        checks["errors"].append(
            f"AIOPS_LLM_BASE_URL={base!r} 指向已关机的本地 vLLM——请映射真实代理 ANTHROPIC_BASE_URL"
        )
    if not os.environ.get("AIOPS_LLM_AUTH_TOKEN"):
        checks["errors"].append("AIOPS_LLM_AUTH_TOKEN 未设置（真实代理鉴权需要）")
    if not os.environ.get("GITHUB_TOKEN"):
        checks["errors"].append("GITHUB_TOKEN 未设置（D 家族读 GitHub master 的 workspace 克隆时可能用到）")

    if cfg is None:
        checks["errors"].append("找不到活的 flagd 容器/挂载配置——docker 环境没起？")
    else:
        flags_state = json.loads(cfg.read_text(encoding="utf-8"))["flags"]
        disabled = [k for k, v in flags_state.items() if v.get("state") != "ENABLED"]
        if disabled:
            checks["errors"].append(f"flagd 配置里有 state!=ENABLED 的开关（历史事故形态）: {disabled}")
        not_off = _all_flags_off()
        if not_off:
            checks["errors"].append(f"基线不是全 off（会被上一轮残留污染）: {not_off}")

    if not AUG_MAP_PATH.exists():
        checks["errors"].append(
            f"缺少 {_v9_map_rel()}——先跑 generate_seeds.py --v9-augmentation 生成 47 条种子"
        )

    if checks["errors"]:
        checks["ok"] = False
    return checks


def _v9_map_rel() -> str:
    return str(AUG_MAP_PATH.relative_to(REPO_ROOT))


def collect_one(
    stem: str,
    info: dict[str, Any],
    *,
    out_dir: Path,
    round_no: int,
    state: dict[str, Any],
    force: bool = False,
) -> dict[str, Any]:
    out_path = out_dir / f"{stem}.json"
    if out_path.exists() and not force:
        print(f"[skip] {stem} 已有输出 {out_path}（--force 可重采）")
        return {"stem": stem, "skipped": True}

    seed_path = SEEDS_DIR / f"{stem}.json"
    alert = json.loads(seed_path.read_text(encoding="utf-8"))
    cfg = fi.discover_flagd_config()

    inj = info.get("injection") or {"mode": "none"}

    run: dict[str, Any] = {
        "stem": stem, "family": info.get("family"), "round": round_no,
        "ts": _now(), "injection": inj.get("mode"),
    }
    t0 = time.time()
    try:
        if inj.get("mode") == "pre_inject":
            flag = inj["flag"]
            print(f"[pre-inject] {stem}: 注入 {flag} {inj['on_s']}s（留下真实历史痕迹）…")
            assert cfg is not None
            touched = fi.activate_flags(cfg, [flag])
            assert touched, f"pre-inject 失败：{flag} 不在 flagd 配置里"
            time.sleep(1.2)
            v = _ofrep_variant(flag)
            assert v not in (None, "off"), f"pre-inject 后 OFREP 校验失败: {flag}={v}"
            print(f"[pre-inject] {flag} OFREP variant={v}，保持 {inj['on_s']}s…")
            time.sleep(inj["on_s"])
            fi.reset_all_flags(cfg)
            time.sleep(1.2)
            v2 = _ofrep_variant(flag)
            assert v2 == "off", f"复位后 OFREP 校验失败: {flag}={v2}"
            print(f"[pre-inject] {flag} 已复位（off），消化等待 {inj['drain_s']}s…")
            time.sleep(inj["drain_s"])
        elif inj.get("mode") == "flags":
            touched = fi.inject_alert_fault(alert, cfg) if cfg else []
            if touched:
                time.sleep(1.2)
                bad = [f for f in touched if _ofrep_variant(f) in (None, "off")]
                assert not bad, f"注入后 OFREP 校验失败（未生效）: {bad}"
                print(f"[inject] {stem}: {touched} 已注入且 OFREP 验证生效")

        try:
            traj = asyncio.run(ct._run_diagnose_capturing_steps(alert))
        finally:
            if cfg is not None:
                fi.reset_all(cfg)
        leftover = _all_flags_off()
        if leftover:
            print(f"[warn] {stem}: 复位后仍有开关未关: {leftover}（已再次复位）")
            fi.reset_all_flags(cfg)

        traj["alert_id"] = stem
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(traj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        meta = traj.get("meta") or {}
        run.update(
            steps=len(traj.get("steps") or []),
            degraded=meta.get("degraded"),
            cost_usd=meta.get("cost_usd") or 0.0,
            usage=meta.get("usage"),
            teacher_model=meta.get("teacher_model"),
            duration_s=round(time.time() - t0, 1),
        )
        if meta.get("cost_usd") is None:
            print(f"[warn] {stem}: meta.cost_usd=None（代理未回传计价？记账按 0，注意核对）")

        gt_path = GT_DIR / f"{stem}.json"
        if gt_path.exists():
            gt = json.loads(gt_path.read_text(encoding="utf-8"))
            verdict = rs.check_trajectory(traj, gt)
            run["verdict"] = "ACCEPTED" if verdict["keep"] else "REJECTED"
            run["reject_reasons"] = verdict["reasons"]
        else:
            run["verdict"] = "NO_GT"
            run["reject_reasons"] = [f"missing GT: {gt_path}"]
    except Exception as exc:
        run["verdict"] = "ERROR"
        run["reject_reasons"] = [f"{type(exc).__name__}: {exc}"]
        print(f"[error] {stem}: {type(exc).__name__}: {exc}")
        if cfg is not None:
            try:
                fi.reset_all(cfg)
            except Exception:
                pass

    state["runs"].append(run)
    state["total_cost_usd"] = round(state.get("total_cost_usd", 0.0) + (run.get("cost_usd") or 0.0), 4)
    _save_state(state)
    print(
        f"[done] {stem} round={round_no} steps={run.get('steps')} degraded={run.get('degraded')} "
        f"verdict={run.get('verdict')} cost=${run.get('cost_usd') or 0:.4f} "
        f"累计=${state['total_cost_usd']:.2f} ({run.get('duration_s')}s)"
    )
    if run.get("reject_reasons"):
        for r in run["reject_reasons"]:
            print(f"        reject: {r}")
    return run


def load_aug_map() -> dict[str, dict[str, Any]]:
    return json.loads(AUG_MAP_PATH.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="v9 数据增强采集编排器（Phase C）")
    parser.add_argument("--preflight-only", action="store_true", help="只跑前置检查")
    parser.add_argument("--only", type=str, default=None, help="只跑一个种子（试跑验证用）")
    parser.add_argument("--stems", type=str, default=None, help="逗号分隔的种子列表")
    parser.add_argument("--all", action="store_true", help="按 map 顺序跑全部（已存在输出则跳过）")
    parser.add_argument("--round", type=int, default=1, help="轮次编号（写进进度，B1a 第 2/3 轮用 2/3）")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR / "round1",
                        help="轨迹输出目录（默认 trajectories_v9_raw/round<N> 由 --round 决定）")
    parser.add_argument("--force", action="store_true", help="输出已存在也重采")
    parser.add_argument("--budget-stop", type=float, default=50.0, help="累计花费达到该值立即停机（默认 50）")
    args = parser.parse_args()

    checks = preflight()
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    if not checks["ok"]:
        print("[preflight] 未通过，退出。")
        sys.exit(2)
    if args.preflight_only:
        return

    import os
    aug = load_aug_map()
    state = _load_state()
    state["teacher_model"] = os.environ.get("AIOPS_MODEL", "?")
    state["llm_base_url"] = os.environ.get("AIOPS_LLM_BASE_URL", "?")

    if args.only:
        stems = [args.only]
    elif args.stems:
        stems = [s.strip() for s in args.stems.split(",") if s.strip()]
    elif args.all:
        stems = list(aug.keys())
    else:
        print("请指定 --only / --stems / --all 之一")
        sys.exit(2)

    unknown = [s for s in stems if s not in aug]
    if unknown:
        print(f"[abort] 这些种子不在 v9 增强映射里: {unknown}")
        sys.exit(2)

    out_dir = args.out_dir if str(args.out_dir) != str(DEFAULT_OUT_DIR / "round1") else DEFAULT_OUT_DIR / f"round{args.round}"
    print(f"[collect_v9] 本轮 {len(stems)} 条，输出目录 {out_dir}，轮次 round{args.round}")

    for stem in stems:
        if state.get("total_cost_usd", 0.0) >= args.budget_stop:
            print(f"[budget] 累计花费 ${state['total_cost_usd']:.2f} 已达停机线 ${args.budget_stop}，停止采集并汇报。")
            break
        collect_one(stem, aug[stem], out_dir=out_dir, round_no=args.round, state=state, force=args.force)

    print(f"[collect_v9] 结束。累计 ${state['total_cost_usd']:.2f}；进度见 {PROGRESS_MD}")


if __name__ == "__main__":
    main()
