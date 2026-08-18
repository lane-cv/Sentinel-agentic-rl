#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

FEATURES = ["ranking", "dedupe", "stats", "pagination", "cache", "pricing", "leak"]

OPEN_RE = re.compile(r"^\s*<<f:(\w+)>>\s*$")
CLOSE_RE = re.compile(r"^\s*<</f:(\w+)>>\s*$")
OPEN_NEG_RE = re.compile(r"^\s*<<!f:(\w+)>>\s*$")
CLOSE_NEG_RE = re.compile(r"^\s*<</!f:(\w+)>>\s*$")

FAILED_RE = re.compile(r"^FAILED\s+(\S+?)(?:\s+-.*)?$", re.MULTILINE)


def preprocess(text: str, active: set) -> str:
    lines = text.split("\n")
    out: list = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m_open = OPEN_RE.match(line) or OPEN_NEG_RE.match(line)
        if not m_open:
            out.append(line)
            i += 1
            continue
        name = m_open.group(1)
        neg = bool(OPEN_NEG_RE.match(line))
        close = CLOSE_NEG_RE if neg else CLOSE_RE
        j = i + 1
        while j < len(lines) and not close.match(lines[j]):
            j += 1
        if j >= len(lines):
            raise ValueError(f"unclosed feature block <<{'!' if neg else ''}f:{name}>>")
        keep = (name not in active) if neg else (name in active)
        if keep:
            out.extend(lines[i + 1 : j])
        i = j + 1
    rendered = "\n".join(out)
    for stray in (OPEN_RE, CLOSE_RE, OPEN_NEG_RE, CLOSE_NEG_RE):
        if stray.search(rendered):
            raise ValueError(f"stray marker left after preprocessing: {stray.pattern}")
    rendered = re.sub(r"\n{3,}", "\n\n", rendered)
    return rendered.rstrip("\n") + "\n"


def _run(cmd, cwd=None, env=None, check=True):
    proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}"
        )
    return proc


def _git(workdir, *args, env=None):
    return _run(["git", "-C", str(workdir), *args], env=env)


def _short_sha(workdir):
    return _git(workdir, "rev-parse", "--short", "HEAD").stdout.strip()


def _failing_tests(workdir):
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = _run(
        [sys.executable, "-m", "pytest", "-q", "--tb=no", "-p", "no:cacheprovider"],
        cwd=str(workdir), env=env, check=False,
    )
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"pytest could not run:\n{proc.stdout}\n{proc.stderr}")
    return set(FAILED_RE.findall(proc.stdout))


def _ts_git(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%S %z")


def stage_table():
    return [
        dict(
            stage="baseline", active=[], deploy_note="baseline",
            subject="recommendation service baseline (bounded recent-id cache)",
            body=None, files={"test_recommendation.py": "test_recommendation.py",
                              "requirements.txt": "requirements.txt",
                              "Dockerfile": "Dockerfile",
                              "README.stage1.md": "README.md", ".gitignore": ".gitignore"},
            expect_failed=set(),
        ),
        dict(
            stage="ranking", active=["ranking"], deploy_note="popularity ranking rollout",
            subject="feat: rank recommendations by popularity",
            body="Surface the most-popular catalog items first instead of arbitrary catalog order.",
            files={"ranking.py": "ranking.py", "test_ranking.py": "test_ranking.py"},
            expect_failed={"test_ranking.py::test_rank_by_score_orders_most_popular_first"},
        ),
        dict(
            stage="dedupe", active=["ranking", "dedupe"], deploy_note="trending + dedupe rollout",
            subject="feat: merge merchandising trending ids into recommendations",
            body="Trending ids can overlap with the catalog match set, so run both sources "
                 "through a dedupe step; a response must also never echo back an id the "
                 "caller already has.",
            files={"dedupe.py": "dedupe.py", "test_dedupe.py": "test_dedupe.py"},
            expect_failed={"test_dedupe.py::test_dedupe_removes_exact_duplicates"},
        ),
        dict(
            stage="stats", active=["ranking", "dedupe", "stats"], deploy_note="category hit stats rollout",
            subject="feat: track per-category recommendation hits",
            body="A plain in-process counter per category, so we can see which category gets "
                 "recommended most without a metrics backend.",
            files={"stats.py": "stats.py", "test_stats.py": "test_stats.py"},
            expect_failed={"test_stats.py::test_record_hit_is_thread_safe"},
        ),
        dict(
            stage="pagination", active=["ranking", "dedupe", "stats", "pagination"],
            deploy_note="catalog pagination rollout",
            subject="feat: paginated catalog browsing",
            body="Serve the catalog in fixed-size pages so /catalog stays constant-size as "
                 "the catalog grows.",
            files={"pagination.py": "pagination.py", "test_pagination.py": "test_pagination.py"},
            expect_failed={"test_pagination.py::test_first_page_starts_at_the_beginning",
                           "test_pagination.py::test_pages_cover_all_items_without_gaps"},
        ),
        dict(
            stage="cache", active=["ranking", "dedupe", "stats", "pagination", "cache"],
            deploy_note="trending cache rollout",
            subject="feat: cache the trending list with a TTL",
            body="The trending list is derived data; cache it for 60s so /trending stays cheap.",
            files={"cache.py": "cache.py", "test_cache.py": "test_cache.py"},
            expect_failed={"test_cache.py::test_entry_expires_after_ttl"},
        ),
        dict(
            stage="pricing", active=FEATURES[:-1], deploy_note="member pricing rollout",
            subject="feat: member-price display for recommended products",
            body="Show list price and discounted member price in cents; round half-up to the "
                 "cent per price-display policy.",
            files={"pricing.py": "pricing.py", "test_pricing.py": "test_pricing.py"},
            expect_failed={"test_pricing.py::test_member_price_rounds_half_up"},
        ),
        dict(
            stage="leak", active=list(FEATURES), deploy_note="LATEST_DEPLOY (suspect)",
            subject="perf: track all seen product ids for analytics",
            body="Collect every requested product id so we can compute popularity later.",
            files={},
            expect_failed={"test_recommendation.py::test_memory_is_bounded"},
        ),
        dict(
            stage="docs", active=list(FEATURES), deploy_note=None,
            subject="docs: start a known-issues list",
            body="Track the open issues QA and ops have reported so they stop getting lost in chat.",
            files={"BUGS.md": "BUGS.md", "README.md": "README.md"},
            expect_failed=set(),
        ),
    ]


GITIGNORE = "__pycache__/\n*.pyc\n.pytest_cache/\n"


def build(fixture: Path, workdir: Path, git_name: str, git_email: str,
          default_branch: str, service: str, run_tests: bool,
          out: Path | None, deploys_log: Path | None) -> list:
    if workdir.exists() and any(workdir.iterdir()):
        raise SystemExit(f"[history] work dir not empty: {workdir}")
    workdir.mkdir(parents=True, exist_ok=True)

    template = (fixture / "recommendation_server.stages.py.in").read_text(encoding="utf-8")
    final_server = (fixture / "recommendation_server.py").read_text(encoding="utf-8")
    rendered_final = preprocess(template, set(FEATURES))
    if rendered_final != final_server:
        raise SystemExit(
            "[history] recommendation_server.stages.py.in (all-features rendering) no longer "
            "matches recommendation_server.py. Regenerate it:\n"
            "  python3 scripts/seed_github_history.py --emit-final-server "
            "scripts/fixtures/recommendation-master/recommendation_server.py\n"
            "then review the diff."
        )

    _git(workdir, "init", "-q", "-b", default_branch)
    stages = stage_table()
    now = datetime.now(timezone.utc)
    cumulative_failed: set = set()
    manifest = []

    for idx, st in enumerate(stages, start=1):
        active = set(st["active"])
        (workdir / "recommendation_server.py").write_text(
            preprocess(template, active), encoding="utf-8")
        for src, dst in st["files"].items():
            if src == ".gitignore":
                (workdir / dst).write_text(GITIGNORE, encoding="utf-8")
            else:
                (workdir / dst).write_text(
                    (fixture / src).read_text(encoding="utf-8"), encoding="utf-8")

        _run([sys.executable, "-m", "py_compile", "recommendation_server.py"], cwd=str(workdir))

        if run_tests:
            if st["stage"] == "docs":
                expected = set(cumulative_failed)
            else:
                cumulative_failed |= st["expect_failed"]
                expected = set(cumulative_failed)
            got = _failing_tests(workdir)
            if got != expected:
                raise SystemExit(
                    f"[history] stage '{st['stage']}' failing-test mismatch:\n"
                    f"  expected: {sorted(expected)}\n"
                    f"  got:      {sorted(got)}\n"
                    "bugs must coexist without interfering — fix the fixture before seeding."
                )
        elif st["stage"] != "docs":
            cumulative_failed |= st["expect_failed"]

        ts = now - timedelta(days=len(stages) - idx)
        env = {
            "GIT_AUTHOR_NAME": git_name, "GIT_AUTHOR_EMAIL": git_email,
            "GIT_COMMITTER_NAME": git_name, "GIT_COMMITTER_EMAIL": git_email,
            "GIT_AUTHOR_DATE": _ts_git(ts), "GIT_COMMITTER_DATE": _ts_git(ts),
        }
        _git(workdir, "add", "-A")
        msg = ["-m", st["subject"]]
        if st["body"]:
            msg += ["-m", st["body"]]
        _git(workdir, "commit", "-q", *msg, env=env)
        sha = _short_sha(workdir)
        manifest.append({
            "stage": st["stage"], "sha": sha,
            "full_sha": _git(workdir, "rev-parse", "HEAD").stdout.strip(),
            "subject": st["subject"], "deploy_note": st["deploy_note"],
            "ts": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
        print(f"[history] {idx}/{len(stages)} {sha}  {st['subject']}")

    if out:
        out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if deploys_log:
        lines = ["# service deploy history (image tag = git SHA)"]
        for e in manifest:
            if e["deploy_note"]:
                lines.append(
                    f"{e['ts']}  {service}  image={service}:{e['sha']}  "
                    f"sha={e['sha']}  note={e['deploy_note']}"
                )
        deploys_log.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[history] deploys.log written to {deploys_log}")

    expected_files = {
        "recommendation_server.py", "requirements.txt", "Dockerfile",
        "README.md", "BUGS.md", ".gitignore",
        *{dst for st in stages for dst in st["files"].values()},
    }
    got_files = {p.name for p in workdir.iterdir()} - {"__pycache__", ".git"}
    if got_files != expected_files:
        raise SystemExit(
            f"[history] final file set mismatch:\n"
            f"  missing: {sorted(expected_files - got_files)}\n"
            f"  extra:   {sorted(got_files - expected_files)}"
        )
    print(f"[history] OK — {len(manifest)} commits, "
          f"{len(cumulative_failed)} coexisting bug regression tests red at HEAD.")
    return manifest


def main():
    ap = argparse.ArgumentParser(description='Build the local git history for the shared GitHub fixture repo — master layout.')
    ap.add_argument("--fixture-dir", default=None,
                    help="default: <repo>/scripts/fixtures/recommendation-master")
    ap.add_argument("--work-dir", required=False, default=None,
                    help="empty dir to build the git history in (default: mktemp)")
    ap.add_argument("--default-branch", default="master")
    ap.add_argument("--service", default="recommendation")
    ap.add_argument("--git-name", default="aiops")
    ap.add_argument("--git-email", default="aiops@example.com")
    ap.add_argument("--skip-tests", action="store_true",
                    help="skip the per-stage pytest self-check (not recommended)")
    ap.add_argument("--out", help="write the commit manifest JSON here")
    ap.add_argument("--deploys-log", help="write deploys.log content here")
    ap.add_argument("--emit-final-server", metavar="PATH",
                    help="render the final server file to PATH and exit "
                         "(regeneration helper for recommendation_server.py)")
    args = ap.parse_args()

    default_fixture = Path(__file__).resolve().parent / "fixtures" / "recommendation-master"
    fixture = Path(args.fixture_dir) if args.fixture_dir else default_fixture

    if args.emit_final_server:
        template = (fixture / "recommendation_server.stages.py.in").read_text(encoding="utf-8")
        Path(args.emit_final_server).write_text(
            preprocess(template, set(FEATURES)), encoding="utf-8")
        print(f"[history] final server rendered to {args.emit_final_server}")
        return

    if args.work_dir:
        workdir = Path(args.work_dir)
    else:
        import tempfile
        workdir = Path(tempfile.mkdtemp(prefix="rec-history-"))
        print(f"[history] building in {workdir} (rm it when done)")

    build(fixture, workdir, args.git_name, args.git_email,
          args.default_branch, args.service,
          run_tests=not args.skip_tests,
          out=Path(args.out) if args.out else None,
          deploys_log=Path(args.deploys_log) if args.deploys_log else None)


if __name__ == "__main__":
    main()
