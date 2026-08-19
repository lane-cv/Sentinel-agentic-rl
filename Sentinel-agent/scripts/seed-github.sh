#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIXTURE="$ROOT/scripts/fixtures/recommendation-master"

PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="python3"

repo_settings="$(cd "$ROOT" && "$PY" - <<'PYCONFIG'
from agent import config

try:
    config.validate_github_repository()
except ValueError as exc:
    raise SystemExit(f"[seed] {exc}")
print(config.GITHUB_UPSTREAM_OWNER, config.GITHUB_REPO, config.GITHUB_DEFAULT_BRANCH)
PYCONFIG
)"
read -r OWNER REPO DEFAULT_BRANCH <<< "$repo_settings"
GIT_EMAIL="${GITHUB_SEED_EMAIL:-aiops@example.com}"
GIT_NAME="${GITHUB_SEED_NAME:-aiops}"
DESCRIPTION="Recommendation service with reproducible faults for testing diagnosis and code fixes. See BUGS.md."

command -v gh >/dev/null || { echo "[seed] gh CLI not found. Install: https://cli.github.com/"; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "[seed] gh not authenticated. Run: gh auth login"; exit 1; }

FULL="$OWNER/$REPO"

echo "[seed] ensuring PUBLIC repo $FULL exists…"
if gh repo view "$FULL" >/dev/null 2>&1; then
  echo "[seed] repo exists; ensuring it is public + description is current…"
  gh repo edit "$FULL" --visibility public --accept-visibility-change-consequences >/dev/null 2>&1 || true
  gh repo edit "$FULL" --description "$DESCRIPTION" >/dev/null 2>&1 || true
else
  gh repo create "$FULL" --public --description "$DESCRIPTION" >/dev/null
  echo "[seed] created $FULL (public)."
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "[seed] building the multi-bug master history (self-check: py_compile + pytest per commit)…"
"$PY" "$ROOT/scripts/seed_github_history.py" \
  --fixture-dir "$FIXTURE" \
  --work-dir "$WORK/repo" \
  --default-branch "$DEFAULT_BRANCH" \
  --service "$REPO" \
  --git-name "$GIT_NAME" --git-email "$GIT_EMAIL" \
  --out "$WORK/manifest.json" \
  --deploys-log "$ROOT/deploys.log"

gh api -X DELETE "repos/$FULL/branches/$DEFAULT_BRANCH/protection" >/dev/null 2>&1 || true
git -C "$WORK/repo" remote add origin "https://github.com/$FULL.git"
git -C "$WORK/repo" -c credential.helper='!gh auth git-credential' push -q --force -u origin "$DEFAULT_BRANCH"

echo "[seed] enabling '$DEFAULT_BRANCH' branch protection (PRs only, enforced for admins too)…"
gh api -X PUT "repos/$FULL/branches/$DEFAULT_BRANCH/protection" \
  -H "Accept: application/vnd.github+json" \
  --input - >/dev/null 2>&1 <<'JSON' || \
  echo "[seed] (branch protection API returned non-2xx; re-run or set it in repo Settings > Branches)"
{
  "required_status_checks": null,
  "enforce_admins": true,
  "required_pull_request_reviews": { "required_approving_review_count": 0 },
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false
}
JSON

if rm -rf "$ROOT/workspace/$REPO" && git clone -q --depth 50 "https://github.com/$FULL.git" "$ROOT/workspace/$REPO"; then
  echo "[seed] workspace/$REPO refreshed to the new master."
else
  echo "[seed] WARN: could not refresh workspace/$REPO (diagnose Agent will read a stale clone)."
fi

echo ""
echo "[seed] DONE. master layout (all bugs live on master, one commit each):"
"$PY" - "$WORK/manifest.json" <<'PY'
import json, sys
for e in json.load(open(sys.argv[1])):
    note = f"  -> {e['deploy_note']}" if e["deploy_note"] else ""
    print(f"  {e['sha']}  {e['subject']}{note}")
PY
echo ""
echo "  upstream repo:  https://github.com/$FULL"
echo "  Known issues: BUGS.md"
echo "  Deploy history: $ROOT/deploys.log"
echo "  Same-repository PR: leave GITHUB_FORK_OWNER empty in .env."
echo "  Cross-fork PR: fork https://github.com/$FULL and set GITHUB_FORK_OWNER in .env."
