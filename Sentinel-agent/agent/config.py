from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = REPO_ROOT / "workspace"

load_dotenv(REPO_ROOT / ".env")

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))

REMEDIATION_ENABLED = os.getenv("AIOPS_REMEDIATION_ENABLED", "1") == "1"
REMEDIATION_BACKEND = os.getenv("AIOPS_BACKEND", "docker").strip().lower()
K8S_NAMESPACE = os.getenv("AIOPS_K8S_NAMESPACE", "otel-demo")
REMEDIATION_ALLOWED_TARGETS = os.getenv(
    "AIOPS_REMEDIATION_TARGETS", "recommendation,ad,frontend,cart,checkout,currency,payment,shipping,quote,email"
)

DEDUP_TTL_SECONDS = int(os.getenv("DEDUP_TTL_SECONDS", str(30 * 60)))

DIAGNOSE_MAX_TURNS = int(os.getenv("DIAGNOSE_MAX_TURNS", "50"))
DIAGNOSE_TIMEOUT_S = int(os.getenv("DIAGNOSE_TIMEOUT_S", "600"))
FIX_MAX_TURNS = int(os.getenv("FIX_MAX_TURNS", "50"))
FIX_TIMEOUT_S = int(os.getenv("FIX_TIMEOUT_S", "900"))
SCHEMA_RETRY_MAX = int(os.getenv("SCHEMA_RETRY_MAX", "2"))

MODEL = os.getenv("AIOPS_MODEL", "opus")

LLM_BASE_URL = os.getenv("AIOPS_LLM_BASE_URL", "http://localhost:8000")
LLM_API_KEY = os.getenv("AIOPS_LLM_API_KEY", "placeholder-key")
LLM_AUTH_TOKEN = os.getenv("AIOPS_LLM_AUTH_TOKEN", "placeholder-token")

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
JAEGER_URL = os.getenv("JAEGER_URL", "http://localhost:16686")
FLAGD_OFREP_URL = os.getenv("FLAGD_OFREP_URL", "http://localhost:8016")

FEISHU_WEBHOOK_URL = os.getenv("FEISHU_WEBHOOK_URL", "")

GITHUB_API_BASE = os.getenv("GITHUB_API_BASE", "https://api.github.com")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_UPSTREAM_OWNER = os.getenv("GITHUB_UPSTREAM_OWNER", "").strip()
GITHUB_REPO = os.getenv("GITHUB_REPO", "").strip()
GITHUB_FORK_OWNER = os.getenv("GITHUB_FORK_OWNER", "").strip() or GITHUB_UPSTREAM_OWNER
GITHUB_DEFAULT_BRANCH = os.getenv("GITHUB_DEFAULT_BRANCH", "master").strip() or "master"

RAG_ENABLED = os.getenv("AIOPS_RAG_ENABLED", "1") == "1"
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "aiops_tickets")
EMBEDDING_MODEL = os.getenv("AIOPS_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "512"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))


def validate_github_repository() -> None:
    for key, value in (
        ("GITHUB_UPSTREAM_OWNER", GITHUB_UPSTREAM_OWNER),
        ("GITHUB_REPO", GITHUB_REPO),
        ("GITHUB_FORK_OWNER", GITHUB_FORK_OWNER),
    ):
        if not value:
            raise ValueError(f"请在 .env 中设置 {key}")
        if value in (".", "..") or not re.fullmatch(r"[a-zA-Z0-9_.-]+", value):
            raise ValueError(f"{key} 应填写单个 GitHub 名称，请检查 .env")
