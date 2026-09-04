from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

CLEAN_DIR = Path(__file__).resolve().parent.parent
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

if str(CLEAN_DIR) not in sys.path:
    sys.path.insert(0, str(CLEAN_DIR))


def load_fixture(name: str) -> dict:
    if not name.endswith(".json"):
        name = f"{name}.json"
    with open(FIXTURES_DIR / name, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_fixtures() -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(FIXTURES_DIR.glob("*.json"))}


@pytest.fixture
def fixtures() -> dict[str, dict]:
    return load_all_fixtures()
