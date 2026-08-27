import json
from pathlib import Path

from agent.agents.prompts import diagnose_prompt


def test_diagnosis_prompt_excludes_fixture_answers():
    alerts_dir = Path(__file__).resolve().parents[1] / "alerts"
    for path in alerts_dir.glob("*.json"):
        alert = json.loads(path.read_text(encoding="utf-8"))
        expected = alert.copy()
        hint = expected.pop("scenario_hint", None)
        prompt = diagnose_prompt(alert)
        payload = prompt.split("```json\n", 1)[1].split("\n```", 1)[0]
        assert json.loads(payload) == expected, path.name
        if hint:
            assert hint not in prompt, path.name
            assert alert["scenario_hint"] == hint
