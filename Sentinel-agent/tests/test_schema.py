from __future__ import annotations

from agent.core.schema import DIAGNOSIS_JSON_SCHEMA, Diagnosis

BASE_DIAG_KWARGS = {
    "summary": "product-catalog 依赖下游超时",
    "kind": "dependency",
    "suspect_service": "product-catalog",
    "remediation_type": "code_fix",
    "remediation_detail": "需修复下游调用超时处理逻辑。",
    "confidence": 0.8,
    "evidence": ["promql: xxx", "trace: yyy"],
}


def test_diagnosis_without_new_fields_still_valid():
    diag = Diagnosis(**BASE_DIAG_KWARGS)

    assert diag.suspect_repo is None
    assert diag.suspect_commit_hint is None
    assert diag.suspect_file_hint is None

    dumped = diag.model_dump()
    assert dumped["suspect_repo"] is None
    assert dumped["suspect_commit_hint"] is None
    assert dumped["suspect_file_hint"] is None

    roundtripped = Diagnosis.model_validate(dumped)
    assert roundtripped == diag


def test_diagnosis_with_new_fields_roundtrips():
    kwargs = {
        **BASE_DIAG_KWARGS,
        "suspect_repo": "octo-org/product-catalog",
        "suspect_commit_hint": "a1b2c3d",
        "suspect_file_hint": "src/product_catalog_server.py",
        "also_code_fix": False,
    }
    diag = Diagnosis.model_validate(kwargs)

    assert diag.suspect_repo == "octo-org/product-catalog"
    assert diag.suspect_commit_hint == "a1b2c3d"
    assert diag.suspect_file_hint == "src/product_catalog_server.py"

    dumped = diag.model_dump(by_alias=True)
    roundtripped = Diagnosis.model_validate(dumped)
    assert roundtripped == diag


def test_fallback_still_constructs_without_new_fields():
    diag = Diagnosis.fallback("测试降级原因")

    assert diag.suspect_repo is None
    assert diag.suspect_commit_hint is None
    assert diag.suspect_file_hint is None
    assert diag.confidence == 0.0


def test_json_schema_has_new_optional_properties():
    assert DIAGNOSIS_JSON_SCHEMA["type"] == "object"
    assert DIAGNOSIS_JSON_SCHEMA["additionalProperties"] is False

    properties = DIAGNOSIS_JSON_SCHEMA["properties"]
    required = DIAGNOSIS_JSON_SCHEMA["required"]

    for field in ("suspect_repo", "suspect_commit_hint", "suspect_file_hint"):
        assert field in properties, f"{field} missing from DIAGNOSIS_JSON_SCHEMA properties"
        assert properties[field] == {"type": ["string", "null"]}
        assert field not in required, f"{field} must stay optional (not required)"

    assert required == [
        "summary",
        "kind",
        "suspect_service",
        "remediation_type",
        "remediation_detail",
        "confidence",
        "evidence",
    ]


def test_json_schema_is_valid_jsonschema():
    try:
        import jsonschema
    except ImportError:
        return

    jsonschema.Draft7Validator.check_schema(DIAGNOSIS_JSON_SCHEMA)

    sample_without_new_fields = dict(BASE_DIAG_KWARGS)
    jsonschema.validate(sample_without_new_fields, DIAGNOSIS_JSON_SCHEMA)

    sample_with_new_fields = {
        **BASE_DIAG_KWARGS,
        "suspect_repo": "octo-org/product-catalog",
        "suspect_commit_hint": "a1b2c3d",
        "suspect_file_hint": "src/product_catalog_server.py",
    }
    jsonschema.validate(sample_with_new_fields, DIAGNOSIS_JSON_SCHEMA)
