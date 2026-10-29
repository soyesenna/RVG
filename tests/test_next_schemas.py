"""The public runtime schemas shipped as package resources: each accepts its example and the
runtime request rejects every evaluation field."""

import json
from importlib.resources import files
from pathlib import Path

import jsonschema
import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "schemas"


def schema(name):
    return json.loads((files("rvg") / "resources" / "schemas" / f"{name}.schema.json").read_text())


def example(name):
    return json.loads((FIXTURES / f"{name}.json").read_text())


@pytest.mark.parametrize("name", ["runtime_request", "evidence_unit", "trace_node"])
def test_examples(name):
    jsonschema.Draft202012Validator.check_schema(schema(name))
    jsonschema.validate(example(name), schema(name))


@pytest.mark.parametrize(
    "field",
    [
        "gold_answer",
        "dataset_name",
        "wrong",
        "expected_status",
        "is_impossible",
        "relevant_doc_ids",
    ],
)
def test_runtime_rejects_eval_data(field):
    req = example("runtime_request")
    req[field] = "leaked"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(req, schema("runtime_request"))
