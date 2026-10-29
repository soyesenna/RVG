"""0.1.0 (parent docs/PUB_2_REPORT.md): the build backend is pinned to one version. The wheel's
`WHEEL` file names its generator (`Generator: hatchling X`), so an unpinned backend gave other
wheel bytes on the day a new hatchling came out."""

import tomllib
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[2]


def test_the_build_backend_is_pinned_exactly():
    system = tomllib.loads((ROOT / "pyproject.toml").read_text())["build-system"]
    (backend,) = [Requirement(r) for r in system["requires"]]
    assert backend.name == "hatchling" and system["build-backend"] == "hatchling.build"
    (spec,) = backend.specifier
    assert spec.operator == "==" and "*" not in spec.version
