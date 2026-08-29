"""Packaging contract: the required dependency floor stays minimal.

Every package in [project].dependencies must correspond to an unguarded
top-level import somewhere in app/.  Everything else belongs in an extra,
because the code already guards it with try/except ImportError.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"

REQUIRED_FLOOR = {
    "fastapi",
    "uvicorn",
    "pydantic",
    "pydantic-settings",
    "httpx",
    "watchdog",
}

# Packages the code imports inside try/except ImportError, or does not
# import at all.  None of these may appear in [project].dependencies.
MUST_BE_OPTIONAL = {
    "faiss-cpu",      # never imported anywhere (P2)
    "portkey-ai",     # never imported anywhere (P3)
    "pyrit",          # redteam sidecar only (P5)
    "garak",          # redteam sidecar only (P5)
    "llm-guard",
    "routellm",
    "spacy",
    "langfuse",
    "guardrails-ai",
}


def _dep_names(specs: list[str]) -> set[str]:
    """Strip version specifiers and extras: 'uvicorn[standard]>=0.29' -> 'uvicorn'."""
    names = set()
    for spec in specs:
        name = spec.split(";")[0].strip()
        for sep in (">=", "<=", "==", "~=", ">", "<", "!="):
            name = name.split(sep)[0]
        names.add(name.split("[")[0].strip().lower())
    return names


@pytest.fixture(scope="module")
def pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text())


def test_required_dependencies_are_exactly_the_floor(pyproject: dict) -> None:
    assert _dep_names(pyproject["project"]["dependencies"]) == REQUIRED_FLOOR


def test_no_optional_package_is_a_hard_requirement(pyproject: dict) -> None:
    required = _dep_names(pyproject["project"]["dependencies"])
    leaked = required & MUST_BE_OPTIONAL
    assert leaked == set(), f"these belong in extras, not dependencies: {sorted(leaked)}"


def test_expected_extras_exist(pyproject: dict) -> None:
    extras = pyproject["project"]["optional-dependencies"]
    for group in ("llm", "safety", "cache", "grounded", "observe", "dev", "all"):
        assert group in extras, f"missing extra: {group}"


def test_all_extra_covers_every_runtime_extra(pyproject: dict) -> None:
    """`all` must name every runtime extra.

    Adding a sixth extra and forgetting to list it here is the likely future
    regression: `pip install '.[all]'` would silently omit a whole pipeline
    stage, and nothing else in this file would notice.
    """
    project = pyproject["project"]
    extras = project["optional-dependencies"]
    runtime = set(extras) - {"all", "dev"}

    specs = extras["all"]
    assert len(specs) == 1, "`all` should be a single self-reference"
    name, _, bracket = specs[0].partition("[")
    assert name == project["name"], (
        f"`all` self-references {name!r}, but the package is {project['name']!r}"
    )
    named = {group.strip() for group in bracket.rstrip("]").split(",")}
    assert named == runtime, (
        f"`all` names {sorted(named)}; runtime extras are {sorted(runtime)}"
    )


def test_no_extra_is_empty(pyproject: dict) -> None:
    """An extra emptied to [] still has its key, so check the contents."""
    for group, specs in pyproject["project"]["optional-dependencies"].items():
        assert specs, f"extra {group!r} is declared but empty"


def test_no_commercial_only_packages_anywhere(pyproject: dict) -> None:
    """faiss-cpu and portkey-ai are dead code — they must be gone entirely."""
    all_specs = list(pyproject["project"]["dependencies"])
    for group_specs in pyproject["project"]["optional-dependencies"].values():
        all_specs.extend(group_specs)
    names = _dep_names(all_specs)
    assert "portkey-ai" not in names, "portkey-ai is never imported (P3)"
    assert "faiss-cpu" not in names, "faiss-cpu is never imported (P2)"
