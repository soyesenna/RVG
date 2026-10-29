"""Prompt and judgement-question texts, read from package resources (SEP-04).

Every text a product module sends to a model as an instruction, a criterion, a level description
or a system prompt is one file under `rvg/resources/prompts/`, holding exactly the characters of
that text as UTF-8 (no added newline). `manifest.json` names each text's file, byte size and
sha256. `prompt(id)` reads the file through `importlib.resources`, checks size and hash and
decodes it strictly; nothing is stripped, dedented, translated or normalised, and a missing,
changed or unknown text is an error, never a fallback to another copy. How texts are combined
(concatenation, `str.format`, f-string interpolation) stays in the module that uses them.

A module-level text is read when its module is first imported, like the policy resource; a text
used inside a function is read on its first use. Each text is read once and kept. `import rvg`
alone reads none."""

from __future__ import annotations

import hashlib
import json
from functools import cache, lru_cache
from importlib.resources import files

from .errors import ConfigurationError

ROOT = files("rvg") / "resources" / "prompts"
MANIFEST_SCHEMA = "rvg-prompt-manifest/1"


class PromptResourceError(ConfigurationError):
    """A prompt resource is unknown, missing or not the bytes its manifest records."""

    def __init__(self, code: str, prompt_id: str | None = None):
        self.code, self.prompt_id = code, prompt_id
        super().__init__(code if prompt_id is None else f"{code}: {prompt_id}")


def _manifest_bytes(root=ROOT) -> bytes:
    try:
        return root.joinpath("manifest.json").read_bytes()
    except FileNotFoundError:
        raise PromptResourceError("prompt_manifest_missing") from None


def read_manifest(root=ROOT) -> dict:
    body = json.loads(_manifest_bytes(root).decode("utf-8"))
    if body.get("schema") != MANIFEST_SCHEMA or body.get("encoding") != "utf-8":
        raise PromptResourceError("prompt_manifest_schema")
    return body


def read_prompt(prompt_id: str, root=ROOT, manifest: dict | None = None) -> str:
    entry = (manifest or read_manifest(root))["prompts"].get(prompt_id)
    if entry is None:
        raise PromptResourceError("prompt_id_unknown", prompt_id)
    try:
        data = root.joinpath(*entry["path"].split("/")).read_bytes()
    except FileNotFoundError:
        raise PromptResourceError("prompt_resource_missing", prompt_id) from None
    if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise PromptResourceError("prompt_resource_hash_mismatch", prompt_id)
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        raise PromptResourceError("prompt_resource_not_utf8", prompt_id) from None


@lru_cache(maxsize=1)
def manifest() -> dict:
    """The installed prompt manifest (read once)."""
    return read_manifest()


@lru_cache(maxsize=1)
def manifest_sha256() -> str:
    """sha256 of the manifest file's bytes: one fingerprint for every prompt text."""
    return hashlib.sha256(_manifest_bytes()).hexdigest()


@cache
def prompt(prompt_id: str) -> str:
    """The text of one prompt resource, checked against the manifest (read once)."""
    return read_prompt(prompt_id, manifest=manifest())


def verify_all(root=ROOT) -> dict:
    """Every text of the manifest checked, without raising: for liveness checks and tools.
    Reads only package resources."""
    try:
        body = read_manifest(root)
    except (PromptResourceError, ValueError) as exc:
        code = exc.code if isinstance(exc, PromptResourceError) else "prompt_manifest_unreadable"
        return {"ok": False, "count": 0, "failures": [{"id": None, "code": code}]}
    failures = []
    for prompt_id in body["prompts"]:
        try:
            read_prompt(prompt_id, root, body)
        except PromptResourceError as exc:
            failures.append({"id": prompt_id, "code": exc.code})
    return {
        "ok": not failures,
        "count": len(body["prompts"]),
        "manifest_sha256": hashlib.sha256(_manifest_bytes(root)).hexdigest(),
        "failures": failures,
    }
