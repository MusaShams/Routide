from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CORPUS_SCHEMA_VERSION = 1
EXPECTED_CATEGORIES = (
    "conversation",
    "code",
    "mathematics",
    "reasoning",
    "expository",
)


class CorpusValidationError(ValueError):
    """Raised when a prompt corpus violates the versioned contract."""


@dataclass(frozen=True)
class Prompt:
    id: str
    category: str
    text: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Corpus:
    id: str
    description: str
    prompts: tuple[Prompt, ...]
    sha256: str


def _canonical_digest(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_corpus(path: str | Path) -> Corpus:
    corpus_path = Path(path)
    try:
        value = json.loads(corpus_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise CorpusValidationError(f"invalid corpus JSON: {error.msg}") from error
    if not isinstance(value, dict):
        raise CorpusValidationError("corpus must be an object")
    required = {"schema_version", "corpus_id", "description", "prompts"}
    missing = required - value.keys()
    if missing:
        raise CorpusValidationError(
            f"corpus is missing fields: {', '.join(sorted(missing))}"
        )
    if value["schema_version"] != CORPUS_SCHEMA_VERSION:
        raise CorpusValidationError(
            f"unsupported corpus schema version {value['schema_version']!r}"
        )
    if not isinstance(value["corpus_id"], str) or not value["corpus_id"]:
        raise CorpusValidationError("corpus_id must be a nonempty string")
    if not isinstance(value["description"], str) or not value["description"]:
        raise CorpusValidationError("description must be a nonempty string")
    if not isinstance(value["prompts"], list) or not value["prompts"]:
        raise CorpusValidationError("prompts must be a nonempty array")

    prompts: list[Prompt] = []
    identifiers: set[str] = set()
    category_counts = {category: 0 for category in EXPECTED_CATEGORIES}
    for index, prompt_value in enumerate(value["prompts"]):
        if not isinstance(prompt_value, dict):
            raise CorpusValidationError(f"prompt {index} must be an object")
        if set(prompt_value) != {"id", "category", "text"}:
            raise CorpusValidationError(
                f"prompt {index} must contain only id, category, and text"
            )
        prompt = Prompt(**prompt_value)
        if not prompt.id or not prompt.text.strip():
            raise CorpusValidationError(f"prompt {index} has an empty id or text")
        if prompt.id in identifiers:
            raise CorpusValidationError(f"duplicate prompt id: {prompt.id}")
        if prompt.category not in EXPECTED_CATEGORIES:
            raise CorpusValidationError(
                f"prompt {prompt.id} has unknown category {prompt.category!r}"
            )
        if not prompt.id.startswith(f"{prompt.category}-"):
            raise CorpusValidationError(
                f"prompt {prompt.id} must be prefixed by its category"
            )
        identifiers.add(prompt.id)
        category_counts[prompt.category] += 1
        prompts.append(prompt)

    if len(set(category_counts.values())) != 1:
        raise CorpusValidationError(
            f"corpus categories are not balanced: {category_counts}"
        )

    return Corpus(
        id=value["corpus_id"],
        description=value["description"],
        prompts=tuple(prompts),
        sha256=_canonical_digest(value),
    )
