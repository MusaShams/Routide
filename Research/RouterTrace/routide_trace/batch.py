from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .corpus import EXPECTED_CATEGORIES, Corpus, Prompt, load_corpus
from .preflight import current_host, evaluate_capacity, load_experiment
from .schema import iter_trace


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    partial.replace(path)


def build_manifest(
    experiment: dict[str, Any],
    corpus_id: str,
    corpus_sha256: str,
    preflight: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": experiment["experiment_id"],
        "created_at": _timestamp(),
        "updated_at": _timestamp(),
        "status": preflight["status"],
        "corpus": {
            "id": corpus_id,
            "sha256": corpus_sha256,
        },
        "model": experiment["model"],
        "generation": experiment["generation"],
        "preflight": preflight,
        "prompts": {},
    }


def _load_manifest(
    path: Path,
    experiment: dict[str, Any],
    corpus_id: str,
    corpus_sha256: str,
    preflight: dict[str, Any],
) -> dict[str, Any]:
    if not path.exists():
        return build_manifest(
            experiment,
            corpus_id,
            corpus_sha256,
            preflight,
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("experiment_id") != experiment["experiment_id"]:
        raise ValueError("manifest experiment_id does not match")
    if value.get("corpus", {}).get("sha256") != corpus_sha256:
        raise ValueError("manifest corpus digest does not match")
    if value.get("model") != experiment["model"]:
        raise ValueError("manifest model configuration does not match")
    if value.get("generation") != experiment["generation"]:
        raise ValueError("manifest generation configuration does not match")
    value["preflight"] = preflight
    return value


def _completed_trace_matches(path: Path, prompt_id: str, prompt_sha256: str) -> bool:
    records = iter_trace(path)
    header, first = next(records)
    if first["record_type"] != "header":
        return False
    generation = header["generation"]
    if (
        generation["prompt_id"] != prompt_id
        or generation["prompt_sha256"] != prompt_sha256
    ):
        return False
    for _ in records:
        pass
    return True


def _remove_stale_trace_partial(output: Path) -> bool:
    partial = output.with_name(output.name + ".partial")
    if not partial.exists():
        return False
    partial.unlink()
    return True


def _select_prompts(corpus: Corpus, limit: int | None) -> tuple[Prompt, ...]:
    if limit is None or limit >= len(corpus.prompts):
        return corpus.prompts
    by_category = {
        category: [
            prompt
            for prompt in corpus.prompts
            if prompt.category == category
        ]
        for category in EXPECTED_CATEGORIES
    }
    selected: list[Prompt] = []
    round_index = 0
    while len(selected) < limit:
        for category in EXPECTED_CATEGORIES:
            prompts = by_category[category]
            if round_index < len(prompts):
                selected.append(prompts[round_index])
                if len(selected) == limit:
                    return tuple(selected)
        round_index += 1
    return tuple(selected)


def _completion_status(
    manifest: dict[str, Any],
    corpus: Corpus,
) -> str:
    completed = {
        prompt_id
        for prompt_id, result in manifest["prompts"].items()
        if result.get("status") == "completed"
    }
    expected = {prompt.id for prompt in corpus.prompts}
    return "completed" if completed >= expected else "partial"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a resumable Routide routing-trace corpus."
    )
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path(__file__).parents[1] / "experiment-v1.json",
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.limit is not None and arguments.limit <= 0:
        parser.error("--limit must be positive")
    return arguments


def main() -> None:
    arguments = _arguments()
    experiment = load_experiment(arguments.experiment)
    corpus_path = arguments.experiment.parent / experiment["corpus"]
    corpus = load_corpus(corpus_path)
    preflight = evaluate_capacity(
        experiment["model"],
        current_host(arguments.output_directory),
    )
    manifest_path = arguments.output_directory / "manifest.json"
    manifest = _load_manifest(
        manifest_path,
        experiment,
        corpus.id,
        corpus.sha256,
        preflight.json_value(),
    )
    manifest["updated_at"] = _timestamp()
    manifest["status"] = preflight.status
    _write_json_atomic(manifest_path, manifest)

    if preflight.status == "blocked":
        print(json.dumps(preflight.json_value(), indent=2, sort_keys=True))
        raise SystemExit(2)
    if arguments.preflight_only:
        print(json.dumps(preflight.json_value(), indent=2, sort_keys=True))
        return

    from .capture import capture_prompt, load_capture_runtime

    runtime = load_capture_runtime(
        experiment["model"]["id"],
        experiment["model"]["revision"],
    )
    prompts = _select_prompts(corpus, arguments.limit)
    manifest["status"] = "running"
    _write_json_atomic(manifest_path, manifest)

    for prompt in prompts:
        output = arguments.output_directory / f"{prompt.id}.trace.jsonl"
        if output.exists():
            if not _completed_trace_matches(output, prompt.id, prompt.sha256):
                raise ValueError(f"existing trace does not match prompt {prompt.id}")
            manifest["prompts"][prompt.id] = {
                "status": "completed",
                "trace": output.name,
                "prompt_sha256": prompt.sha256,
            }
            continue

        manifest["prompts"][prompt.id] = {
            "status": "running",
            "prompt_sha256": prompt.sha256,
            "recovered_incomplete_trace": _remove_stale_trace_partial(output),
        }
        manifest["updated_at"] = _timestamp()
        _write_json_atomic(manifest_path, manifest)
        try:
            capture_prompt(
                runtime=runtime,
                prompt=prompt.text,
                prompt_id=prompt.id,
                output=output,
                max_tokens=experiment["generation"]["max_tokens"],
                temperature=experiment["generation"]["temperature"],
                seed=experiment["generation"]["seed"],
                prefill_step_size=experiment["generation"]["prefill_step_size"],
            )
        except BaseException as error:
            manifest["prompts"][prompt.id] = {
                "status": "failed",
                "prompt_sha256": prompt.sha256,
                "error_type": type(error).__name__,
                "error": str(error),
            }
            manifest["status"] = "failed"
            manifest["updated_at"] = _timestamp()
            _write_json_atomic(manifest_path, manifest)
            raise
        manifest["prompts"][prompt.id] = {
            "status": "completed",
            "trace": output.name,
            "prompt_sha256": prompt.sha256,
        }
        manifest["updated_at"] = _timestamp()
        _write_json_atomic(manifest_path, manifest)

    manifest["status"] = _completion_status(manifest, corpus)
    manifest["updated_at"] = _timestamp()
    _write_json_atomic(manifest_path, manifest)


if __name__ == "__main__":
    main()
