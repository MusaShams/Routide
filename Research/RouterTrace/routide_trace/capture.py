from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import SCHEMA_VERSION

DEFAULT_MODEL = "mlx-community/Qwen3.6-35B-A3B-4bit"
MLX_LM_REVISION = "cf10f962b7a20e63a6df43dbf0faf06070153d40"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class AtomicTraceWriter:
    def __init__(self, output: Path, force: bool) -> None:
        self.output = output
        self.partial = output.with_name(output.name + ".partial")
        if not force and (self.output.exists() or self.partial.exists()):
            raise FileExistsError(
                f"trace output or incomplete capture exists: "
                f"{self.output} or {self.partial}; use --force to replace it"
            )
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.partial.unlink(missing_ok=True)
        self.stream = self.partial.open("x", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        self.stream.write(json.dumps(record, separators=(",", ":"), sort_keys=True))
        self.stream.write("\n")

    def complete(self, summary: dict[str, Any]) -> None:
        self.write(summary)
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        self.partial.replace(self.output)

    def close_incomplete(self) -> None:
        if not self.stream.closed:
            self.stream.close()


class RouteRecorder:
    def __init__(
        self,
        writer: AtomicTraceWriter,
        prompt_tokens: list[int],
        num_layers: int,
    ) -> None:
        self.writer = writer
        self.prompt_token_count = len(prompt_tokens)
        self.num_layers = num_layers
        self.next_token_position = 0
        self.forward_index = 0
        self.current_tokens: list[int] | None = None
        self.current_positions: list[int] | None = None

    def begin_forward(self, inputs: Any) -> None:
        nested = inputs.tolist()
        if len(nested) != 1:
            raise ValueError("routing capture currently requires batch size one")
        self.current_tokens = [int(token) for token in nested[0]]
        start = self.next_token_position
        self.current_positions = list(range(start, start + len(self.current_tokens)))
        self.next_token_position += len(self.current_tokens)

    def end_forward(self) -> None:
        self.current_tokens = None
        self.current_positions = None
        self.forward_index += 1

    def record(
        self,
        layer: int,
        indices: Any,
        router_scores: Any,
        routing_weights: Any,
        expert_execution_nanoseconds: int,
        mx: Any,
    ) -> None:
        if self.current_tokens is None or self.current_positions is None:
            raise RuntimeError("router event occurred outside a model forward pass")
        if not 0 <= layer < self.num_layers:
            raise ValueError(f"router layer {layer} is out of range")
        mx.eval(indices, router_scores, routing_weights)
        index_values = indices.tolist()
        score_values = router_scores.tolist()
        weight_values = routing_weights.tolist()
        if len(index_values) != 1 or len(index_values[0]) != len(self.current_tokens):
            raise ValueError("router output shape does not match the model input")

        token_start = self.current_positions[0]
        batch_phase = (
            "prefill" if token_start < self.prompt_token_count else "decode"
        )
        self.writer.write(
            {
                "record_type": "route_batch",
                "forward_index": self.forward_index,
                "phase": batch_phase,
                "layer": layer,
                "token_start": token_start,
                "token_count": len(self.current_tokens),
                "routed_expert_execution_nanoseconds": (
                    expert_execution_nanoseconds
                ),
            }
        )
        for offset, token_id in enumerate(self.current_tokens):
            ranked = sorted(
                zip(
                    index_values[0][offset],
                    score_values[0][offset],
                    weight_values[0][offset],
                ),
                key=lambda item: (-float(item[1]), int(item[0])),
            )
            position = self.current_positions[offset]
            self.writer.write(
                {
                    "record_type": "route",
                    "forward_index": self.forward_index,
                    "phase": (
                        "prefill" if position < self.prompt_token_count else "decode"
                    ),
                    "token_position": position,
                    "token_id": token_id,
                    "layer": layer,
                    "selected_experts": [int(item[0]) for item in ranked],
                    "router_scores": [float(item[1]) for item in ranked],
                    "routing_weights": [float(item[2]) for item in ranked],
                }
            )


class TracingModel:
    def __init__(self, model: Any, recorder: RouteRecorder) -> None:
        self.model = model
        self.text_model = resident_text_model(model)
        self.recorder = recorder

    @property
    def layers(self) -> Any:
        return self.text_model.layers

    def make_cache(self) -> Any:
        cache = self.text_model.make_cache()
        return cache() if callable(cache) else cache

    def __call__(self, inputs: Any, **kwargs: Any) -> Any:
        self.recorder.begin_forward(inputs)
        try:
            return self.model(inputs, **kwargs)
        finally:
            self.recorder.end_forward()


def resident_text_model(model: Any) -> Any:
    text_model = getattr(model, "language_model", model)
    required = ("args", "model", "layers", "make_cache")
    missing = [name for name in required if not hasattr(text_model, name)]
    if missing:
        raise TypeError(
            "loaded model has no compatible resident text model: "
            + ", ".join(missing)
        )
    return text_model


@contextmanager
def traced_sparse_blocks(model: Any, recorder: RouteRecorder, mx: Any):
    from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock

    layers = list(resident_text_model(model).layers)
    sparse_layers: list[Any] = []
    for layer_index, layer in enumerate(layers):
        block = getattr(layer, "mlp", None)
        if not isinstance(block, Qwen3NextSparseMoeBlock):
            raise TypeError(f"layer {layer_index} is not a Qwen3.6 sparse MoE block")
        block._routide_layer_index = layer_index
        sparse_layers.append(block)

    original_call = Qwen3NextSparseMoeBlock.__call__

    def traced_call(block: Any, x: Any) -> Any:
        if block.sharding_group is not None:
            raise RuntimeError("routing capture does not support distributed sharding")

        gates = mx.softmax(block.gate(x), axis=-1, precise=True)
        indices = mx.argpartition(gates, kth=-block.top_k, axis=-1)[..., -block.top_k :]
        router_scores = mx.take_along_axis(gates, indices, axis=-1)
        routing_weights = router_scores
        if block.norm_topk_prob:
            routing_weights = router_scores / router_scores.sum(
                axis=-1, keepdims=True
            )

        mx.eval(indices, router_scores, routing_weights)
        expert_start = time.perf_counter_ns()
        routed = block.switch_mlp(x, indices)
        routed = (routed * routing_weights[..., None]).sum(axis=-2)
        mx.eval(routed)
        expert_execution_nanoseconds = time.perf_counter_ns() - expert_start
        recorder.record(
            block._routide_layer_index,
            indices,
            router_scores,
            routing_weights,
            expert_execution_nanoseconds,
            mx,
        )

        shared = block.shared_expert(x)
        shared = mx.sigmoid(block.shared_expert_gate(x)) * shared
        return routed + shared

    Qwen3NextSparseMoeBlock.__call__ = traced_call
    try:
        yield
    finally:
        Qwen3NextSparseMoeBlock.__call__ = original_call
        for block in sparse_layers:
            del block._routide_layer_index


def _expert_bytes_by_layer(model: Any) -> list[int]:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    sizes: list[int] = []
    for layer_index, layer in enumerate(resident_text_model(model).layers):
        block = getattr(layer, "mlp", None)
        switch_mlp = getattr(block, "switch_mlp", None)
        num_experts = getattr(block, "num_experts", 0)
        if switch_mlp is None or num_experts <= 0:
            raise TypeError(f"layer {layer_index} has no measurable expert weights")
        arrays = [value for _, value in tree_flatten(switch_mlp.parameters())]
        if not arrays or any(not isinstance(value, mx.array) for value in arrays):
            raise TypeError(f"layer {layer_index} has invalid expert parameters")
        total_bytes = sum(value.nbytes for value in arrays)
        if total_bytes % num_experts != 0:
            raise ValueError(f"layer {layer_index} expert weights are not uniform")
        sizes.append(total_bytes // num_experts)
    return sizes


def _resolve_model_revision(model_id: str, requested_revision: str) -> str:
    if Path(model_id).exists():
        return "local"
    from huggingface_hub import HfApi

    info = HfApi().model_info(model_id, revision=requested_revision)
    if not info.sha:
        raise RuntimeError(f"unable to resolve model revision for {model_id}")
    return info.sha


def _installed_mlx_lm_revision() -> str:
    distribution = importlib.metadata.Distribution.from_name("mlx-lm")
    direct_url = distribution.read_text("direct_url.json")
    if direct_url is None:
        raise RuntimeError(
            "mlx-lm was not installed from the pinned VCS requirement"
        )
    metadata = json.loads(direct_url)
    revision = metadata.get("vcs_info", {}).get("commit_id")
    if revision != MLX_LM_REVISION:
        raise RuntimeError(
            f"installed mlx-lm revision {revision!r} does not match "
            f"the required revision {MLX_LM_REVISION}"
        )
    return revision


def _eos_token_ids(tokenizer: Any) -> set[int]:
    values = getattr(tokenizer, "eos_token_ids", None)
    if values is None:
        values = getattr(tokenizer, "eos_token_id", None)
    if values is None:
        return set()
    if isinstance(values, int):
        return {values}
    return {int(value) for value in values}


@dataclass(frozen=True)
class CaptureRuntime:
    model_id: str
    model_revision: str
    runtime_revision: str
    model: Any
    tokenizer: Any
    config: dict[str, Any]
    mx: Any
    generate_step: Any
    make_sampler: Any


def load_capture_runtime(
    model_id: str,
    model_revision: str,
) -> CaptureRuntime:
    try:
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_sampler
    except ImportError as error:
        raise RuntimeError(
            "MLX LM is required. Install Research/RouterTrace/requirements.txt."
        ) from error

    runtime_revision = _installed_mlx_lm_revision()
    resolved_revision = _resolve_model_revision(model_id, model_revision)
    model, tokenizer, config = load(
        model_id,
        revision=(None if resolved_revision == "local" else resolved_revision),
        return_config=True,
    )
    if config.get("model_type") != "qwen3_5_moe":
        raise TypeError("capture requires a qwen3_5_moe model")
    return CaptureRuntime(
        model_id=model_id,
        model_revision=resolved_revision,
        runtime_revision=runtime_revision,
        model=model,
        tokenizer=tokenizer,
        config=config,
        mx=mx,
        generate_step=generate_step,
        make_sampler=make_sampler,
    )


def capture_prompt(
    runtime: CaptureRuntime,
    prompt: str,
    prompt_id: str,
    output: Path,
    max_tokens: int,
    temperature: float,
    seed: int,
    prefill_step_size: int,
    raw_prompt: bool = False,
    force: bool = False,
) -> Path:
    if not prompt:
        raise ValueError("prompt must not be empty")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if temperature < 0:
        raise ValueError("temperature must be nonnegative")
    if prefill_step_size <= 0:
        raise ValueError("prefill_step_size must be positive")

    formatted_prompt = prompt
    if not raw_prompt:
        formatted_prompt = runtime.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    prompt_tokens = [
        int(token) for token in runtime.tokenizer.encode(formatted_prompt)
    ]
    if not prompt_tokens:
        raise ValueError("the formatted prompt produced no tokens")

    text_config = runtime.config.get("text_config", runtime.config)
    num_layers = int(text_config["num_hidden_layers"])
    num_experts = int(text_config["num_experts"])
    top_k = int(text_config["num_experts_per_tok"])
    expert_sizes = _expert_bytes_by_layer(runtime.model)
    writer = AtomicTraceWriter(output, force)
    writer.write(
        {
            "record_type": "header",
            "schema_version": SCHEMA_VERSION,
            "trace_id": str(uuid.uuid4()),
            "created_at": _timestamp(),
            "model": {
                "id": runtime.model_id,
                "revision": runtime.model_revision,
                "architecture": "qwen3_5_moe",
                "num_layers": num_layers,
                "num_experts": num_experts,
                "top_k": top_k,
                "expert_bytes_by_layer": expert_sizes,
            },
            "runtime": {
                "name": "mlx-lm",
                "version": importlib.metadata.version("mlx-lm"),
                "revision": runtime.runtime_revision,
            },
            "generation": {
                "prompt_id": prompt_id,
                "prompt_sha256": _sha256(prompt),
                "prompt_tokens": len(prompt_tokens),
                "max_generated_tokens": max_tokens,
                "temperature": temperature,
                "seed": seed,
                "prefill_step_size": prefill_step_size,
            },
        }
    )

    recorder = RouteRecorder(writer, prompt_tokens, num_layers)
    traced_model = TracingModel(runtime.model, recorder)
    generated: list[int] = []
    runtime.mx.random.seed(seed)
    sampler = runtime.make_sampler(temp=temperature)

    try:
        with traced_sparse_blocks(runtime.model, recorder, runtime.mx):
            for token, _ in runtime.generate_step(
                runtime.mx.array(prompt_tokens),
                traced_model,
                max_tokens=max_tokens,
                sampler=sampler,
                prefill_step_size=prefill_step_size,
            ):
                generated.append(int(token))
                if token in _eos_token_ids(runtime.tokenizer):
                    break

        decoded_output = runtime.tokenizer.decode(generated)
        writer.complete(
            {
                "record_type": "summary",
                "completed_at": _timestamp(),
                "generated_tokens": len(generated),
                "generated_token_ids": generated,
                "output_sha256": _sha256(decoded_output),
            }
        )
    except BaseException:
        writer.close_incomplete()
        raise

    return output


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture Qwen3.6 expert routing on macOS.")
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt")
    prompt_group.add_argument("--prompt-file", type=Path)
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--raw-prompt", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    if arguments.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if arguments.temperature < 0:
        parser.error("--temperature must be nonnegative")
    if arguments.prefill_step_size <= 0:
        parser.error("--prefill-step-size must be positive")
    return arguments


def main() -> None:
    arguments = _arguments()
    prompt = (
        arguments.prompt_file.read_text(encoding="utf-8")
        if arguments.prompt_file
        else arguments.prompt
    )
    runtime = load_capture_runtime(
        arguments.model,
        arguments.model_revision,
    )
    capture_prompt(
        runtime=runtime,
        prompt=prompt,
        prompt_id=arguments.prompt_id,
        output=arguments.output,
        max_tokens=arguments.max_tokens,
        temperature=arguments.temperature,
        seed=arguments.seed,
        prefill_step_size=arguments.prefill_step_size,
        raw_prompt=arguments.raw_prompt,
        force=arguments.force,
    )
    print(f"Wrote {arguments.output}")


if __name__ == "__main__":
    main()
