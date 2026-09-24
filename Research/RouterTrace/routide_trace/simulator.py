from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .schema import iter_trace

ExpertKey = tuple[int, int]
DEFAULT_POLICIES = ("lru", "lfu", "hybrid", "oracle")
POLICIES = DEFAULT_POLICIES + ("fifo", "random")


@dataclass
class CacheEntry:
    size: int
    frequency: int
    last_access: int


@dataclass
class SimulationResult:
    policy: str
    budget_bytes: int
    phase: str
    requests: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    bytes_read: int = 0
    peak_cache_bytes: int = 0
    oversized_expert_bypasses: int = 0
    working_set_bypasses: int = 0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.requests if self.requests else 0.0

    def json_value(self) -> dict[str, Any]:
        value = asdict(self)
        value["hit_rate"] = self.hit_rate
        return value


def parse_byte_count(value: str) -> int:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([kmgt]?i?b)?\s*", value.lower())
    if not match:
        raise argparse.ArgumentTypeError(f"invalid byte count: {value!r}")
    number = float(match.group(1))
    suffix = match.group(2) or "b"
    multipliers = {
        "b": 1,
        "kb": 1000,
        "mb": 1000**2,
        "gb": 1000**3,
        "tb": 1000**4,
        "kib": 1024,
        "mib": 1024**2,
        "gib": 1024**3,
        "tib": 1024**4,
    }
    result = int(number * multipliers[suffix])
    if result <= 0:
        raise argparse.ArgumentTypeError("byte count must be positive")
    return result


def _route_requests(
    trace_path: str | Path, phase: str,
) -> tuple[dict[str, Any], list[list[ExpertKey]]]:
    header: dict[str, Any] | None = None
    events: list[list[ExpertKey]] = []
    for current_header, record in iter_trace(trace_path):
        header = current_header
        if record["record_type"] != "route":
            continue
        if phase != "all" and record["phase"] != phase:
            continue
        layer = record["layer"]
        events.append([(layer, expert) for expert in sorted(record["selected_experts"])])
    assert header is not None
    return header, events


@dataclass
class Workload:
    header: dict[str, Any]
    events: list[list[ExpertKey]]
    phase: str


def load_workload(trace_path: str | Path, phase: str = "all") -> Workload:
    if phase not in {"all", "prefill", "decode"}:
        raise ValueError(f"unsupported phase: {phase}")
    header, events = _route_requests(trace_path, phase)
    return Workload(header=header, events=events, phase=phase)


class ExpertCache:
    def __init__(
        self,
        policy: str,
        budget_bytes: int,
        future_accesses: dict[ExpertKey, deque[int]] | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        if policy not in POLICIES:
            raise ValueError(f"unsupported policy: {policy}")
        if type(budget_bytes) is not int or budget_bytes <= 0:
            raise ValueError("cache budget must be a positive integer")
        if policy == "random":
            if type(seed) is not int or seed < 0:
                raise ValueError("random eviction requires an explicit nonnegative integer seed")
        elif seed is not None:
            raise ValueError("a random seed is only valid for random eviction")
        self.policy = policy
        self.budget_bytes = budget_bytes
        self.future_accesses = future_accesses
        self.entries: dict[ExpertKey, CacheEntry] = {}
        self.cache_bytes = 0
        self.clock = 0
        self.random = random.Random(seed) if policy == "random" else None

    def _victim(self, protected: set[ExpertKey]) -> ExpertKey | None:
        candidates = [key for key in self.entries if key not in protected]
        if not candidates:
            return None
        if self.policy == "fifo":
            # Dict insertion order is unchanged by hits; reinsertion goes to the end.
            return candidates[0]
        if self.policy == "random":
            assert self.random is not None
            return self.random.choice(candidates)
        if self.policy == "lru":
            return min(candidates, key=lambda key: self.entries[key].last_access)
        if self.policy == "lfu":
            return min(
                candidates,
                key=lambda key: (
                    self.entries[key].frequency,
                    self.entries[key].last_access,
                ),
            )
        if self.policy == "hybrid":
            return min(
                candidates,
                key=lambda key: (
                    self.entries[key].frequency
                    / (1 + self.clock - self.entries[key].last_access),
                    self.entries[key].last_access,
                ),
            )
        assert self.future_accesses is not None
        return max(
            candidates,
            key=lambda key: (
                self.future_accesses[key][0]
                if self.future_accesses[key]
                else math.inf
            ),
        )

    def access(
        self,
        key: ExpertKey,
        size: int,
        request_index: int,
        protected: set[ExpertKey],
        result: SimulationResult,
    ) -> bool:
        self.clock += 1
        result.requests += 1
        if self.future_accesses is not None:
            positions = self.future_accesses[key]
            if not positions or positions[0] != request_index:
                raise RuntimeError("oracle future-access index is inconsistent")
            positions.popleft()

        if entry := self.entries.get(key):
            result.hits += 1
            entry.frequency += 1
            entry.last_access = self.clock
            protected.add(key)
            return True

        result.misses += 1
        result.bytes_read += size
        if size > self.budget_bytes:
            result.oversized_expert_bypasses += 1
            return False
        protected_bytes = sum(
            self.entries[protected_key].size
            for protected_key in protected
            if protected_key in self.entries
        )
        if protected_bytes + size > self.budget_bytes:
            result.working_set_bypasses += 1
            return False

        while self.cache_bytes + size > self.budget_bytes:
            victim = self._victim(protected)
            if victim is None:
                result.working_set_bypasses += 1
                return False
            self.cache_bytes -= self.entries.pop(victim).size
            result.evictions += 1

        self.entries[key] = CacheEntry(size=size, frequency=1, last_access=self.clock)
        self.cache_bytes += size
        result.peak_cache_bytes = max(result.peak_cache_bytes, self.cache_bytes)
        protected.add(key)
        return False


def simulate_workload(
    workload: Workload, budget_bytes: int, policy: str, *, seed: int | None = None
) -> SimulationResult:
    result, _ = simulate_workload_phases(workload, budget_bytes, policy, seed=seed)
    return result


def simulate_workload_phases(
    workload: Workload,
    budget_bytes: int,
    policy: str,
    *,
    seed: int | None = None,
    event_phases: list[str] | None = None,
    protect_route: bool = True,
) -> tuple[SimulationResult, dict[str, SimulationResult]]:
    if event_phases is None:
        event_phases = [workload.phase] * len(workload.events)
    if len(event_phases) != len(workload.events) or any(
        not isinstance(phase, str) or not phase for phase in event_phases
    ):
        raise ValueError("each workload event requires a nonempty phase label")
    sizes = workload.header["model"]["expert_bytes_by_layer"]

    future_accesses: dict[ExpertKey, deque[int]] | None = None
    if policy == "oracle":
        future_accesses = defaultdict(deque)
        request_index = 0
        for event in workload.events:
            for key in event:
                future_accesses[key].append(request_index)
                request_index += 1

    cache = ExpertCache(policy, budget_bytes, future_accesses, seed=seed)
    phases: dict[str, SimulationResult] = {}
    request_index = 0
    for phase, event in zip(event_phases, workload.events):
        if phase not in phases:
            phases[phase] = SimulationResult(policy=policy, budget_bytes=budget_bytes, phase=phase)
        result = phases[phase]
        result.peak_cache_bytes = max(result.peak_cache_bytes, cache.cache_bytes)
        protected: set[ExpertKey] = set()
        for key in event:
            cache.access(
                key,
                sizes[key[0]],
                request_index,
                protected if protect_route else set(),
                result,
            )
            request_index += 1
    combined = SimulationResult(policy=policy, budget_bytes=budget_bytes, phase=workload.phase)
    counters = (
        "requests", "hits", "misses", "evictions", "bytes_read",
        "oversized_expert_bypasses", "working_set_bypasses",
    )
    for result in phases.values():
        for counter in counters:
            setattr(combined, counter, getattr(combined, counter) + getattr(result, counter))
        combined.peak_cache_bytes = max(combined.peak_cache_bytes, result.peak_cache_bytes)
    return combined, phases


def simulate(
    trace_path: str | Path,
    budget_bytes: int,
    policy: str,
    phase: str = "all",
    *,
    seed: int | None = None,
) -> SimulationResult:
    return simulate_workload(
        load_workload(trace_path, phase=phase), budget_bytes, policy, seed=seed
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate a byte-budgeted expert cache.")
    parser.add_argument("trace", type=Path)
    parser.add_argument("--budget", required=True, type=parse_byte_count)
    parser.add_argument(
        "--phase",
        choices=("decode", "prefill", "all"),
        default="decode",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=POLICIES,
        default=list(DEFAULT_POLICIES),
    )
    parser.add_argument("--random-seed", type=int)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if "random" in arguments.policies:
        if arguments.random_seed is None or arguments.random_seed < 0:
            parser.error("random eviction requires --random-seed with a nonnegative integer")
    elif arguments.random_seed is not None:
        parser.error("--random-seed requires the random policy")

    workload = load_workload(arguments.trace, phase=arguments.phase)
    report = {
        "schema_version": 1,
        "trace": str(arguments.trace),
        "trace_id": workload.header["trace_id"],
        "model_id": workload.header["model"]["id"],
        "model_revision": workload.header["model"]["revision"],
        "budget_bytes": arguments.budget,
        "phase": arguments.phase,
        "oracle_scope": (
            "Farthest-next-use with progressive route pins is a constrained reference "
            "schedule. The classical exact equal-size guarantee applies to relaxed "
            "unpinned demand paging; heterogeneous sizes are heuristic."
        ),
        "results": [
            simulate_workload(
                workload, arguments.budget, policy,
                seed=arguments.random_seed if policy == "random" else None,
            ).json_value()
            for policy in arguments.policies
        ],
    }
    if arguments.random_seed is not None:
        report["random_seed"] = arguments.random_seed
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
