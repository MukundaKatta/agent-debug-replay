"""
agent-debug-replay: Step-through replay of JSONL agent run traces.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterator, Optional


class StepKind(str, Enum):
    MESSAGE = "message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    ERROR = "error"
    CHECKPOINT = "checkpoint"
    CUSTOM = "custom"


@dataclass
class ReplayStep:
    index: int
    kind: str
    data: dict[str, Any]
    raw: str

    @property
    def timestamp(self) -> Optional[str]:
        return self.data.get("timestamp")

    @property
    def tool_name(self) -> Optional[str]:
        return self.data.get("tool_name") or self.data.get("tool")

    @property
    def content(self) -> Any:
        return self.data.get("content") or self.data.get("message") or self.data.get("output")

    def __repr__(self) -> str:
        return f"ReplayStep(index={self.index}, kind={self.kind!r})"


class ReplayFilter:
    def __init__(self) -> None:
        self._kind_whitelist: Optional[set[str]] = None
        self._kind_blacklist: set[str] = set()
        self._tool_names: Optional[set[str]] = None
        self._custom: list[Callable[[ReplayStep], bool]] = []

    def only_kinds(self, *kinds: str) -> "ReplayFilter":
        self._kind_whitelist = set(kinds)
        return self

    def exclude_kinds(self, *kinds: str) -> "ReplayFilter":
        self._kind_blacklist.update(kinds)
        return self

    def only_tools(self, *tool_names: str) -> "ReplayFilter":
        self._tool_names = set(tool_names)
        return self

    def add(self, fn: Callable[[ReplayStep], bool]) -> "ReplayFilter":
        self._custom.append(fn)
        return self

    def matches(self, step: ReplayStep) -> bool:
        if self._kind_whitelist is not None and step.kind not in self._kind_whitelist:
            return False
        if step.kind in self._kind_blacklist:
            return False
        if self._tool_names is not None and step.tool_name not in self._tool_names:
            return False
        return all(fn(step) for fn in self._custom)


class DebugReplay:
    """Load and replay a JSONL agent trace step by step."""

    def __init__(self, steps: list[ReplayStep]) -> None:
        self._steps = steps
        self._cursor = 0

    @classmethod
    def from_jsonl(cls, path: str) -> "DebugReplay":
        steps: list[ReplayStep] = []
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                kind = data.get("kind", StepKind.CUSTOM.value)
                steps.append(ReplayStep(index=i, kind=kind, data=data, raw=line))
        return cls(steps)

    @classmethod
    def from_string(cls, text: str) -> "DebugReplay":
        steps: list[ReplayStep] = []
        for i, line in enumerate(text.splitlines()):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            kind = data.get("kind", StepKind.CUSTOM.value)
            steps.append(ReplayStep(index=i, kind=kind, data=data, raw=line))
        return cls(steps)

    @classmethod
    def from_dicts(cls, records: list[dict[str, Any]]) -> "DebugReplay":
        steps = [
            ReplayStep(index=i, kind=r.get("kind", StepKind.CUSTOM.value), data=r, raw=json.dumps(r))
            for i, r in enumerate(records)
        ]
        return cls(steps)

    def reset(self) -> None:
        self._cursor = 0

    def seek(self, index: int) -> None:
        if index < 0 or index > len(self._steps):
            raise IndexError(f"Index {index} out of range [0, {len(self._steps)}]")
        self._cursor = index

    @property
    def current(self) -> Optional[ReplayStep]:
        if self._cursor < len(self._steps):
            return self._steps[self._cursor]
        return None

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def at_end(self) -> bool:
        return self._cursor >= len(self._steps)

    def step(self) -> Optional[ReplayStep]:
        if self.at_end:
            return None
        s = self._steps[self._cursor]
        self._cursor += 1
        return s

    def step_n(self, n: int) -> list[ReplayStep]:
        result = []
        for _ in range(n):
            s = self.step()
            if s is None:
                break
            result.append(s)
        return result

    def all_steps(self) -> list[ReplayStep]:
        return list(self._steps)

    def steps_by_kind(self, kind: str) -> list[ReplayStep]:
        return [s for s in self._steps if s.kind == kind]

    def filter(self, f: ReplayFilter) -> list[ReplayStep]:
        return [s for s in self._steps if f.matches(s)]

    def iterate(self, f: Optional[ReplayFilter] = None) -> Iterator[ReplayStep]:
        for s in self._steps:
            if f is None or f.matches(s):
                yield s

    @property
    def total_steps(self) -> int:
        return len(self._steps)

    def kind_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for s in self._steps:
            counts[s.kind] = counts.get(s.kind, 0) + 1
        return counts

    def tool_calls(self) -> list[ReplayStep]:
        return self.steps_by_kind(StepKind.TOOL_CALL.value)

    def errors(self) -> list[ReplayStep]:
        return self.steps_by_kind(StepKind.ERROR.value)

    def summary(self) -> dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "kind_counts": self.kind_counts(),
            "tool_calls": len(self.tool_calls()),
            "errors": len(self.errors()),
        }

    def __len__(self) -> int:
        return len(self._steps)

    def __repr__(self) -> str:
        return f"DebugReplay(total_steps={self.total_steps}, cursor={self._cursor})"


__all__ = ["DebugReplay", "ReplayStep", "ReplayFilter", "StepKind"]
