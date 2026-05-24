"""Core Replayer + Step + DiffResult + Summary implementation."""

from __future__ import annotations

import json
import sys
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------- Step ----------

# Fields that come straight out of the step-log "spine" of a step. These are
# the dataclass attributes. Everything else is folded into `attrs`.
_SPINE_FIELDS = {"name", "started_at", "duration_ms", "run_id", "meta"}


@dataclass
class Step:
    """One step of an agent run, parsed from a step-log JSONL line.

    The dataclass holds the well-known spine fields plus an open-ended
    `attrs` dict for everything else (input, output, cost_usd, tokens_in,
    tokens_out, model, tool_name, tool_args, tool_output, error, ...).

    `meta` is mutable so the Replayer can flip a `_stale` flag without
    rebuilding the whole step. The rest behaves like a value type:
    `replace(**changes)` returns a new Step rather than mutating.
    """

    name: str = ""
    started_at: str | None = None
    duration_ms: float | None = None
    run_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    attrs: dict[str, Any] = field(default_factory=dict)

    # ---- attribute-style access for the open-ended fields ----

    def __getattr__(self, name: str) -> Any:
        # __getattr__ runs only if normal lookup failed, so spine fields and
        # actual dataclass attrs are never routed here.
        if name.startswith("_"):
            raise AttributeError(name)
        attrs = self.__dict__.get("attrs", {})
        if name in attrs:
            return attrs[name]
        raise AttributeError(name)

    def get(self, name: str, default: Any = None) -> Any:
        """Uniform getter across spine + attrs + meta. `meta` is reached as
        `step.get("meta")` so this never aliases its individual keys."""
        if name in _SPINE_FIELDS:
            return getattr(self, name)
        if name in self.attrs:
            return self.attrs[name]
        return default

    def replace(self, **changes: Any) -> Step:
        """Return a new Step with the given fields replaced. Spine fields go
        to the dataclass; everything else goes into `attrs`."""
        new_meta = dict(self.meta)
        new_attrs = dict(self.attrs)
        spine_changes: dict[str, Any] = {
            "name": self.name,
            "started_at": self.started_at,
            "duration_ms": self.duration_ms,
            "run_id": self.run_id,
        }
        for k, v in changes.items():
            if k == "meta":
                new_meta = dict(v) if v is not None else {}
            elif k in spine_changes:
                spine_changes[k] = v
            else:
                new_attrs[k] = v
        return Step(
            name=spine_changes["name"],
            started_at=spine_changes["started_at"],
            duration_ms=spine_changes["duration_ms"],
            run_id=spine_changes["run_id"],
            meta=new_meta,
            attrs=new_attrs,
        )

    def to_dict(self) -> dict[str, Any]:
        """Flat dict suitable for json.dumps. Spine fields first, then attrs,
        then meta (only if non-empty so round-tripping a minimal step stays
        minimal)."""
        out: dict[str, Any] = {}
        if self.name:
            out["name"] = self.name
        if self.started_at is not None:
            out["started_at"] = self.started_at
        if self.duration_ms is not None:
            out["duration_ms"] = self.duration_ms
        if self.run_id is not None:
            out["run_id"] = self.run_id
        # attrs are inlined into the top-level object (step-log shape)
        for k, v in self.attrs.items():
            out[k] = v
        if self.meta:
            out["meta"] = dict(self.meta)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Step:
        spine: dict[str, Any] = {
            "name": data.get("name", ""),
            "started_at": data.get("started_at"),
            "duration_ms": data.get("duration_ms"),
            "run_id": data.get("run_id"),
        }
        meta = data.get("meta")
        meta = dict(meta) if isinstance(meta, dict) else {}
        attrs = {
            k: v
            for k, v in data.items()
            if k not in _SPINE_FIELDS
        }
        return cls(
            name=str(spine["name"]) if spine["name"] is not None else "",
            started_at=spine["started_at"],
            duration_ms=spine["duration_ms"],
            run_id=spine["run_id"],
            meta=meta,
            attrs=attrs,
        )


# ---------- Diff + Summary ----------


@dataclass(frozen=True)
class DiffResult:
    """Field-by-field comparison between two Replayers.

    `differences` is a flat list. Each entry is a dict with keys:
      * step_index: int
      * field: str (dotted for nested meta, e.g. "meta._stale")
      * before: Any
      * after: Any

    Added or removed steps show up with field="<step>" and a sentinel
    before/after of None on the missing side.
    """

    differences: list[dict[str, Any]]

    @property
    def is_empty(self) -> bool:
        return not self.differences

    def __bool__(self) -> bool:
        return bool(self.differences)


@dataclass(frozen=True)
class Summary:
    step_count: int
    total_cost: float
    total_duration_ms: float
    by_step_name: dict[str, int]


# ---------- Replayer ----------


class Replayer:
    """Immutable view over a list of Steps loaded from a step-log JSONL file.

    Mutation-returning methods (`edit_step`, `replay_through`) build a new
    Replayer; the receiver is left untouched. The only place we mutate is
    inside the *new* Replayer's freshly built step list, before the
    Replayer is handed back to the caller.
    """

    __slots__ = ("_steps",)

    def __init__(self, steps: list[Step]) -> None:
        # store a private copy so external mutation of the input list
        # doesn't show up in our view
        self._steps = list(steps)

    # ---- construction ----

    @classmethod
    def from_log(cls, path: str | Path) -> Replayer:
        """Parse a step-log JSONL file. Empty file -> empty Replayer.
        Corrupt JSON lines are skipped with a warning on stderr."""
        p = Path(path).expanduser()
        steps: list[Step] = []
        if not p.exists():
            raise FileNotFoundError(f"step-log file not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            for line_no, raw in enumerate(f, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError as e:
                    print(
                        f"agent-debug-replay: skipping corrupt JSON line {line_no} in {p}: {e}",
                        file=sys.stderr,
                    )
                    continue
                if not isinstance(data, dict):
                    print(
                        f"agent-debug-replay: skipping non-object line {line_no} in {p}",
                        file=sys.stderr,
                    )
                    continue
                steps.append(Step.from_dict(data))
        return cls(steps)

    @classmethod
    def from_steps(cls, steps: list[Step]) -> Replayer:
        return cls(list(steps))

    # ---- access ----

    @property
    def steps(self) -> list[Step]:
        """Return a shallow copy of the step list. The caller may mutate the
        returned list without affecting the Replayer."""
        return list(self._steps)

    def __len__(self) -> int:
        return len(self._steps)

    def step(self, index: int) -> Step:
        return self._steps[index]

    def iter(self) -> Iterator[Step]:
        return iter(self._steps)

    def __iter__(self) -> Iterator[Step]:
        return iter(self._steps)

    # ---- edit ----

    def edit_step(self, index: int, **changes: Any) -> Replayer:
        """Return a new Replayer with the step at `index` replaced by
        `step.replace(**changes)`. All downstream steps (index > target)
        are cloned with `meta._stale=True` so the caller can see that their
        captured outputs are no longer guaranteed to follow from the new
        upstream state."""
        if index < 0 or index >= len(self._steps):
            raise IndexError(f"step index {index} out of range (len={len(self._steps)})")
        new_steps: list[Step] = []
        for i, s in enumerate(self._steps):
            if i < index:
                new_steps.append(s)
            elif i == index:
                new_steps.append(s.replace(**changes))
            else:
                # mark downstream stale (preserve other meta)
                new_meta = dict(s.meta)
                new_meta["_stale"] = True
                new_steps.append(s.replace(meta=new_meta))
        return Replayer(new_steps)

    # ---- diff ----

    def diff(self, other: Replayer) -> DiffResult:
        """Field-by-field comparison against another Replayer. Compares
        spine fields, attrs, and meta keys at the same index. Reports
        added or removed steps as a single entry with field='<step>'."""
        differences: list[dict[str, Any]] = []
        n = max(len(self._steps), len(other._steps))
        for i in range(n):
            a = self._steps[i] if i < len(self._steps) else None
            b = other._steps[i] if i < len(other._steps) else None
            if a is None:
                differences.append(
                    {"step_index": i, "field": "<step>", "before": None, "after": b.to_dict()}
                )
                continue
            if b is None:
                differences.append(
                    {"step_index": i, "field": "<step>", "before": a.to_dict(), "after": None}
                )
                continue
            differences.extend(_step_diff(i, a, b))
        return DiffResult(differences=differences)

    # ---- summary / filter / export / replay_through ----

    def summary(self) -> Summary:
        total_cost = 0.0
        total_duration = 0.0
        counts: Counter[str] = Counter()
        for s in self._steps:
            counts[s.name] += 1
            if s.duration_ms is not None:
                with suppress(TypeError, ValueError):
                    total_duration += float(s.duration_ms)
            cost = s.attrs.get("cost_usd")
            if cost is not None:
                with suppress(TypeError, ValueError):
                    total_cost += float(cost)
        return Summary(
            step_count=len(self._steps),
            total_cost=total_cost,
            total_duration_ms=total_duration,
            by_step_name=dict(counts),
        )

    def filter(
        self,
        *,
        name: str | None = None,
        tool_name: str | None = None,
    ) -> list[Step]:
        """Filter steps by name and/or tool_name. Both predicates AND
        together when both are passed. Returns a flat list."""
        out: list[Step] = []
        for s in self._steps:
            if name is not None and s.name != name:
                continue
            if tool_name is not None and s.attrs.get("tool_name") != tool_name:
                continue
            out.append(s)
        return out

    def replay_through(self, replay_fn: Callable[[Step], Step]) -> Replayer:
        """For each step in order, call `replay_fn(step) -> Step`. The
        returned Step replaces the input step in the new Replayer. The
        function is not required to keep the step's name or shape; it owns
        the output entirely."""
        new_steps: list[Step] = []
        for s in self._steps:
            result = replay_fn(s)
            if not isinstance(result, Step):
                raise TypeError(
                    f"replay_fn must return a Step, got {type(result).__name__}"
                )
            new_steps.append(result)
        return Replayer(new_steps)

    def export(self, path: str | Path) -> None:
        """Write all steps as JSONL to `path`. One step per line. Existing
        file is overwritten."""
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for s in self._steps:
                f.write(json.dumps(s.to_dict(), ensure_ascii=False))
                f.write("\n")


# ---------- StepEdit (named helper for callers who want a typed change spec) ----------


@dataclass(frozen=True)
class StepEdit:
    """Typed bundle of changes to pass through to `Replayer.edit_step`.

    `Replayer.edit_step` itself takes **kwargs for ergonomic use; this
    helper exists so callers building edits programmatically have a value
    type to pass around.
    """

    index: int
    changes: dict[str, Any]

    def apply(self, r: Replayer) -> Replayer:
        return r.edit_step(self.index, **self.changes)


# ---------- internals ----------


def _step_diff(idx: int, a: Step, b: Step) -> list[dict[str, Any]]:
    """Yield differences between two Steps at the same index."""
    out: list[dict[str, Any]] = []
    # spine
    for field_name in ("name", "started_at", "duration_ms", "run_id"):
        av = getattr(a, field_name)
        bv = getattr(b, field_name)
        if av != bv:
            out.append(
                {"step_index": idx, "field": field_name, "before": av, "after": bv}
            )
    # attrs - union of keys
    a_keys = set(a.attrs.keys())
    b_keys = set(b.attrs.keys())
    for key in sorted(a_keys | b_keys):
        av = a.attrs.get(key)
        bv = b.attrs.get(key)
        if av != bv:
            out.append(
                {"step_index": idx, "field": key, "before": av, "after": bv}
            )
    # meta - flat, dotted
    a_meta_keys = set(a.meta.keys())
    b_meta_keys = set(b.meta.keys())
    for key in sorted(a_meta_keys | b_meta_keys):
        av = a.meta.get(key)
        bv = b.meta.get(key)
        if av != bv:
            out.append(
                {"step_index": idx, "field": f"meta.{key}", "before": av, "after": bv}
            )
    return out
