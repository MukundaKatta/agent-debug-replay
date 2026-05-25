"""Tests for agent-debug-replay."""
import json
import tempfile
import os
import pytest
from agent_debug_replay import DebugReplay, ReplayStep, ReplayFilter, StepKind

SAMPLE_RECORDS = [
    {"kind": "message", "content": "Hello"},
    {"kind": "tool_call", "tool_name": "search", "content": "query"},
    {"kind": "tool_result", "tool_name": "search", "output": "results"},
    {"kind": "llm_response", "content": "Done"},
    {"kind": "error", "content": "oops"},
]


def make_replay() -> DebugReplay:
    return DebugReplay.from_dicts(SAMPLE_RECORDS)


def test_from_dicts_count():
    r = make_replay()
    assert r.total_steps == 5


def test_from_string():
    lines = "\n".join(json.dumps(d) for d in SAMPLE_RECORDS)
    r = DebugReplay.from_string(lines)
    assert r.total_steps == 5


def test_from_jsonl_file(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text("\n".join(json.dumps(d) for d in SAMPLE_RECORDS))
    r = DebugReplay.from_jsonl(str(path))
    assert r.total_steps == 5


def test_step_advances_cursor():
    r = make_replay()
    s = r.step()
    assert s is not None
    assert s.kind == "message"
    assert r.cursor == 1


def test_step_returns_none_at_end():
    r = make_replay()
    for _ in range(5):
        r.step()
    assert r.step() is None
    assert r.at_end


def test_step_n():
    r = make_replay()
    steps = r.step_n(3)
    assert len(steps) == 3
    assert r.cursor == 3


def test_reset():
    r = make_replay()
    r.step_n(3)
    r.reset()
    assert r.cursor == 0


def test_seek():
    r = make_replay()
    r.seek(2)
    assert r.cursor == 2
    assert r.current.kind == "tool_result"


def test_seek_out_of_range():
    r = make_replay()
    with pytest.raises(IndexError):
        r.seek(100)


def test_current_at_start():
    r = make_replay()
    assert r.current.kind == "message"


def test_steps_by_kind():
    r = make_replay()
    tool_calls = r.steps_by_kind("tool_call")
    assert len(tool_calls) == 1
    assert tool_calls[0].tool_name == "search"


def test_tool_calls_helper():
    r = make_replay()
    assert len(r.tool_calls()) == 1


def test_errors_helper():
    r = make_replay()
    assert len(r.errors()) == 1


def test_kind_counts():
    r = make_replay()
    counts = r.kind_counts()
    assert counts["message"] == 1
    assert counts["tool_call"] == 1
    assert counts["error"] == 1


def test_summary():
    r = make_replay()
    s = r.summary()
    assert s["total_steps"] == 5
    assert s["tool_calls"] == 1
    assert s["errors"] == 1


def test_filter_only_kinds():
    r = make_replay()
    f = ReplayFilter().only_kinds("tool_call", "tool_result")
    result = r.filter(f)
    assert len(result) == 2
    assert all(s.kind in ("tool_call", "tool_result") for s in result)


def test_filter_exclude_kinds():
    r = make_replay()
    f = ReplayFilter().exclude_kinds("error")
    result = r.filter(f)
    assert all(s.kind != "error" for s in result)


def test_filter_only_tools():
    r = make_replay()
    f = ReplayFilter().only_tools("search")
    result = r.filter(f)
    assert len(result) == 2  # tool_call + tool_result both have tool_name=search


def test_iterate():
    r = make_replay()
    steps = list(r.iterate())
    assert len(steps) == 5


def test_iterate_with_filter():
    r = make_replay()
    f = ReplayFilter().only_kinds("message")
    steps = list(r.iterate(f))
    assert len(steps) == 1


def test_len():
    r = make_replay()
    assert len(r) == 5


def test_replay_step_properties():
    s = ReplayStep(index=0, kind="tool_call", data={"kind": "tool_call", "tool_name": "calc"}, raw="{}")
    assert s.tool_name == "calc"
    assert s.timestamp is None


def test_from_string_skips_blank_lines():
    text = '{"kind":"message"}\n\n{"kind":"error"}'
    r = DebugReplay.from_string(text)
    assert r.total_steps == 2
