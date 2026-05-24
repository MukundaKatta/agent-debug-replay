"""Tests for agent-debug-replay."""

from __future__ import annotations

import json

import pytest

from agent_debug_replay import DiffResult, Replayer, Step, StepEdit, Summary

# ---------- helpers ----------


def _write_jsonl(path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _sample_rows() -> list[dict]:
    return [
        {
            "name": "agent.start",
            "started_at": "2026-05-24T10:00:00Z",
            "duration_ms": 1.0,
            "run_id": "run-1",
            "input": "user prompt",
        },
        {
            "name": "llm.call",
            "started_at": "2026-05-24T10:00:01Z",
            "duration_ms": 800.0,
            "run_id": "run-1",
            "model": "claude-sonnet-4.6",
            "input": "user prompt",
            "output": "I'll search for it.",
            "tokens_in": 25,
            "tokens_out": 12,
            "cost_usd": 0.0015,
        },
        {
            "name": "tool.call",
            "started_at": "2026-05-24T10:00:02Z",
            "duration_ms": 400.0,
            "run_id": "run-1",
            "tool_name": "search_web",
            "tool_args": {"q": "python jsonl"},
            "tool_output": "results...",
        },
        {
            "name": "llm.call",
            "started_at": "2026-05-24T10:00:03Z",
            "duration_ms": 1200.0,
            "run_id": "run-1",
            "model": "claude-sonnet-4.6",
            "input": "summarize: results...",
            "output": "Here is the summary.",
            "tokens_in": 80,
            "tokens_out": 40,
            "cost_usd": 0.0048,
        },
        {
            "name": "agent.end",
            "started_at": "2026-05-24T10:00:04Z",
            "duration_ms": 0.5,
            "run_id": "run-1",
            "output": "Here is the summary.",
        },
    ]


# ---------- from_log / parsing ----------


def test_from_log_round_trips(tmp_path):
    log = tmp_path / "run.jsonl"
    rows = _sample_rows()
    _write_jsonl(log, rows)
    r = Replayer.from_log(log)
    assert len(r) == 5
    assert r.step(0).name == "agent.start"
    assert r.step(1).model == "claude-sonnet-4.6"
    assert r.step(1).cost_usd == 0.0015
    assert r.step(2).tool_name == "search_web"
    assert r.step(2).tool_args == {"q": "python jsonl"}


def test_from_log_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Replayer.from_log(tmp_path / "does-not-exist.jsonl")


def test_from_log_handles_empty_file(tmp_path):
    log = tmp_path / "empty.jsonl"
    log.write_text("", encoding="utf-8")
    r = Replayer.from_log(log)
    assert len(r) == 0
    assert r.steps == []
    assert list(r.iter()) == []


def test_from_log_skips_corrupt_lines(tmp_path, capsys):
    log = tmp_path / "mixed.jsonl"
    log.write_text(
        '{"name": "a"}\n'
        "{bad json line}\n"
        '{"name": "b"}\n'
        "\n"  # blank line ignored silently
        '"just a string"\n'  # non-object skipped
        '{"name": "c"}\n',
        encoding="utf-8",
    )
    r = Replayer.from_log(log)
    assert [s.name for s in r.iter()] == ["a", "b", "c"]
    err = capsys.readouterr().err
    assert "corrupt JSON" in err
    assert "non-object" in err


def test_from_log_expands_user(tmp_path, monkeypatch):
    # write to tmp_path then make ~ resolve to tmp_path so ~/foo.jsonl points
    # to the same file. covers the .expanduser() branch.
    monkeypatch.setenv("HOME", str(tmp_path))
    target = tmp_path / "run.jsonl"
    _write_jsonl(target, [{"name": "x"}])
    r = Replayer.from_log("~/run.jsonl")
    assert r.step(0).name == "x"


# ---------- from_steps ----------


def test_from_steps_direct():
    steps = [Step(name="a"), Step(name="b")]
    r = Replayer.from_steps(steps)
    assert len(r) == 2
    # input list copy: mutating the caller list must not change the Replayer
    steps.append(Step(name="c"))
    assert len(r) == 2


# ---------- iter / access ----------


def test_iter_yields_in_order(tmp_path):
    log = tmp_path / "run.jsonl"
    _write_jsonl(log, _sample_rows())
    r = Replayer.from_log(log)
    names = [s.name for s in r.iter()]
    assert names == ["agent.start", "llm.call", "tool.call", "llm.call", "agent.end"]
    # also via __iter__
    assert [s.name for s in r] == names


def test_steps_property_is_a_copy(tmp_path):
    log = tmp_path / "run.jsonl"
    _write_jsonl(log, _sample_rows())
    r = Replayer.from_log(log)
    listed = r.steps
    listed.append(Step(name="extra"))
    assert len(r) == 5  # original unchanged


def test_step_index_access(tmp_path):
    log = tmp_path / "run.jsonl"
    _write_jsonl(log, _sample_rows())
    r = Replayer.from_log(log)
    assert r.step(0).name == "agent.start"
    assert r.step(-1).name == "agent.end"


# ---------- edit_step ----------


def test_edit_step_returns_new_replayer_original_unchanged(tmp_path):
    log = tmp_path / "run.jsonl"
    _write_jsonl(log, _sample_rows())
    r = Replayer.from_log(log)
    edited = r.edit_step(index=3, output="new summary")
    assert edited is not r
    assert edited.step(3).output == "new summary"
    # original untouched
    assert r.step(3).output == "Here is the summary."


def test_edit_step_marks_downstream_stale(tmp_path):
    log = tmp_path / "run.jsonl"
    _write_jsonl(log, _sample_rows())
    r = Replayer.from_log(log)
    edited = r.edit_step(index=1, output="different llm output")
    # the edited step itself is NOT stale
    assert edited.step(1).meta.get("_stale") is not True
    # everything after is stale
    for i in range(2, len(edited)):
        assert edited.step(i).meta.get("_stale") is True
    # everything before is not stale
    assert edited.step(0).meta.get("_stale") is not True


def test_edit_step_out_of_range_raises(tmp_path):
    log = tmp_path / "run.jsonl"
    _write_jsonl(log, _sample_rows())
    r = Replayer.from_log(log)
    with pytest.raises(IndexError):
        r.edit_step(index=99, output="x")


def test_step_edit_typed_helper(tmp_path):
    log = tmp_path / "run.jsonl"
    _write_jsonl(log, _sample_rows())
    r = Replayer.from_log(log)
    e = StepEdit(index=1, changes={"output": "via StepEdit"})
    edited = e.apply(r)
    assert edited.step(1).output == "via StepEdit"


# ---------- diff ----------


def test_diff_finds_field_changes(tmp_path):
    log = tmp_path / "run.jsonl"
    _write_jsonl(log, _sample_rows())
    r = Replayer.from_log(log)
    edited = r.edit_step(index=3, output="changed!")
    diff = r.diff(edited)
    assert isinstance(diff, DiffResult)
    assert not diff.is_empty
    assert bool(diff) is True
    output_changes = [d for d in diff.differences if d["field"] == "output"]
    assert any(
        d["step_index"] == 3 and d["before"] == "Here is the summary." and d["after"] == "changed!"
        for d in output_changes
    )
    # downstream stale flag should also show up
    stale_changes = [d for d in diff.differences if d["field"] == "meta._stale"]
    assert any(d["step_index"] == 4 and d["after"] is True for d in stale_changes)


def test_diff_finds_added_and_removed_steps(tmp_path):
    log_a = tmp_path / "a.jsonl"
    log_b = tmp_path / "b.jsonl"
    rows = _sample_rows()
    _write_jsonl(log_a, rows[:3])
    _write_jsonl(log_b, rows)  # b has 2 extra
    a = Replayer.from_log(log_a)
    b = Replayer.from_log(log_b)
    diff = a.diff(b)
    added = [d for d in diff.differences if d["field"] == "<step>" and d["before"] is None]
    assert len(added) == 2
    assert added[0]["step_index"] == 3
    # also reverse direction (removed)
    diff_rev = b.diff(a)
    removed = [d for d in diff_rev.differences if d["field"] == "<step>" and d["after"] is None]
    assert len(removed) == 2


def test_diff_empty_when_identical(tmp_path):
    log = tmp_path / "run.jsonl"
    _write_jsonl(log, _sample_rows())
    r1 = Replayer.from_log(log)
    r2 = Replayer.from_log(log)
    diff = r1.diff(r2)
    assert diff.is_empty
    assert bool(diff) is False
    assert diff.differences == []


# ---------- summary ----------


def test_summary_totals_correct(tmp_path):
    log = tmp_path / "run.jsonl"
    _write_jsonl(log, _sample_rows())
    r = Replayer.from_log(log)
    s = r.summary()
    assert isinstance(s, Summary)
    assert s.step_count == 5
    # 1 + 800 + 400 + 1200 + 0.5
    assert s.total_duration_ms == pytest.approx(2401.5)
    # 0.0015 + 0.0048
    assert s.total_cost == pytest.approx(0.0063)
    assert s.by_step_name == {
        "agent.start": 1,
        "llm.call": 2,
        "tool.call": 1,
        "agent.end": 1,
    }


def test_summary_handles_missing_cost_and_duration():
    r = Replayer.from_steps(
        [
            Step(name="a"),
            Step(name="b", duration_ms=10.0, attrs={"cost_usd": 0.5}),
            Step(name="c", duration_ms="not-a-number", attrs={"cost_usd": "also-bad"}),
        ]
    )
    s = r.summary()
    assert s.step_count == 3
    assert s.total_duration_ms == pytest.approx(10.0)
    assert s.total_cost == pytest.approx(0.5)


# ---------- filter ----------


def test_filter_by_name(tmp_path):
    log = tmp_path / "run.jsonl"
    _write_jsonl(log, _sample_rows())
    r = Replayer.from_log(log)
    calls = r.filter(name="llm.call")
    assert len(calls) == 2
    assert all(s.name == "llm.call" for s in calls)


def test_filter_by_tool_name(tmp_path):
    log = tmp_path / "run.jsonl"
    _write_jsonl(log, _sample_rows())
    r = Replayer.from_log(log)
    tool_calls = r.filter(tool_name="search_web")
    assert len(tool_calls) == 1
    assert tool_calls[0].tool_name == "search_web"


def test_filter_by_name_and_tool_name(tmp_path):
    log = tmp_path / "run.jsonl"
    _write_jsonl(log, _sample_rows())
    r = Replayer.from_log(log)
    none_match = r.filter(name="llm.call", tool_name="search_web")
    assert none_match == []


def test_filter_no_predicates_returns_all(tmp_path):
    log = tmp_path / "run.jsonl"
    _write_jsonl(log, _sample_rows())
    r = Replayer.from_log(log)
    assert len(r.filter()) == 5


# ---------- export ----------


def test_export_round_trips(tmp_path):
    log = tmp_path / "run.jsonl"
    _write_jsonl(log, _sample_rows())
    r = Replayer.from_log(log)
    out = tmp_path / "out" / "round-trip.jsonl"  # nested dir on purpose
    r.export(out)
    r2 = Replayer.from_log(out)
    assert len(r2) == len(r)
    for a, b in zip(r.iter(), r2.iter(), strict=True):
        assert a.name == b.name
        assert a.attrs == b.attrs
        assert a.duration_ms == b.duration_ms


def test_export_after_edit_persists_stale_flag(tmp_path):
    log = tmp_path / "run.jsonl"
    _write_jsonl(log, _sample_rows())
    r = Replayer.from_log(log)
    edited = r.edit_step(index=1, output="new")
    out = tmp_path / "edited.jsonl"
    edited.export(out)
    reloaded = Replayer.from_log(out)
    assert reloaded.step(2).meta.get("_stale") is True
    assert reloaded.step(1).output == "new"


# ---------- replay_through ----------


def test_replay_through_calls_fn_per_step(tmp_path):
    log = tmp_path / "run.jsonl"
    _write_jsonl(log, _sample_rows())
    r = Replayer.from_log(log)
    calls = []

    def fn(step: Step) -> Step:
        calls.append(step.name)
        if step.name == "llm.call":
            return step.replace(output="REPLAYED")
        return step

    new_r = r.replay_through(fn)
    assert calls == [s.name for s in r.iter()]
    assert new_r is not r
    for s in new_r.iter():
        if s.name == "llm.call":
            assert s.output == "REPLAYED"
    # original untouched
    assert r.step(1).output == "I'll search for it."


def test_replay_through_rejects_non_step_return(tmp_path):
    log = tmp_path / "run.jsonl"
    _write_jsonl(log, _sample_rows())
    r = Replayer.from_log(log)
    with pytest.raises(TypeError):
        r.replay_through(lambda s: "not a step")  # type: ignore[arg-type]


# ---------- Step ----------


def test_step_replace_does_not_mutate_original():
    s = Step(name="a", attrs={"x": 1}, meta={"m": 2})
    s2 = s.replace(name="b", x=99, y=2)
    assert s.name == "a"
    assert s.attrs == {"x": 1}
    assert s2.name == "b"
    assert s2.attrs == {"x": 99, "y": 2}
    # meta is copied through unchanged
    assert s2.meta == {"m": 2}


def test_step_attribute_access_for_attrs():
    s = Step(name="a", attrs={"model": "gpt-5.4"})
    assert s.model == "gpt-5.4"
    with pytest.raises(AttributeError):
        _ = s.does_not_exist


def test_step_get_uniform():
    s = Step(name="a", duration_ms=5.0, attrs={"input": "hi"}, meta={"k": "v"})
    assert s.get("name") == "a"
    assert s.get("duration_ms") == 5.0
    assert s.get("input") == "hi"
    assert s.get("missing", "default") == "default"
    # meta returns the dict
    assert s.get("meta") == {"k": "v"}


def test_step_to_dict_round_trip():
    data = {
        "name": "llm.call",
        "started_at": "2026-05-24T10:00:00Z",
        "duration_ms": 100.0,
        "run_id": "run-1",
        "input": "hi",
        "output": "bye",
        "meta": {"_stale": True},
    }
    s = Step.from_dict(data)
    assert s.to_dict() == data
