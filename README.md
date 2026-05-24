# agent-debug-replay

[![PyPI](https://img.shields.io/pypi/v/agent-debug-replay.svg)](https://pypi.org/project/agent-debug-replay/)
[![Python](https://img.shields.io/pypi/pyversions/agent-debug-replay.svg)](https://pypi.org/project/agent-debug-replay/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Whole-run replay for agent step-log JSONL files.**

You logged a long agent run with [`agent-step-log`](https://github.com/MukundaKatta/agent-step-log).
The run did the wrong thing at step 3. You want to know what would change downstream
if step 3 had a different output, without re-running the whole agent.

`agent-debug-replay` loads the JSONL, lets you walk the steps, edit one in place
(returns a new `Replayer`, original unchanged), flags downstream steps as `_stale`,
diffs two runs field-by-field, filters by name or tool, and (optionally) re-runs each
step through a user-supplied function.

Zero runtime deps. Pure stdlib.

## Install

```bash
pip install agent-debug-replay
```

## Walk a run

```python
from agent_debug_replay import Replayer

r = Replayer.from_log("~/agent-runs/run-2026-05-24.jsonl")

for step in r.iter():
    print(f"{step.name}: input={step.input!r} output={step.output!r}")

print(len(r.steps), "steps")
print(r.step(3).output)
```

## Edit a step, see downstream marked stale

`edit_step` returns a new `Replayer`. The original is untouched. All steps with a
higher index than the edited one get `_stale=True` in their `meta` so you know
their captured outputs may no longer be valid.

```python
edited = r.edit_step(index=3, output="new output value")

assert edited is not r
assert edited.step(3).output == "new output value"
assert edited.step(4).meta.get("_stale") is True
```

## Diff two runs

```python
diff = r.diff(edited)
for change in diff.differences:
    print(change)
# {'step_index': 3, 'field': 'output', 'before': '...', 'after': 'new output value'}
# {'step_index': 4, 'field': 'meta._stale', 'before': None, 'after': True}
```

`diff` also reports added or removed steps if the two runs differ in length.

## Filter

```python
tool_steps = r.filter(tool_name="search_web")
llm_steps = r.filter(name="llm.call")
```

## Summary

```python
s = r.summary()
s.step_count
s.total_cost
s.total_duration_ms
s.by_step_name   # {"llm.call": 12, "tool.call": 4, ...}
```

## replay_through: simulate re-running

Pass a function that takes a `Step` and returns a (possibly modified) `Step`.
Useful when you want to score, rewrite, or re-execute each step under a new model
or prompt and capture the result as a fresh run.

```python
def rerun(step):
    if step.name == "llm.call":
        # call your model with step.input, return a new Step with the new output
        new_output = my_model(step.input)
        return step.replace(output=new_output)
    return step

replayed = r.replay_through(rerun)
replayed.export("rerun.jsonl")
```

## Export

```python
edited.export("edited-run.jsonl")
```

Round-trips. `Replayer.from_log("edited-run.jsonl")` parses it back.

## Where it fits

Three sibling libraries cover three different replay shapes:

- [`agent-step-log`](https://github.com/MukundaKatta/agent-step-log) writes the JSONL.
- [`agent-debug-replay`](https://github.com/MukundaKatta/agent-debug-replay) (this) loads, walks, edits, and diffs a whole run.
- [`llm-fixture-replay`](https://github.com/MukundaKatta/llm-fixture-replay) replays a single LLM HTTP call from a recorded fixture.
- [`agentsnap`](https://github.com/MukundaKatta/agentsnap) asserts a whole agent trace matches a saved snapshot.

## What it does NOT do

- It does not actually re-execute LLM calls or tools. `replay_through` is a hook
  for you to pass a function. If you want HTTP-level call replay, reach for
  `llm-fixture-replay`.
- It does not validate the JSONL schema beyond best-effort parsing. Corrupt
  lines are skipped with a warning on stderr.
- It does not modify files in place. Editing returns a new `Replayer`. Writing
  back happens only when you call `.export(path)`.

## License

MIT
