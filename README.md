# agent-debug-replay

Step-through replay of JSONL agent run traces.

```python
from agent_debug_replay import DebugReplay, ReplayFilter, StepKind

replay = DebugReplay.from_jsonl("run.jsonl")

# step through
while not replay.at_end:
    s = replay.step()
    print(s.kind, s.content)

# filter to tool calls only
f = ReplayFilter().only_kinds(StepKind.TOOL_CALL.value)
tool_steps = replay.filter(f)

# summary
print(replay.summary())
```

Load from JSONL file, string, or list of dicts. Navigate with `.step()`, `.step_n()`, `.seek()`, `.reset()`. Filter with `ReplayFilter`.
