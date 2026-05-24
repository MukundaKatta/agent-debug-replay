"""agent-debug-replay - whole-run replay for agent step-log JSONL files.

Load a JSONL trace, walk the steps, edit one and see downstream marked
stale, diff two runs field-by-field, filter by name or tool, and optionally
re-run each step through a user-supplied function. Zero runtime deps.

    from agent_debug_replay import Replayer

    r = Replayer.from_log("~/agent-runs/run.jsonl")
    for step in r.iter():
        print(step.name, step.input, step.output)

    edited = r.edit_step(index=3, output="new value")
    diff = r.diff(edited)
    edited.export("edited-run.jsonl")

Siblings:
  * agent-step-log: writes the JSONL this library consumes.
  * llm-fixture-replay: record/replay a single LLM HTTP call.
  * agentsnap: assert a whole trace matches a saved snapshot.
"""

from agent_debug_replay.replay import (
    DiffResult,
    Replayer,
    Step,
    StepEdit,
    Summary,
)

__version__ = "0.1.0"

__all__ = [
    "DiffResult",
    "Replayer",
    "Step",
    "StepEdit",
    "Summary",
    "__version__",
]
