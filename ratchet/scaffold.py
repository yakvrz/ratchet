from __future__ import annotations

from pathlib import Path


SUPPORTED_TEMPLATES = {"python_function", "python_cli"}


FUNCTION_AGENT_TEMPLATE = """from __future__ import annotations

from typing import Any


def run_agent(spec: dict[str, Any], case_payload: dict[str, Any]) -> dict[str, Any]:
    \"\"\"Replace this stub with your real agent invocation.

    `spec` is the Ratchet AgentSpec after an optional patch has been applied.
    Return a JSON-serializable externally visible output plus optional metrics:
    - output
    - raw_output_text
    - tool_calls
    - turns
    - terminal_reason
    - latency_s
    - input_tokens
    - output_tokens
    - total_tokens
    - cost_usd
    \"\"\"
    raise NotImplementedError("Replace run_agent() with your agent entrypoint.")
"""


CLI_AGENT_TEMPLATE = """from __future__ import annotations

import json
import sys
from typing import Any


def run_agent(spec: dict[str, Any], case_payload: dict[str, Any]) -> dict[str, Any]:
    \"\"\"Replace this stub with your real CLI or subprocess-backed agent.\"\"\"
    raise NotImplementedError("Replace run_agent() with your CLI invocation.")


def main() -> None:
    request = json.loads(sys.stdin.read())
    response = run_agent(
        spec=dict(request["spec"]),
        case_payload=dict(request["case"]),
    )
    sys.stdout.write(json.dumps(response, sort_keys=True))


if __name__ == "__main__":
    main()
"""


FUNCTION_ADAPTER_TEMPLATE = """from __future__ import annotations

import json
from pathlib import Path

from agent import run_agent
from ratchet.grading import exact_text_grade
from ratchet.types import (
    AgentPatch,
    AgentSpec,
    DiagnosticTrace,
    EvalCase,
    GradeResult,
    InteractionTurn,
    OperationalMetrics,
    RunRecord,
)


BASE_SPEC = AgentSpec(
    name="scaffolded-python-function-agent",
    model="primary",
    model_options=["primary", "cheaper"],
    instructions={"system_prompt": "Answer with the exact final text only."},
    output_contract="Return the exact expected answer text.",
    runtime={"output_cap": 128},
)


class ScaffoldAdapter:
    def agent_spec(self) -> AgentSpec:
        return BASE_SPEC

    def run_case(self, case: EvalCase, patch: AgentPatch | None = None) -> RunRecord:
        spec = BASE_SPEC.apply_patch(patch)
        payload = run_agent(spec.to_dict(), case.to_dict())
        total_tokens = int(payload.get("total_tokens", int(payload.get("input_tokens", 0)) + int(payload.get("output_tokens", 0))))
        output = payload.get("output")
        raw_output = str(payload.get("raw_output_text", output if output is not None else ""))
        turns = [
            InteractionTurn.from_dict(turn)
            for turn in payload.get("turns", [])
            if isinstance(turn, dict)
        ]
        return RunRecord(
            output=output,
            metrics=OperationalMetrics(
                latency_s=float(payload.get("latency_s", 0.0)),
                input_tokens=int(payload.get("input_tokens", 0)),
                output_tokens=int(payload.get("output_tokens", 0)),
                total_tokens=total_tokens,
                cost_usd=float(payload.get("cost_usd", 0.0)),
                model_calls=int(payload.get("model_calls", 1)),
                tool_calls=int(payload.get("tool_call_count", len(payload.get("tool_calls", [])))),
                turns=int(payload.get("turn_count", len(turns) or 1)),
            ),
            diagnostics=DiagnosticTrace(
                tool_calls=[str(item) for item in payload.get("tool_calls", [])],
                raw_output_text=raw_output,
                turns=turns,
                terminal_state=dict(payload.get("terminal_state", {})),
                terminal_reason=str(payload.get("terminal_reason", "")),
            ),
        )

    def grade(self, case: EvalCase, output: object) -> GradeResult:
        return exact_text_grade(case, output)

    def export(self, patch: AgentPatch, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "patch.json").write_text(json.dumps(patch.to_dict(), indent=2, sort_keys=True))
        (out_dir / "agent_spec.json").write_text(json.dumps(BASE_SPEC.apply_patch(patch).to_dict(), indent=2, sort_keys=True))


adapter = ScaffoldAdapter()
"""


CLI_ADAPTER_TEMPLATE = """from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from ratchet.grading import exact_text_grade
from ratchet.types import (
    AgentPatch,
    AgentSpec,
    DiagnosticTrace,
    EvalCase,
    GradeResult,
    InteractionTurn,
    OperationalMetrics,
    RunRecord,
)


BASE_SPEC = AgentSpec(
    name="scaffolded-python-cli-agent",
    model="primary",
    model_options=["primary", "cheaper"],
    instructions={"system_prompt": "Answer with the exact final text only."},
    output_contract="Return the exact expected answer text.",
    runtime={"output_cap": 128},
)


class ScaffoldCliAdapter:
    def agent_spec(self) -> AgentSpec:
        return BASE_SPEC

    def run_case(self, case: EvalCase, patch: AgentPatch | None = None) -> RunRecord:
        spec = BASE_SPEC.apply_patch(patch)
        request = json.dumps({"spec": spec.to_dict(), "case": case.to_dict()}, sort_keys=True)
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("agent_cli.py"))],
            input=request,
            text=True,
            capture_output=True,
            check=True,
        )
        payload = json.loads(completed.stdout)
        total_tokens = int(payload.get("total_tokens", int(payload.get("input_tokens", 0)) + int(payload.get("output_tokens", 0))))
        output = payload.get("output")
        raw_output = str(payload.get("raw_output_text", output if output is not None else ""))
        turns = [
            InteractionTurn.from_dict(turn)
            for turn in payload.get("turns", [])
            if isinstance(turn, dict)
        ]
        return RunRecord(
            output=output,
            metrics=OperationalMetrics(
                latency_s=float(payload.get("latency_s", 0.0)),
                input_tokens=int(payload.get("input_tokens", 0)),
                output_tokens=int(payload.get("output_tokens", 0)),
                total_tokens=total_tokens,
                cost_usd=float(payload.get("cost_usd", 0.0)),
                model_calls=int(payload.get("model_calls", 1)),
                tool_calls=int(payload.get("tool_call_count", len(payload.get("tool_calls", [])))),
                turns=int(payload.get("turn_count", len(turns) or 1)),
            ),
            diagnostics=DiagnosticTrace(
                tool_calls=[str(item) for item in payload.get("tool_calls", [])],
                raw_output_text=raw_output,
                turns=turns,
                terminal_state=dict(payload.get("terminal_state", {})),
                terminal_reason=str(payload.get("terminal_reason", "")),
            ),
        )

    def grade(self, case: EvalCase, output: object) -> GradeResult:
        return exact_text_grade(case, output)

    def export(self, patch: AgentPatch, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "patch.json").write_text(json.dumps(patch.to_dict(), indent=2, sort_keys=True))
        (out_dir / "agent_spec.json").write_text(json.dumps(BASE_SPEC.apply_patch(patch).to_dict(), indent=2, sort_keys=True))


adapter = ScaffoldCliAdapter()
"""


CONFIG_TEMPLATE = """[ratchet]
adapter = "ratchet_adapter:adapter"
evals = "evals.sample.jsonl"
out = "results/run"
env_file = ".env"
dev_budget = 8
holdout_budget = 3
optimizer_model = "gpt-5.4"
optimizer_reasoning = "medium"
# Optional per-role overrides:
# diagnoser_model = "gpt-5.4"
# research_planner_model = "gpt-5.4"
# candidate_implementer_model = "gpt-5.4"
# measurement_selector_model = "gpt-5.4"
samples_per_case = 1

[ratchet.objective]
mode = "correctness"

[ratchet.objective.constraints]
allowed_edits = ["instruction", "model", "runtime", "output"]
allowed_models = ["primary", "cheaper"]
max_latency_ratio = 1.1
"""


EVALS_TEMPLATE = """{"id": "dev-1", "split": "dev", "input": "replace me", "expected": "replace me", "metadata": {"category": "sample"}}
{"id": "holdout-1", "split": "holdout", "input": "replace me", "expected": "replace me", "metadata": {"category": "sample"}}
"""


README_TEMPLATE = """# Ratchet Agent Scaffold

Wire your agent in `agent.py` or `agent_cli.py`, then run:

```bash
python -m ratchet check --config ratchet.toml
python -m ratchet optimize --config ratchet.toml
```

The adapter exposes only a descriptive `AgentSpec`, a baseline run method, grading, and export.
Ratchet generates the optimization surface and patches itself.
"""


def init_scaffold(out_dir: str | Path, template: str = "python_function") -> Path:
    if template not in SUPPORTED_TEMPLATES:
        raise ValueError(f"Unsupported template {template!r}. Choose one of {sorted(SUPPORTED_TEMPLATES)}.")
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    if template == "python_function":
        (root / "agent.py").write_text(FUNCTION_AGENT_TEMPLATE)
        (root / "ratchet_adapter.py").write_text(FUNCTION_ADAPTER_TEMPLATE)
    else:
        (root / "agent_cli.py").write_text(CLI_AGENT_TEMPLATE)
        (root / "ratchet_adapter.py").write_text(CLI_ADAPTER_TEMPLATE)
    (root / "ratchet.toml").write_text(CONFIG_TEMPLATE)
    (root / "evals.sample.jsonl").write_text(EVALS_TEMPLATE)
    (root / "README.md").write_text(README_TEMPLATE)
    return root
