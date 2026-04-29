from __future__ import annotations

from pathlib import Path


SUPPORTED_TEMPLATES = {"python_function", "python_cli"}


FUNCTION_AGENT_TEMPLATE = """from __future__ import annotations

from typing import Any


def build_model_input(case_payload: dict[str, Any]) -> Any:
    \"\"\"Map an eval case into the model input sent by the generated adapter.\"\"\"
    return case_payload["input"]


def parse_model_output(raw_output_text: str) -> object:
    \"\"\"Map the raw model response into the externally graded output.\"\"\"
    return raw_output_text.strip()


def create_model_client() -> object | None:
    \"\"\"Return a custom Responses-compatible model client, or None to use Ratchet's configured client.\"\"\"
    return None
"""


CLI_AGENT_TEMPLATE = """from __future__ import annotations

import json
import sys
from typing import Any


def build_model_input(case_payload: dict[str, Any]) -> Any:
    \"\"\"Map an eval case into the model input sent by the generated adapter.\"\"\"
    return case_payload["input"]


def create_model_client() -> object | None:
    \"\"\"Return a custom Responses-compatible model client, or None to use Ratchet's configured client.\"\"\"
    return None


def main() -> None:
    request = json.loads(sys.stdin.read())
    response = {"input": build_model_input(dict(request["case"]))}
    sys.stdout.write(json.dumps(response, sort_keys=True))


if __name__ == "__main__":
    main()
"""


FUNCTION_ADAPTER_TEMPLATE = """from __future__ import annotations

import os

from agent import build_model_input, create_model_client, parse_model_output
from ratchet.adapter_generation import (
    GeneratedSingleCallAdapter,
    ModelRequest,
    context_graph_from_spec,
    model_config_from_spec,
)
from ratchet.grading import exact_text_grade
from ratchet.types import (
    AgentSpec,
    EvalCase,
    GradeResult,
)


BASE_SPEC = AgentSpec(
    name="scaffolded-python-function-agent",
    model="gpt-5.4-mini",
    model_options=["gpt-5.4-mini", "gpt-5.4-nano"],
    instructions={"system_prompt": "Answer with the exact final text only."},
    output_contract="Return the exact expected answer text.",
    runtime={"output_cap": 128},
)


class ScaffoldHarness:
    def agent_spec(self) -> AgentSpec:
        return BASE_SPEC

    def build_model_request(self, spec: AgentSpec, case: EvalCase) -> ModelRequest:
        return ModelRequest(
            context=context_graph_from_spec(spec, include_output_contract=True),
            input=build_model_input(case.to_dict()),
            model_config=model_config_from_spec(spec),
        )

    def parse_output(self, raw_output_text: str) -> object:
        return parse_model_output(raw_output_text)

    def grade(self, case: EvalCase, output: object) -> GradeResult:
        return exact_text_grade(case, output)


class ScaffoldAdapter(GeneratedSingleCallAdapter):
    def __init__(self, env_path: str | None = None, client: object | None = None) -> None:
        super().__init__(
            harness=ScaffoldHarness(),
            env_path=env_path or os.environ.get("RATCHET_ENV_FILE", ".env"),
            client=client or create_model_client(),
        )

adapter = ScaffoldAdapter()
"""


CLI_ADAPTER_TEMPLATE = """from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from agent_cli import create_model_client
from ratchet.adapter_generation import (
    GeneratedSingleCallAdapter,
    ModelRequest,
    context_graph_from_spec,
    model_config_from_spec,
)
from ratchet.grading import exact_text_grade
from ratchet.types import (
    AgentSpec,
    EvalCase,
    GradeResult,
)


BASE_SPEC = AgentSpec(
    name="scaffolded-python-cli-agent",
    model="gpt-5.4-mini",
    model_options=["gpt-5.4-mini", "gpt-5.4-nano"],
    instructions={"system_prompt": "Answer with the exact final text only."},
    output_contract="Return the exact expected answer text.",
    runtime={"output_cap": 128},
)


class ScaffoldCliHarness:
    def agent_spec(self) -> AgentSpec:
        return BASE_SPEC

    def build_model_request(self, spec: AgentSpec, case: EvalCase) -> ModelRequest:
        request = json.dumps({"case": case.to_dict()}, sort_keys=True)
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("agent_cli.py"))],
            input=request,
            text=True,
            capture_output=True,
            check=True,
        )
        payload = json.loads(completed.stdout)
        return ModelRequest(
            context=context_graph_from_spec(spec, include_output_contract=True),
            input=payload["input"],
            model_config=model_config_from_spec(spec),
        )

    def parse_output(self, raw_output_text: str) -> object:
        return raw_output_text.strip()

    def grade(self, case: EvalCase, output: object) -> GradeResult:
        return exact_text_grade(case, output)


class ScaffoldCliAdapter(GeneratedSingleCallAdapter):
    def __init__(self, env_path: str | None = None, client: object | None = None) -> None:
        super().__init__(
            harness=ScaffoldCliHarness(),
            env_path=env_path or os.environ.get("RATCHET_ENV_FILE", ".env"),
            client=client or create_model_client(),
        )

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
# research_theorist_model = "gpt-5.4"
# research_planner_model = "gpt-5.4"
# candidate_implementer_model = "gpt-5.4"
# measurement_selector_model = "gpt-5.4"
samples_per_case = 1

[ratchet.objective]
mode = "correctness"

[ratchet.objective.constraints]
allowed_edits = ["instruction", "model", "runtime", "output"]
allowed_models = ["gpt-5.4-mini", "gpt-5.4-nano"]
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

The adapter exposes a small harness: `AgentSpec`, model-input construction, output parsing, and grading.
Ratchet generates the optimization surface, runtime hooks, model call, instrumentation, and export path.
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
