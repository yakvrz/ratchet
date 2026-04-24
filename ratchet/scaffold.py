from __future__ import annotations

from pathlib import Path


SUPPORTED_TEMPLATES = {"python_function", "python_cli"}


FUNCTION_AGENT_TEMPLATE = """from __future__ import annotations

from typing import Any


def run_agent(
    candidate: dict[str, str],
    case_payload: dict[str, Any],
    hooks: dict[str, Any] | None = None,
) -> dict[str, Any]:
    \"\"\"Replace this stub with your real agent invocation.

    `hooks` contains bounded Ratchet-managed code artifacts such as:
    - pre_tool_query_hook(query, context) -> str
    - post_tool_context_hook(cards, context) -> list[dict[str, str]]
    - post_answer_validator_hook(output, context) -> object

    Expected output keys:
    - output (JSON-serializable externally visible result)
    - raw_output_text (str, optional)
    - tool_calls (list[str], optional)
    - latency_s (float, optional)
    - input_tokens (int, optional)
    - output_tokens (int, optional)
    - total_tokens (int, optional)
    - cost_usd (float, optional)
    \"\"\"
    raise NotImplementedError("Replace run_agent() with your harness entrypoint.")
"""


CLI_AGENT_TEMPLATE = """from __future__ import annotations

import json
import sys
from typing import Any

from ratchet.code_artifacts import CodeArtifactLoader
from ratchet.types import CodeArtifactSpec


def run_agent(
    candidate: dict[str, str],
    case_payload: dict[str, Any],
    hooks: dict[str, Any] | None = None,
) -> dict[str, Any]:
    \"\"\"Replace this stub with your real CLI or subprocess-backed harness.\"\"\"
    raise NotImplementedError("Replace run_agent() with your CLI invocation.")


def main() -> None:
    request = json.loads(sys.stdin.read())
    loader = CodeArtifactLoader()
    hook_specs = [CodeArtifactSpec.from_dict(item) for item in request.get("hook_specs", [])]
    hook_sources = dict(request.get("hook_sources", {}))
    hooks = loader.build_hooks(hook_sources, hook_specs) if hook_specs else {}
    response = run_agent(
        candidate=dict(request["candidate"]),
        case_payload=dict(request["case"]),
        hooks=hooks,
    )
    sys.stdout.write(json.dumps(response, sort_keys=True))


if __name__ == "__main__":
    main()
"""


FUNCTION_ADAPTER_TEMPLATE = """from __future__ import annotations

import json
from pathlib import Path

from agent import run_agent
from ratchet.code_artifacts import CodeArtifactLoader, default_hook_source
from ratchet.grading import exact_text_grade, json_field_grade
from ratchet.types import (
    CodeArtifactSpec,
    ComponentSpec,
    DiagnosticTrace,
    EnumKnobSpec,
    EvalCase,
    GradeResult,
    OperationalMetrics,
    RunRecord,
    SearchSpace,
    TextArtifactSpec,
)


def build_search_space() -> SearchSpace:
    hook_specs = [
        CodeArtifactSpec(
            name="pre_tool_query_hook",
            language="python",
            callable_name="pre_tool_query_hook",
            signature="(query, context)",
            default=default_hook_source(
                CodeArtifactSpec(
                    name="pre_tool_query_hook",
                    language="python",
                    callable_name="pre_tool_query_hook",
                    signature="(query, context)",
                    default="",
                    max_chars=600,
                    max_lines=12,
                    description="Rewrite the tool query before lookup.",
                )
            ),
            max_chars=600,
            max_lines=12,
            depends_on={"search_tool_enabled": ["on"]},
            description="Bounded source-level hook that can rewrite a tool query before retrieval.",
        ),
        CodeArtifactSpec(
            name="post_tool_context_hook",
            language="python",
            callable_name="post_tool_context_hook",
            signature="(cards, context)",
            default=default_hook_source(
                CodeArtifactSpec(
                    name="post_tool_context_hook",
                    language="python",
                    callable_name="post_tool_context_hook",
                    signature="(cards, context)",
                    default="",
                    max_chars=600,
                    max_lines=12,
                    description="Rewrite retrieved context before answer generation.",
                )
            ),
            max_chars=600,
            max_lines=12,
            depends_on={"search_tool_enabled": ["on"]},
            description="Bounded source-level hook that can filter or reorder retrieved context.",
        ),
        CodeArtifactSpec(
            name="post_answer_validator_hook",
            language="python",
            callable_name="post_answer_validator_hook",
            signature="(output, context)",
            default=default_hook_source(
                CodeArtifactSpec(
                    name="post_answer_validator_hook",
                    language="python",
                    callable_name="post_answer_validator_hook",
                    signature="(output, context)",
                    default="",
                    max_chars=600,
                    max_lines=12,
                    description="Validate or rewrite the final answer payload.",
                )
            ),
            max_chars=600,
            max_lines=12,
            depends_on={"grounding_validator_enabled": ["on"]},
            description="Bounded source-level hook that can validate or rewrite the final answer payload.",
        ),
    ]
    return SearchSpace(
        enum_knobs=[
            EnumKnobSpec(
                name="model",
                kind="model",
                values=["primary", "cheaper"],
                default="primary",
                description="Example model knob. Replace with your real choices.",
            ),
            EnumKnobSpec(
                name="search_tool_enabled",
                kind="tool",
                values=["off", "on"],
                default="on",
                description="Whether the retrieval/search tool is available to the agent.",
            ),
        ],
        text_artifacts=[
            TextArtifactSpec(
                name="system_prompt",
                kind="prompt",
                default="Answer with the exact final text only.",
                max_chars=200,
                description="Editable prompt artifact. Ratchet may rewrite this directly.",
            ),
            TextArtifactSpec(
                name="search_tool_description",
                kind="tool",
                default="Search the project knowledge base for directly relevant facts.",
                max_chars=200,
                depends_on={"search_tool_enabled": ["on"]},
                description="Editable tool-description artifact. Replace with your real tool instructions.",
            ),
            TextArtifactSpec(
                name="grounding_validator_rule",
                kind="component",
                default="If the final answer is not supported by retrieved evidence, replace it with unknown.",
                max_chars=220,
                depends_on={"grounding_validator_enabled": ["on"]},
                description="Example component artifact. Replace with your real validator or guardrail policy.",
            ),
        ],
        components=[
            ComponentSpec(
                name="grounding_validator_enabled",
                kind="validator",
                values=["off", "on"],
                default="off",
                depends_on={"search_tool_enabled": ["on"]},
                description="Example structural component toggle for a post-answer grounding validator.",
            )
        ],
        code_artifacts=hook_specs,
    )


class ExternalAgentAdapter:
    def __init__(self) -> None:
        self._search_space = build_search_space()
        self._hook_loader = CodeArtifactLoader()

    def baseline(self) -> dict[str, str]:
        return {
            "model": "primary",
            "system_prompt": "Answer with the exact final text only.",
            "search_tool_enabled": "on",
            "search_tool_description": "Search the project knowledge base for directly relevant facts.",
            "grounding_validator_enabled": "off",
            "grounding_validator_rule": "If the final answer is not supported by retrieved evidence, replace it with unknown.",
        }

    def search_space(self) -> SearchSpace:
        return self._search_space

    def run_case(self, candidate: dict[str, str], case: EvalCase) -> RunRecord:
        hooks = self._hook_loader.build_hooks(candidate, self._search_space.code_artifacts)
        payload = run_agent(candidate=candidate, case_payload=case.to_dict(), hooks=hooks)
        return RunRecord(
            output=payload.get("output", payload.get("answer", "")),
            metrics=OperationalMetrics(
                latency_s=float(payload.get("latency_s", 0.0)),
                input_tokens=int(payload.get("input_tokens", 0)),
                output_tokens=int(payload.get("output_tokens", 0)),
                total_tokens=int(payload.get("total_tokens", 0)),
                cost_usd=float(payload.get("cost_usd", 0.0)),
                error=payload.get("error"),
            ),
            diagnostics=DiagnosticTrace(
                tool_calls=[str(item) for item in payload.get("tool_calls", [])],
                raw_output_text=str(payload.get("raw_output_text", payload.get("answer", ""))),
                metadata=dict(payload.get("metadata", {})),
            ),
        )

    def grade(self, case: EvalCase, output: object) -> GradeResult:
        # For exact-text tasks:
        return exact_text_grade(case, output)

        # For JSON field grading instead, use something like:
        # return json_field_grade(case, output, required_fields=["decision", "label"])

    def export(self, candidate: dict[str, str], out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "candidate.json").write_text(json.dumps(candidate, indent=2, sort_keys=True))


adapter = ExternalAgentAdapter()
"""


CLI_ADAPTER_TEMPLATE = """from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from ratchet.code_artifacts import default_hook_source
from ratchet.grading import exact_text_grade, json_field_grade
from ratchet.types import (
    CodeArtifactSpec,
    ComponentSpec,
    DiagnosticTrace,
    EnumKnobSpec,
    EvalCase,
    GradeResult,
    OperationalMetrics,
    RunRecord,
    SearchSpace,
    TextArtifactSpec,
)


def build_search_space() -> SearchSpace:
    hook_specs = [
        CodeArtifactSpec(
            name="pre_tool_query_hook",
            language="python",
            callable_name="pre_tool_query_hook",
            signature="(query, context)",
            default=default_hook_source(
                CodeArtifactSpec(
                    name="pre_tool_query_hook",
                    language="python",
                    callable_name="pre_tool_query_hook",
                    signature="(query, context)",
                    default="",
                    max_chars=600,
                    max_lines=12,
                    description="Rewrite the tool query before lookup.",
                )
            ),
            max_chars=600,
            max_lines=12,
            depends_on={"search_tool_enabled": ["on"]},
            description="Bounded source-level hook that can rewrite a tool query before retrieval.",
        ),
        CodeArtifactSpec(
            name="post_tool_context_hook",
            language="python",
            callable_name="post_tool_context_hook",
            signature="(cards, context)",
            default=default_hook_source(
                CodeArtifactSpec(
                    name="post_tool_context_hook",
                    language="python",
                    callable_name="post_tool_context_hook",
                    signature="(cards, context)",
                    default="",
                    max_chars=600,
                    max_lines=12,
                    description="Rewrite retrieved context before answer generation.",
                )
            ),
            max_chars=600,
            max_lines=12,
            depends_on={"search_tool_enabled": ["on"]},
            description="Bounded source-level hook that can filter or reorder retrieved context.",
        ),
        CodeArtifactSpec(
            name="post_answer_validator_hook",
            language="python",
            callable_name="post_answer_validator_hook",
            signature="(output, context)",
            default=default_hook_source(
                CodeArtifactSpec(
                    name="post_answer_validator_hook",
                    language="python",
                    callable_name="post_answer_validator_hook",
                    signature="(output, context)",
                    default="",
                    max_chars=600,
                    max_lines=12,
                    description="Validate or rewrite the final answer payload.",
                )
            ),
            max_chars=600,
            max_lines=12,
            depends_on={"grounding_validator_enabled": ["on"]},
            description="Bounded source-level hook that can validate or rewrite the final answer payload.",
        ),
    ]
    return SearchSpace(
        enum_knobs=[
            EnumKnobSpec(
                name="model",
                kind="model",
                values=["primary", "cheaper"],
                default="primary",
                description="Example model knob. Replace with your real choices.",
            ),
            EnumKnobSpec(
                name="search_tool_enabled",
                kind="tool",
                values=["off", "on"],
                default="on",
                description="Whether the retrieval/search tool is available to the agent.",
            ),
        ],
        text_artifacts=[
            TextArtifactSpec(
                name="system_prompt",
                kind="prompt",
                default="Answer with the exact final text only.",
                max_chars=200,
                description="Editable prompt artifact. Ratchet may rewrite this directly.",
            ),
            TextArtifactSpec(
                name="search_tool_description",
                kind="tool",
                default="Search the project knowledge base for directly relevant facts.",
                max_chars=200,
                depends_on={"search_tool_enabled": ["on"]},
                description="Editable tool-description artifact. Replace with your real tool instructions.",
            ),
            TextArtifactSpec(
                name="grounding_validator_rule",
                kind="component",
                default="If the final answer is not supported by retrieved evidence, replace it with unknown.",
                max_chars=220,
                depends_on={"grounding_validator_enabled": ["on"]},
                description="Example component artifact. Replace with your real validator or guardrail policy.",
            ),
        ],
        components=[
            ComponentSpec(
                name="grounding_validator_enabled",
                kind="validator",
                values=["off", "on"],
                default="off",
                depends_on={"search_tool_enabled": ["on"]},
                description="Example structural component toggle for a post-answer grounding validator.",
            )
        ],
        code_artifacts=hook_specs,
    )


class ExternalAgentAdapter:
    def __init__(self) -> None:
        self._search_space = build_search_space()

    def baseline(self) -> dict[str, str]:
        return {
            "model": "primary",
            "system_prompt": "Answer with the exact final text only.",
            "search_tool_enabled": "on",
            "search_tool_description": "Search the project knowledge base for directly relevant facts.",
            "grounding_validator_enabled": "off",
            "grounding_validator_rule": "If the final answer is not supported by retrieved evidence, replace it with unknown.",
        }

    def search_space(self) -> SearchSpace:
        return self._search_space

    def run_case(self, candidate: dict[str, str], case: EvalCase) -> RunRecord:
        request = json.dumps(
            {
                "candidate": candidate,
                "case": case.to_dict(),
                "hook_specs": [spec.to_dict() for spec in self._search_space.code_artifacts],
                "hook_sources": {
                    spec.name: candidate[spec.name] for spec in self._search_space.code_artifacts
                },
            },
            sort_keys=True,
        )
        command = [sys.executable, str(Path(__file__).with_name("agent_cli.py"))]
        env = os.environ.copy()
        python_path_entries = [entry for entry in sys.path if entry]
        existing_python_path = env.get("PYTHONPATH")
        if existing_python_path:
            python_path_entries.append(existing_python_path)
        env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(python_path_entries))
        completed = subprocess.run(
            command,
            input=request,
            text=True,
            capture_output=True,
            check=True,
            env=env,
        )
        payload = json.loads(completed.stdout)
        return RunRecord(
            output=payload.get("output", payload.get("answer", "")),
            metrics=OperationalMetrics(
                latency_s=float(payload.get("latency_s", 0.0)),
                input_tokens=int(payload.get("input_tokens", 0)),
                output_tokens=int(payload.get("output_tokens", 0)),
                total_tokens=int(payload.get("total_tokens", 0)),
                cost_usd=float(payload.get("cost_usd", 0.0)),
                error=payload.get("error"),
            ),
            diagnostics=DiagnosticTrace(
                tool_calls=[str(item) for item in payload.get("tool_calls", [])],
                raw_output_text=str(payload.get("raw_output_text", payload.get("answer", ""))),
                metadata=dict(payload.get("metadata", {})),
            ),
        )

    def grade(self, case: EvalCase, output: object) -> GradeResult:
        # For exact-text tasks:
        return exact_text_grade(case, output)

        # For JSON field grading instead, use something like:
        # return json_field_grade(case, output, required_fields=["decision", "label"])

    def export(self, candidate: dict[str, str], out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "candidate.json").write_text(json.dumps(candidate, indent=2, sort_keys=True))


adapter = ExternalAgentAdapter()
"""


README_TEMPLATE = """# Ratchet Integration Scaffold

This directory is a thin Ratchet adapter scaffold for your Python agent.

Files:
- `ratchet_adapter.py`: the adapter Ratchet imports
- `{agent_filename}`: replace the stub with your real agent invocation
- `evals.sample.jsonl`: minimal sample eval set
- `ratchet.toml`: config for `python3 -m ratchet run --config ratchet.toml`

Next steps:
1. Replace the stub in `{agent_filename}` with a real agent call.
2. Replace the example search space in `ratchet_adapter.py` with your real enum knobs, text artifacts, components, and code artifacts.
3. Keep externally visible outputs stable and grade only those outputs.
4. Replace `evals.sample.jsonl` with your own `dev` and `holdout` cases.
5. Run `python3 -m ratchet check --config ratchet.toml`.
6. Run `python3 -m ratchet run --config ratchet.toml`.

Constraints:
- Python only
- evals are required
- Ratchet changes only declared knobs and bounded artifacts
- Ratchet can rewrite bounded source-level hook logic
- Ratchet does not perform arbitrary repo-wide code rewriting
"""


RATCHET_TOML_TEMPLATE = """[ratchet]
adapter = "ratchet_adapter:adapter"
evals = "{evals_filename}"
out = "results/run"
env_file = ".env"
dev_budget = 20
holdout_top_k = 5
harnesser_model = "gpt-5.4"
harnesser_reasoning = "medium"
harnesser_enabled = true
max_case_retries = 2
case_timeout_s = 180
fail_fast = false
"""


EVALS_TEMPLATE = """{"id": "dev-sample", "split": "dev", "input": "Replace this with a real dev case.", "expected": "sample-answer", "metadata": {"category": "sample"}}
{"id": "holdout-sample", "split": "holdout", "input": "Replace this with a real holdout case.", "expected": "sample-answer", "metadata": {"category": "sample"}}
"""


def init_scaffold(out_dir: str | Path, template: str = "python_function") -> Path:
    if template not in SUPPORTED_TEMPLATES:
        raise ValueError(f"Unsupported scaffold template: {template}")
    root = Path(out_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Scaffold directory {root} already exists and is not empty.")
    root.mkdir(parents=True, exist_ok=True)

    if template == "python_function":
        agent_filename = "agent.py"
        adapter_body = FUNCTION_ADAPTER_TEMPLATE
        agent_body = FUNCTION_AGENT_TEMPLATE
    else:
        agent_filename = "agent_cli.py"
        adapter_body = CLI_ADAPTER_TEMPLATE
        agent_body = CLI_AGENT_TEMPLATE

    (root / "ratchet_adapter.py").write_text(adapter_body)
    (root / agent_filename).write_text(agent_body)
    (root / "ratchet.toml").write_text(
        RATCHET_TOML_TEMPLATE.format(evals_filename="evals.sample.jsonl")
    )
    (root / "evals.sample.jsonl").write_text(EVALS_TEMPLATE)
    (root / "README.md").write_text(README_TEMPLATE.format(agent_filename=agent_filename))
    return root
