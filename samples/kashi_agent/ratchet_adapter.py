from __future__ import annotations

import json
import os
from pathlib import Path
import re
from hashlib import sha256

from ratchet.grading import extract_json_payload
from ratchet.model_client import ResponsesModelClient
from ratchet.types import AgentPatch, AgentSpec, EvalCase, GradeResult, RunRecord

try:
    from agent import KashiAgentRunner
except ModuleNotFoundError:
    from .agent import KashiAgentRunner


BASELINE_ASSISTANT_PROMPT = (
    "You are a helpful assistant for The General Insurance. Your job is to respond to user "
    "inquiries using a predefined set of reason-response mappings. Always be polite, "
    "professional, and helpful. If the user's question matches one of the known reasons, "
    "respond with the corresponding message. If it does not match any known reason, use the "
    "fallback response labeled 'Any question we don't have a preset message for'. Do not use "
    "standard markdown syntax in the URL. Do not add or remove or change response punctuation."
)

BASELINE_MAPPING_RULE = (
    "Select the reason-response mapping from the user's most recent message and the prior "
    "conversation. Apply CallHours exactly. Return only the mandated response text; do not "
    "explain the mapping decision."
)

BASE_SPEC = AgentSpec(
    name="kashi-agent",
    model="gpt-4o-2024-08-06",
    model_options=[
        "gpt-4o-2024-08-06",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "gpt-5.2",
        "gpt-5.4",
    ],
    instructions={
        "assistant_prompt": BASELINE_ASSISTANT_PROMPT,
        "mapping_selection_rule": BASELINE_MAPPING_RULE,
    },
    output_contract="Return only the mandated response text for the selected reason-response mapping.",
    runtime={"reasoning_effort": "none", "output_cap": 160},
)


class KashiJudge:
    def __init__(self, env_path: str, judge_prompt_path: Path) -> None:
        self.client = ResponsesModelClient(env_path=env_path)
        self.template = judge_prompt_path.read_text()

    def grade(self, case: EvalCase, output: object) -> GradeResult:
        prompt = self.template
        replacements = {
            "input_messages": case.input,
            "eastern_time": str(case.metadata.get("eastern_time", "")),
            "original_output": str(output),
        }
        for key, value in replacements.items():
            prompt = prompt.replace("{" + key + "}", value)
            prompt = prompt.replace("{" + key.replace("_", "\\_") + "}", value)
        response = self.client.create_response(
            model="gpt-5.2",
            reasoning={"effort": "low"},
            input=prompt,
            max_output_tokens=1200,
            text={
                "verbosity": "low",
                "format": {
                    "type": "json_schema",
                    "name": "kashi_judgment",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string", "enum": ["correct", "incorrect"]},
                            "explanation": {"type": "string"},
                        },
                        "required": ["label", "explanation"],
                        "additionalProperties": False,
                    },
                },
            },
        )
        payload = extract_json_payload(response.output_text) or {}
        label = str(payload.get("label", "incorrect")).strip().lower()
        if label not in {"correct", "incorrect"}:
            label_match = re.search(r'"label"\s*:\s*"(correct|incorrect)"', response.output_text)
            if label_match:
                label = label_match.group(1)
        explanation = str(payload.get("explanation", response.output_text))
        passed = label == "correct"
        return GradeResult(
            score=1.0 if passed else 0.0,
            passed=passed,
            labels=[] if passed else ["judge_incorrect"],
            notes=explanation,
        )


class KashiAdapter:
    def __init__(
        self,
        env_path: str | None = None,
        runner: KashiAgentRunner | None = None,
        judge: KashiJudge | None = None,
    ) -> None:
        self.env_path = env_path or os.environ.get("RATCHET_ENV_FILE", ".env")
        self._runner = runner
        self._judge = judge

    def agent_spec(self) -> AgentSpec:
        return BASE_SPEC

    def run_case(self, case: EvalCase, patch: AgentPatch | None = None) -> RunRecord:
        if self._runner is None:
            self._runner = KashiAgentRunner(env_path=self.env_path)
        return self._runner.run_case(BASE_SPEC.apply_patch(patch), case)

    def grade(self, case: EvalCase, output: object) -> GradeResult:
        if self._judge is None:
            self._judge = KashiJudge(
                env_path=self.env_path,
                judge_prompt_path=Path(__file__).with_name("judge_prompt.md"),
            )
        return self._judge.grade(case, output)

    def export(self, patch: AgentPatch, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        spec = BASE_SPEC.apply_patch(patch)
        (out_dir / "patch.json").write_text(json.dumps(patch.to_dict(), indent=2, sort_keys=True))
        (out_dir / "agent_spec.json").write_text(json.dumps(spec.to_dict(), indent=2, sort_keys=True))
        (out_dir / "README.md").write_text(
            "# Exported Kashi Patch\n\n"
            "This bundle contains the selected Ratchet V2 patch. The judge remains frozen at gpt-5.2.\n"
        )

    def fingerprint(self) -> dict[str, str]:
        prompt_text = Path(__file__).with_name("judge_prompt.md").read_text()
        return {
            "judge_prompt_sha256": sha256(prompt_text.encode("utf-8")).hexdigest(),
            "judge_model": "gpt-5.2",
        }


adapter = KashiAdapter()
