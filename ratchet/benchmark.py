from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Iterable

from ratchet.types import EvalCase


@dataclass(frozen=True)
class KnowledgeDoc:
    doc_id: str
    title: str
    body: str
    distilled: str


@dataclass(frozen=True)
class Task:
    task_id: str
    split: str
    category: str
    question: str
    answer_type: str
    expected_text: str | None = None
    expected_number: float | None = None
    tolerance: float = 0.01
    aliases: tuple[str, ...] = field(default_factory=tuple)

    @property
    def needs_calculator(self) -> bool:
        return self.category == "math"

    @property
    def needs_knowledge(self) -> bool:
        return True

    @property
    def canonical_answer(self) -> str:
        if self.answer_type == "number":
            assert self.expected_number is not None
            return f"{self.expected_number:.2f}".rstrip("0").rstrip(".")
        assert self.expected_text is not None
        return self.expected_text


KNOWLEDGE_BASE: tuple[KnowledgeDoc, ...] = (
    KnowledgeDoc(
        doc_id="refunds_and_exceptions",
        title="Refunds and exception approvals",
        body=(
            "Northstar Fulfillment uses a tiered approval policy for customer-facing credits and "
            "exceptions. Goodwill credits up to $50 can be approved by the floor lead on shift. "
            "Credits from $51 through $150 move to the finance desk. Any goodwill credit above "
            "$150 requires the finance duty manager. Lost parcels are escalated under code OP-17. "
            "When dock teams uncover an invoice variance above two percent, the escalation owner "
            "is the revenue controller rather than warehouse operations. These rules are restated "
            "in training decks and weekly dispatch notes, but the handbook remains the source of truth."
        ),
        distilled=(
            "goodwill credit <= $50 -> floor lead\n"
            "goodwill credit $51-$150 -> finance desk\n"
            "goodwill credit > $150 -> finance duty manager\n"
            "lost parcel escalation code -> OP-17\n"
            "invoice variance > 2% approver -> revenue controller"
        ),
    ),
    KnowledgeDoc(
        doc_id="security_and_retention",
        title="Security, retention, and emergency channels",
        body=(
            "Northstar separates retention windows by geography and incident type. EU debug logs are "
            "retained for 14 days, while US debug logs are retained for 30 days. Cold-chain incident "
            "photos are preserved for 45 days because customer claims lag physical scans. Vendor access "
            "badges must be renewed every 90 days. If a production incident requires emergency privilege "
            "escalation, staff use the sev-override channel. These controls exist to keep access short-lived "
            "and traceable across partner sites."
        ),
        distilled=(
            "EU debug logs retention -> 14 days\n"
            "US debug logs retention -> 30 days\n"
            "cold-chain incident photos retention -> 45 days\n"
            "vendor access badge renewal -> 90 days\n"
            "break-glass override channel -> sev-override"
        ),
    ),
    KnowledgeDoc(
        doc_id="ops_rhythm_and_labels",
        title="Operations rhythm, labeling, and routing",
        body=(
            "The weekend warehouse cutover starts at 22:30 local time to avoid overlap with the late "
            "linehaul sweep. Hazardous parcels receive an amber sticker, not red, because the red label "
            "is reserved for quarantine holds. LATAM customs paperwork belongs in the border-ops-latam "
            "queue. Cooler replenishment is triggered when only six packs remain in reserve. Premium chat "
            "tickets have a 2 minute first-response SLA, while the standard email queue has a 45 minute "
            "first-response SLA. APAC standard delivery commitments are modeled as a 4 day window."
        ),
        distilled=(
            "weekend warehouse cutover -> 22:30\n"
            "hazardous parcel sticker -> amber\n"
            "LATAM customs queue -> border-ops-latam\n"
            "cooler replenishment threshold -> 6 packs\n"
            "premium chat first-response SLA -> 2 minutes\n"
            "standard email first-response SLA -> 45 minutes\n"
            "APAC standard delivery window -> 4 days"
        ),
    ),
    KnowledgeDoc(
        doc_id="pricing_and_fees",
        title="Pricing and handling fees",
        body=(
            "Northstar's fee deck is compact but easy to misread under pressure. The Zone B same-day "
            "surcharge is $7.50. Frozen parcels routed through Zone C add a $12.25 surcharge. Return "
            "handling fees are $3.20 for lite returns, $5.40 for standard returns, and $9.80 for oversize "
            "returns. Dry-ice replenishment packs are billed at $1.75 each. Priority picks add $2.10 per "
            "order. Shipment insurance is billed at $0.85 for every $100 of declared value, including "
            "partial hundreds rounded down by the agent before calculation."
        ),
        distilled=(
            "Zone B same-day surcharge -> $7.50\n"
            "Zone C frozen surcharge -> $12.25\n"
            "lite return handling fee -> $3.20\n"
            "standard return handling fee -> $5.40\n"
            "oversize return handling fee -> $9.80\n"
            "dry-ice pack -> $1.75\n"
            "priority pick fee -> $2.10\n"
            "insurance per $100 declared value -> $0.85"
        ),
    ),
    KnowledgeDoc(
        doc_id="dock_and_count_rules",
        title="Dock scheduling and count-control rules",
        body=(
            "Dock teams get a 20 minute reslot grace period before the appointment is treated as a miss. "
            "Pallet recounts are capped at 3 before a supervisor must sign off. These values are small but "
            "important because many downstream exceptions come from late arrivals or repeated recount loops."
        ),
        distilled=(
            "dock reslot grace period -> 20 minutes\n"
            "pallet recount cap before escalation -> 3"
        ),
    ),
)


def _text_task(
    task_id: str,
    split: str,
    category: str,
    question: str,
    expected_text: str,
    *aliases: str,
) -> Task:
    return Task(
        task_id=task_id,
        split=split,
        category=category,
        question=question,
        answer_type="text",
        expected_text=expected_text,
        aliases=tuple(aliases),
    )


def _number_task(
    task_id: str,
    split: str,
    category: str,
    question: str,
    expected_number: float,
    tolerance: float = 0.01,
) -> Task:
    return Task(
        task_id=task_id,
        split=split,
        category=category,
        question=question,
        answer_type="number",
        expected_number=expected_number,
        tolerance=tolerance,
    )


ALL_TASKS: tuple[Task, ...] = (
    _text_task(
        "dev-01",
        "dev",
        "policy",
        "In the Northstar handbook, who approves goodwill credits above $150? Reply with only the role.",
        "finance duty manager",
    ),
    _text_task(
        "dev-02",
        "dev",
        "policy",
        "Which queue owns LATAM customs paperwork? Reply with only the queue name.",
        "border-ops-latam",
    ),
    _text_task(
        "dev-03",
        "dev",
        "policy",
        "What is the lost-parcel escalation code? Reply with only the code.",
        "OP-17",
        "op17",
    ),
    _text_task(
        "dev-04",
        "dev",
        "policy",
        "What color sticker marks hazardous parcels? Reply with only the color.",
        "amber",
    ),
    _text_task(
        "dev-05",
        "dev",
        "policy",
        "At what local time does the weekend warehouse cutover start? Reply in HH:MM.",
        "22:30",
    ),
    _number_task(
        "dev-06",
        "dev",
        "lookup",
        "How many days are EU debug logs kept? Reply with only the number.",
        14,
    ),
    _number_task(
        "dev-07",
        "dev",
        "lookup",
        "How many days are cold-chain incident photos retained? Reply with only the number.",
        45,
    ),
    _number_task(
        "dev-08",
        "dev",
        "lookup",
        "How many packs trigger cooler replenishment? Reply with only the number.",
        6,
    ),
    _text_task(
        "dev-09",
        "dev",
        "policy",
        "Which role approves invoice variances above 2 percent? Reply with only the role.",
        "revenue controller",
    ),
    _number_task(
        "dev-10",
        "dev",
        "lookup",
        "What is the APAC standard delivery window in days? Reply with only the number.",
        4,
    ),
    _number_task(
        "dev-11",
        "dev",
        "lookup",
        "What is the Zone B same-day surcharge in dollars? Reply with only the number.",
        7.50,
    ),
    _number_task(
        "dev-12",
        "dev",
        "lookup",
        "What is the standard return handling fee in dollars? Reply with only the number.",
        5.40,
    ),
    _number_task(
        "dev-13",
        "dev",
        "math",
        "Using the Northstar fee deck, what is the total handling fee for 7 standard returns and 2 oversize returns? Reply with only the number rounded to 2 decimals.",
        57.40,
    ),
    _number_task(
        "dev-14",
        "dev",
        "math",
        "Using the handbook prices, what is the total surcharge for 5 Zone C frozen shipments and 3 dry-ice packs? Reply with only the number rounded to 2 decimals.",
        66.50,
    ),
    _number_task(
        "dev-15",
        "dev",
        "math",
        "Using the handbook prices, what is the total for 8 priority picks plus insurance on $600 declared value? Reply with only the number rounded to 2 decimals.",
        21.90,
    ),
    _number_task(
        "dev-16",
        "dev",
        "math",
        "Using the handbook prices, what is the total for 11 Zone B same-day surcharges and 4 lite return fees? Reply with only the number rounded to 2 decimals.",
        95.30,
    ),
    _number_task(
        "dev-17",
        "dev",
        "math",
        "Using the handbook prices, what is the total for 2 oversize returns, 6 standard returns, and 9 dry-ice packs? Reply with only the number rounded to 2 decimals.",
        67.75,
    ),
    _number_task(
        "dev-18",
        "dev",
        "math",
        "Using the handbook prices, what is the total for 3 priority picks, 2 Zone C frozen surcharges, and insurance on $900 declared value? Reply with only the number rounded to 2 decimals.",
        38.45,
    ),
    _text_task(
        "test-01",
        "test",
        "policy",
        "Who approves goodwill credits between $51 and $150? Reply with only the role.",
        "finance desk",
    ),
    _text_task(
        "test-02",
        "test",
        "policy",
        "Who approves goodwill credits up to $50? Reply with only the role.",
        "floor lead",
    ),
    _text_task(
        "test-03",
        "test",
        "policy",
        "Which channel handles break-glass overrides? Reply with only the channel name.",
        "sev-override",
    ),
    _number_task(
        "test-04",
        "test",
        "lookup",
        "How many days before vendor access badges must be renewed? Reply with only the number.",
        90,
    ),
    _number_task(
        "test-05",
        "test",
        "lookup",
        "What is the dock reslot grace period in minutes? Reply with only the number.",
        20,
    ),
    _number_task(
        "test-06",
        "test",
        "lookup",
        "How many pallet recounts are allowed before escalation? Reply with only the number.",
        3,
    ),
    _number_task(
        "test-07",
        "test",
        "lookup",
        "What is the premium chat first-response SLA in minutes? Reply with only the number.",
        2,
    ),
    _number_task(
        "test-08",
        "test",
        "lookup",
        "What is the standard email first-response SLA in minutes? Reply with only the number.",
        45,
    ),
    _number_task(
        "test-09",
        "test",
        "lookup",
        "What is the lite return handling fee in dollars? Reply with only the number.",
        3.20,
    ),
    _number_task(
        "test-10",
        "test",
        "lookup",
        "What is the oversize return handling fee in dollars? Reply with only the number.",
        9.80,
    ),
    _number_task(
        "test-11",
        "test",
        "math",
        "Using the handbook prices, what is the total for 12 lite returns and 4 Zone B same-day surcharges? Reply with only the number rounded to 2 decimals.",
        68.40,
    ),
    _number_task(
        "test-12",
        "test",
        "math",
        "Using the handbook prices, what is the total for 4 standard returns, 4 dry-ice packs, and insurance on $500 declared value? Reply with only the number rounded to 2 decimals.",
        32.85,
    ),
    _number_task(
        "test-13",
        "test",
        "math",
        "Using the handbook prices, what is the total for 7 priority picks and 2 Zone C frozen surcharges? Reply with only the number rounded to 2 decimals.",
        39.20,
    ),
    _number_task(
        "test-14",
        "test",
        "math",
        "Using the handbook prices, what is the total for 5 oversize returns and insurance on $1200 declared value? Reply with only the number rounded to 2 decimals.",
        59.20,
    ),
    _number_task(
        "test-15",
        "test",
        "math",
        "Using the handbook prices, what is the total for 9 standard returns, 1 oversize return, and 2 Zone B same-day surcharges? Reply with only the number rounded to 2 decimals.",
        73.40,
    ),
    _number_task(
        "test-16",
        "test",
        "math",
        "Using the handbook prices, what is the total for 6 dry-ice packs, 3 priority picks, and insurance on $800 declared value? Reply with only the number rounded to 2 decimals.",
        23.60,
    ),
    _number_task(
        "test-17",
        "test",
        "math",
        "Using the handbook prices, what is the total for 3 Zone C frozen surcharges, 3 lite returns, and 3 standard returns? Reply with only the number rounded to 2 decimals.",
        62.55,
    ),
    _number_task(
        "test-18",
        "test",
        "math",
        "Using the handbook prices, what is the total for 2 oversize returns, 2 standard returns, and insurance on $300 declared value? Reply with only the number rounded to 2 decimals.",
        32.95,
    ),
)


DEV_TASKS: tuple[Task, ...] = tuple(task for task in ALL_TASKS if task.split == "dev")
HOLDOUT_TASKS: tuple[Task, ...] = tuple(task for task in ALL_TASKS if task.split == "test")


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("**", " ")
    text = text.replace("$", "")
    text = re.sub(r"answer\s*:\s*", "", text)
    text = re.sub(r"[^a-z0-9:.-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_first_number(text: str) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not match:
        return None
    return float(match.group(0))


def score_prediction(task: Task, prediction: str) -> bool:
    prediction = prediction.strip()
    if task.answer_type == "number":
        value = extract_first_number(prediction)
        if value is None or task.expected_number is None:
            return False
        return abs(value - task.expected_number) <= task.tolerance

    normalized_prediction = normalize_text(prediction)
    candidates = (task.expected_text, *task.aliases)
    for candidate in candidates:
        if not candidate:
            continue
        normalized_candidate = normalize_text(candidate)
        if normalized_candidate == normalized_prediction:
            return True
        if normalized_candidate in normalized_prediction:
            return True
    return False


def split_tasks(split: str) -> tuple[Task, ...]:
    if split == "dev":
        return DEV_TASKS
    if split == "test":
        return HOLDOUT_TASKS
    raise ValueError(f"Unknown split: {split}")


def iter_doc_texts(use_distilled: bool) -> Iterable[str]:
    for doc in KNOWLEDGE_BASE:
        yield doc.distilled if use_distilled else doc.body


def task_to_eval_case(task: Task) -> EvalCase:
    expected: str | float | None
    if task.answer_type == "number":
        expected = task.expected_number
    else:
        expected = task.expected_text
    metadata = {
        "category": task.category,
        "answer_type": task.answer_type,
        "tolerance": task.tolerance,
        "aliases": list(task.aliases),
    }
    return EvalCase(
        id=task.task_id,
        split="dev" if task.split == "dev" else "holdout",
        input=task.question,
        expected=expected,
        metadata=metadata,
    )


def all_eval_cases() -> tuple[EvalCase, ...]:
    return tuple(task_to_eval_case(task) for task in ALL_TASKS)
