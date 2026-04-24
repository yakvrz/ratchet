from __future__ import annotations

from dataclasses import dataclass
import re


TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")


@dataclass(frozen=True)
class PolicyDoc:
    doc_id: str
    title: str
    body: str
    distilled: str


POLICY_DOCS: tuple[PolicyDoc, ...] = (
    PolicyDoc(
        doc_id="travel_meals",
        title="Business Travel Meals",
        body=(
            "Itemized meals during approved business travel are reimbursable up to the actual spend "
            "when a receipt is attached. Alcohol is never reimbursable."
        ),
        distilled="Approved business-travel meals with a receipt are reimbursable. Alcohol is excluded.",
    ),
    PolicyDoc(
        doc_id="commuting",
        title="Normal Commuting",
        body="Daily commuting between home and the normal office is not reimbursable.",
        distilled="Normal commuting is not reimbursable.",
    ),
    PolicyDoc(
        doc_id="home_office",
        title="Home Office Equipment",
        body=(
            "Home-office accessories such as keyboards, mice, and monitors are reimbursable once every "
            "24 months when the total cost is at or below 300 USD."
        ),
        distilled="Home-office accessories are reimbursable up to 300 USD once every 24 months.",
    ),
    PolicyDoc(
        doc_id="lodging",
        title="Hotel and Lodging",
        body=(
            "Approved business lodging is reimbursable when the nightly rate is at or below 220 USD. "
            "Higher nightly rates require manager review."
        ),
        distilled="Hotel stays up to 220 USD/night are reimbursable. Above that requires review.",
    ),
    PolicyDoc(
        doc_id="rideshare",
        title="Ground Transportation",
        body=(
            "Rideshares to or from airports, clients, and approved off-sites are reimbursable with a receipt. "
            "Personal rides are not reimbursable."
        ),
        distilled="Work rideshares with a receipt are reimbursable.",
    ),
    PolicyDoc(
        doc_id="training",
        title="Training and Conferences",
        body=(
            "Job-related training is reimbursable when preapproved. Without preapproval, the request must be "
            "escalated for review."
        ),
        distilled="Preapproved job-related training is reimbursable. Otherwise escalate.",
    ),
    PolicyDoc(
        doc_id="gift_cards",
        title="Gift Cards",
        body="Gift cards are never reimbursable.",
        distilled="Gift cards are never reimbursable.",
    ),
    PolicyDoc(
        doc_id="saas_tools",
        title="Software and SaaS",
        body=(
            "Recurring SaaS purchases require IT approval or an approved vendor listing. Unapproved vendors "
            "must be escalated for review."
        ),
        distilled="Recurring SaaS from unapproved vendors must be escalated.",
    ),
)


def tokenize(text: str) -> set[str]:
    return {token for token in TOKEN_PATTERN.findall(text.lower()) if token}
