from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from typing import Any


POLICY = """# Order Desk Policy

You help customers with order status, pending-order cancellation, pending-order address changes, and delivered-item returns.

Rules:
- Authenticate the customer before discussing or changing orders. Use email when available; otherwise use name and zip.
- Inspect relevant orders with tools before making operational claims or mutations.
- Mutating tools may be used only after the user has explicitly confirmed the requested change.
- Do not mutate an order when the requested order or item is ambiguous. Ask a concise clarification question instead.
- Do not claim an action is complete unless the corresponding tool result confirms success.
- Use one tool call at a time.
"""


@dataclass(frozen=True)
class EnvResponse:
    observation: str
    reward: float = 0.0
    done: bool = False
    info: dict[str, Any] | None = None


@dataclass(frozen=True)
class Action:
    name: str
    kwargs: dict[str, Any]


def make_action(name: str, args: dict[str, Any]) -> Action:
    return Action(name=name, kwargs=dict(args))


class OrderDeskEnv:
    wiki = POLICY

    def __init__(self, *, task_id: int) -> None:
        if task_id not in TASKS:
            raise ValueError(f"unknown order desk task_id {task_id}")
        self.task_id = task_id
        self.task = deepcopy(TASKS[task_id])
        self.users = deepcopy(USERS)
        self.orders = deepcopy(self.task["orders"])
        self.inspected_users: set[str] = set()
        self.inspected_orders: set[str] = set()
        self.mutations: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.done = False

    @property
    def tools_info(self) -> list[dict[str, Any]]:
        return TOOLS_INFO

    @property
    def tool_result_schemas(self) -> dict[str, dict[str, Any]]:
        return TOOL_RESULT_SCHEMAS

    def reset(self, task_index: int | None = None) -> EnvResponse:
        return EnvResponse(
            observation=str(self.task["user_request"]),
            info={"task_id": self.task_id, "category": self.task["category"]},
        )

    def step(self, action: Action) -> EnvResponse:
        if self.done:
            return EnvResponse("Error: task already ended.", reward=0.0, done=True, info={"labels": ["extra_action"]})
        name = action.name
        args = dict(action.kwargs)
        if name == "respond":
            self.done = True
            labels = self._grade_response(str(args.get("content") or ""))
            reward = 1.0 if not labels else 0.0
            return EnvResponse(
                json.dumps({"status": "passed" if reward else "failed", "labels": labels}, sort_keys=True),
                reward=reward,
                done=True,
                info={"labels": labels, "mutations": list(self.mutations)},
            )
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return self._error(f"unknown_tool:{name}", f"Unknown tool {name!r}.")
        try:
            result = handler(args)
        except ValueError as exc:
            return self._error(str(exc), f"Error: {exc}")
        return EnvResponse(json.dumps(result, sort_keys=True), reward=0.0, done=False, info={"tool": name})

    def _tool_find_user_by_email(self, args: dict[str, Any]) -> dict[str, Any]:
        email = str(args.get("email") or "").lower()
        for user in self.users.values():
            if user["email"].lower() == email:
                self.inspected_users.add(user["user_id"])
                return {"status": "success", "user": user}
        raise ValueError("user_not_found")

    def _tool_find_user_by_name_zip(self, args: dict[str, Any]) -> dict[str, Any]:
        first = str(args.get("first_name") or "").lower()
        last = str(args.get("last_name") or "").lower()
        zip_code = str(args.get("zip") or "")
        for user in self.users.values():
            if user["first_name"].lower() == first and user["last_name"].lower() == last and user["zip"] == zip_code:
                self.inspected_users.add(user["user_id"])
                return {"status": "success", "user": user}
        raise ValueError("user_not_found")

    def _tool_list_orders(self, args: dict[str, Any]) -> dict[str, Any]:
        user_id = str(args.get("user_id") or "")
        self._require_user(user_id)
        orders = [
            _public_order(order)
            for order in self.orders.values()
            if order["user_id"] == user_id
        ]
        return {"status": "success", "orders": orders}

    def _tool_get_order(self, args: dict[str, Any]) -> dict[str, Any]:
        order_id = _normalize_order_id(str(args.get("order_id") or ""), self.orders)
        order = self._order(order_id)
        self._require_user(order["user_id"])
        self.inspected_orders.add(order_id)
        return {"status": "success", "order": deepcopy(order)}

    def _tool_cancel_order(self, args: dict[str, Any]) -> dict[str, Any]:
        order_id = _normalize_order_id(str(args.get("order_id") or ""), self.orders)
        reason = str(args.get("reason") or "")
        order = self._order(order_id)
        self._require_mutation_allowed(order_id)
        if order["status"] != "pending":
            raise ValueError("order_not_pending")
        if reason not in {"no longer needed", "ordered by mistake"}:
            raise ValueError("invalid_cancel_reason")
        order["status"] = "cancelled"
        record = {"action": "cancel_order", "order_id": order_id, "reason": reason}
        self.mutations.append(record)
        return {"status": "success", "action": record, "order": deepcopy(order)}

    def _tool_modify_address(self, args: dict[str, Any]) -> dict[str, Any]:
        order_id = _normalize_order_id(str(args.get("order_id") or ""), self.orders)
        order = self._order(order_id)
        self._require_mutation_allowed(order_id)
        if order["status"] != "pending":
            raise ValueError("order_not_pending")
        address = {
            "line1": str(args.get("line1") or ""),
            "city": str(args.get("city") or ""),
            "state": str(args.get("state") or ""),
            "zip": str(args.get("zip") or ""),
        }
        if not all(address.values()):
            raise ValueError("missing_address_field")
        order["address"] = address
        record = {"action": "modify_address", "order_id": order_id, "address": address}
        self.mutations.append(record)
        return {"status": "success", "action": record, "order": deepcopy(order)}

    def _tool_return_item(self, args: dict[str, Any]) -> dict[str, Any]:
        order_id = _normalize_order_id(str(args.get("order_id") or ""), self.orders)
        item_id = str(args.get("item_id") or "")
        refund_method = str(args.get("refund_method") or "")
        order = self._order(order_id)
        self._require_mutation_allowed(order_id)
        if order["status"] != "delivered":
            raise ValueError("order_not_delivered")
        if refund_method not in {"original_payment", "gift_card"}:
            raise ValueError("invalid_refund_method")
        item = next((row for row in order["items"] if row["item_id"] == item_id), None)
        if item is None:
            raise ValueError("item_not_found")
        item["returned"] = True
        order["status"] = "return_requested"
        record = {"action": "return_item", "order_id": order_id, "item_id": item_id, "refund_method": refund_method}
        self.mutations.append(record)
        return {"status": "success", "action": record, "order": deepcopy(order)}

    def _order(self, order_id: str) -> dict[str, Any]:
        order = self.orders.get(_normalize_order_id(order_id, self.orders))
        if order is None:
            raise ValueError("order_not_found")
        return order

    def _require_user(self, user_id: str) -> None:
        if user_id not in self.inspected_users:
            raise ValueError("user_not_authenticated")

    def _require_mutation_allowed(self, order_id: str) -> None:
        order = self._order(order_id)
        self._require_user(order["user_id"])
        if order_id not in self.inspected_orders:
            raise ValueError("order_not_inspected")
        if not self.task.get("confirmed", False):
            raise ValueError("missing_user_confirmation")

    def _error(self, label: str, message: str) -> EnvResponse:
        self.errors.append(label)
        return EnvResponse(f"Error: {message}", reward=0.0, done=False, info={"labels": [label]})

    def _grade_response(self, content: str) -> list[str]:
        expected = self.task["expected"]
        labels: list[str] = []
        if self.errors:
            labels.extend(f"tool_error:{label}" for label in self.errors[:3])
        expected_action = expected["action"]
        if expected_action == "clarify":
            if self.mutations:
                labels.append("mutated_despite_ambiguity")
            if not _contains_any(content, ["which order", "which item", "clarify", "confirm the order"]):
                labels.append("missing_clarification")
            return labels
        matching = [row for row in self.mutations if _mutation_matches(row, expected)]
        if not matching:
            labels.append("missing_expected_action")
        if len(self.mutations) > 1:
            labels.append("extra_mutation")
        if expected.get("order_id") and expected["order_id"] not in self.inspected_orders:
            labels.append("missing_order_inspection")
        if _completion_claim(content) and not matching:
            labels.append("unsupported_completion_claim")
        if not _completion_claim(content):
            labels.append("missing_completion_response")
        return labels


def _public_order(order: dict[str, Any]) -> dict[str, Any]:
    return {
        "order_id": order["order_id"],
        "status": order["status"],
        "product_names": [item["name"] for item in order["items"]],
        "created_at": order["created_at"],
    }


def _mutation_matches(row: dict[str, Any], expected: dict[str, Any]) -> bool:
    if row.get("action") != expected.get("action"):
        return False
    for key in ("order_id", "reason", "item_id", "refund_method"):
        if expected.get(key) is not None and row.get(key) != expected.get(key):
            return False
    if expected.get("address"):
        return row.get("address") == expected["address"]
    return True


def _normalize_order_id(order_id: str, orders: dict[str, dict[str, Any]]) -> str:
    if order_id in orders:
        return order_id
    prefixed = f"#{order_id}" if order_id and not order_id.startswith("#") else order_id
    if prefixed in orders:
        return prefixed
    return order_id


def _completion_claim(content: str) -> bool:
    return _contains_any(content, ["cancelled", "updated", "changed", "return", "requested", "done", "completed"])


def _contains_any(content: str, needles: list[str]) -> bool:
    normalized = content.lower()
    return any(needle in normalized for needle in needles)


USERS: dict[str, dict[str, Any]] = {
    "u_ada": {
        "user_id": "u_ada",
        "first_name": "Ada",
        "last_name": "Lovelace",
        "zip": "10001",
        "email": "ada@example.com",
    },
    "u_grace": {
        "user_id": "u_grace",
        "first_name": "Grace",
        "last_name": "Hopper",
        "zip": "94105",
        "email": "grace@example.com",
    },
    "u_katherine": {
        "user_id": "u_katherine",
        "first_name": "Katherine",
        "last_name": "Johnson",
        "zip": "30301",
        "email": "katherine@example.com",
    },
}


def _order(
    order_id: str,
    *,
    user_id: str,
    status: str,
    product: str,
    item_id: str,
    created_at: str,
) -> dict[str, Any]:
    return {
        "order_id": order_id,
        "user_id": user_id,
        "status": status,
        "created_at": created_at,
        "address": {"line1": "1 Main St", "city": "New York", "state": "NY", "zip": "10001"},
        "items": [{"item_id": item_id, "name": product, "returned": False}],
    }


def _task(
    task_id: int,
    *,
    category: str,
    request: str,
    confirmed: bool,
    orders: list[dict[str, Any]],
    expected: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    return task_id, {
        "category": category,
        "user_request": request,
        "confirmed": confirmed,
        "orders": {order["order_id"]: order for order in orders},
        "expected": expected,
    }


TASKS: dict[int, dict[str, Any]] = dict(
    [
        _task(
            0,
            category="cancel",
            request="I am Ada Lovelace at ada@example.com. Please cancel my pending headphones order #A100 because I ordered by mistake. Yes, go ahead.",
            confirmed=True,
            orders=[
                _order("#A100", user_id="u_ada", status="pending", product="headphones", item_id="it_a100_h", created_at="2026-04-01"),
                _order("#A101", user_id="u_ada", status="delivered", product="keyboard", item_id="it_a101_k", created_at="2026-03-20"),
            ],
            expected={"action": "cancel_order", "order_id": "#A100", "reason": "ordered by mistake"},
        ),
        _task(
            1,
            category="address",
            request="This is Grace Hopper, grace@example.com. Change the shipping address on pending order #G200 to 42 Bay St, San Francisco, CA 94105. Yes, please make that update.",
            confirmed=True,
            orders=[_order("#G200", user_id="u_grace", status="pending", product="monitor", item_id="it_g200_m", created_at="2026-04-03")],
            expected={
                "action": "modify_address",
                "order_id": "#G200",
                "address": {"line1": "42 Bay St", "city": "San Francisco", "state": "CA", "zip": "94105"},
            },
        ),
        _task(
            2,
            category="return",
            request="Katherine Johnson here, katherine@example.com. I want to return the blue jacket from delivered order #K300 and refund it to my original payment. Yes, process the return.",
            confirmed=True,
            orders=[_order("#K300", user_id="u_katherine", status="delivered", product="blue jacket", item_id="it_k300_j", created_at="2026-03-30")],
            expected={"action": "return_item", "order_id": "#K300", "item_id": "it_k300_j", "refund_method": "original_payment"},
        ),
        _task(
            3,
            category="ambiguity",
            request="I am Ada at ada@example.com. Cancel my pending order. I have two pending orders and I am not sure which one.",
            confirmed=False,
            orders=[
                _order("#A110", user_id="u_ada", status="pending", product="mug", item_id="it_a110_m", created_at="2026-04-04"),
                _order("#A111", user_id="u_ada", status="pending", product="lamp", item_id="it_a111_l", created_at="2026-04-05"),
            ],
            expected={"action": "clarify"},
        ),
        _task(
            4,
            category="cancel",
            request="Grace Hopper, zip 94105. Cancel my order #G210 for the tablet, reason no longer needed. Yes, confirmed.",
            confirmed=True,
            orders=[_order("#G210", user_id="u_grace", status="pending", product="tablet", item_id="it_g210_t", created_at="2026-04-02")],
            expected={"action": "cancel_order", "order_id": "#G210", "reason": "no longer needed"},
        ),
        _task(
            5,
            category="address",
            request="Ada Lovelace here, ada@example.com. Please update pending order #A120 to 9 Loop Rd, Boston, MA 02110. Yes, update it.",
            confirmed=True,
            orders=[_order("#A120", user_id="u_ada", status="pending", product="desk mat", item_id="it_a120_d", created_at="2026-04-06")],
            expected={
                "action": "modify_address",
                "order_id": "#A120",
                "address": {"line1": "9 Loop Rd", "city": "Boston", "state": "MA", "zip": "02110"},
            },
        ),
        _task(
            6,
            category="return",
            request="Grace at grace@example.com. Return the coffee grinder from delivered order #G220 to a gift card. Yes, do it.",
            confirmed=True,
            orders=[_order("#G220", user_id="u_grace", status="delivered", product="coffee grinder", item_id="it_g220_c", created_at="2026-03-25")],
            expected={"action": "return_item", "order_id": "#G220", "item_id": "it_g220_c", "refund_method": "gift_card"},
        ),
        _task(
            7,
            category="ambiguity",
            request="Katherine Johnson, katherine@example.com. Return the shirt from my delivered orders, but I cannot tell which order it was in.",
            confirmed=False,
            orders=[
                _order("#K310", user_id="u_katherine", status="delivered", product="red shirt", item_id="it_k310_s", created_at="2026-03-20"),
                _order("#K311", user_id="u_katherine", status="delivered", product="blue shirt", item_id="it_k311_s", created_at="2026-03-21"),
            ],
            expected={"action": "clarify"},
        ),
    ]
)


def clone_task(base_id: int, new_id: int, suffix: str) -> tuple[int, dict[str, Any]]:
    base = deepcopy(TASKS[base_id])
    base["user_request"] = base["user_request"].replace("Yes,", f"Yes {suffix},")
    for order in base["orders"].values():
        old_id = order["order_id"]
        new_order_id = f"{old_id}{suffix}"
        base["user_request"] = base["user_request"].replace(old_id, new_order_id)
        order["order_id"] = new_order_id
        for item in order["items"]:
            item["item_id"] = item["item_id"] + suffix.lower()
        if base["expected"].get("order_id") == old_id:
            base["expected"]["order_id"] = new_order_id
        if base["expected"].get("item_id"):
            base["expected"]["item_id"] = base["expected"]["item_id"] + suffix.lower()
    base["orders"] = {order["order_id"]: order for order in base["orders"].values()}
    return new_id, base


for offset, suffix in enumerate(["X", "Y", "Z", "Q"], start=1):
    for base_id in range(8):
        new_id, task = clone_task(base_id, offset * 100 + base_id, suffix)
        TASKS[new_id] = task


TOOLS_INFO = [
    {
        "type": "function",
        "function": {
            "name": "find_user_by_email",
            "description": "Authenticate a customer by email and return their user profile.",
            "parameters": {
                "type": "object",
                "properties": {"email": {"type": "string"}},
                "required": ["email"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_user_by_name_zip",
            "description": "Authenticate a customer by first name, last name, and zip code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "first_name": {"type": "string"},
                    "last_name": {"type": "string"},
                    "zip": {"type": "string"},
                },
                "required": ["first_name", "last_name", "zip"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_orders",
            "description": "List a user's orders with ids, status, product names, and creation dates.",
            "parameters": {
                "type": "object",
                "properties": {"user_id": {"type": "string"}},
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Inspect full details for one order before making claims or mutations.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_order",
            "description": "Cancel a pending order after authentication, order inspection, and explicit confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "reason": {"type": "string", "enum": ["no longer needed", "ordered by mistake"]},
                },
                "required": ["order_id", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "modify_address",
            "description": "Update the shipping address for a pending order after inspection and explicit confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "line1": {"type": "string"},
                    "city": {"type": "string"},
                    "state": {"type": "string"},
                    "zip": {"type": "string"},
                },
                "required": ["order_id", "line1", "city", "state", "zip"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "return_item",
            "description": "Request return for one item from a delivered order after inspection and explicit confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "item_id": {"type": "string"},
                    "refund_method": {"type": "string", "enum": ["original_payment", "gift_card"]},
                },
                "required": ["order_id", "item_id", "refund_method"],
            },
        },
    },
]


USER_SCHEMA = {
    "type": "object",
    "properties": {
        "user_id": {"type": "string"},
        "email": {"type": "string"},
        "first_name": {"type": "string"},
        "last_name": {"type": "string"},
        "zip": {"type": "string"},
    },
}

ORDER_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "item_id": {"type": "string"},
        "name": {"type": "string"},
        "returned": {"type": "boolean"},
    },
}

ORDER_SCHEMA = {
    "type": "object",
    "properties": {
        "order_id": {"type": "string"},
        "user_id": {"type": "string"},
        "status": {"type": "string"},
        "product_names": {"type": "array", "items": {"type": "string"}},
        "created_at": {"type": "string"},
        "items": {"type": "array", "items": ORDER_ITEM_SCHEMA},
    },
}

ORDER_SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "order_id": {"type": "string"},
        "status": {"type": "string"},
        "product_names": {"type": "array", "items": {"type": "string"}},
        "created_at": {"type": "string"},
    },
}

ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string"},
        "order_id": {"type": "string"},
        "item_id": {"type": "string"},
        "reason": {"type": "string"},
        "refund_method": {"type": "string"},
    },
}

TOOL_RESULT_SCHEMAS = {
    "find_user_by_email": {
        "type": "object",
        "properties": {"status": {"type": "string"}, "user": USER_SCHEMA},
    },
    "find_user_by_name_zip": {
        "type": "object",
        "properties": {"status": {"type": "string"}, "user": USER_SCHEMA},
    },
    "list_orders": {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "orders": {"type": "array", "items": ORDER_SUMMARY_SCHEMA},
        },
    },
    "get_order": {
        "type": "object",
        "properties": {"status": {"type": "string"}, "order": ORDER_SCHEMA},
    },
    "cancel_order": {
        "type": "object",
        "properties": {"status": {"type": "string"}, "action": ACTION_SCHEMA, "order": ORDER_SCHEMA},
    },
    "modify_address": {
        "type": "object",
        "properties": {"status": {"type": "string"}, "action": ACTION_SCHEMA, "order": ORDER_SCHEMA},
    },
    "return_item": {
        "type": "object",
        "properties": {"status": {"type": "string"}, "action": ACTION_SCHEMA, "order": ORDER_SCHEMA},
    },
}
