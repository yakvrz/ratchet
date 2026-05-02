from __future__ import annotations

from copy import deepcopy
from typing import Any

from order_desk_env import TASKS, _order, _task


SPLITS: dict[str, list[int]] = {"train": [], "dev": [], "holdout": []}


_USERS = {
    "ada": {
        "user_id": "u_ada",
        "full_name": "Ada Lovelace",
        "first": "Ada",
        "last": "Lovelace",
        "email": "ada@example.com",
        "zip": "10001",
    },
    "grace": {
        "user_id": "u_grace",
        "full_name": "Grace Hopper",
        "first": "Grace",
        "last": "Hopper",
        "email": "grace@example.com",
        "zip": "94105",
    },
    "katherine": {
        "user_id": "u_katherine",
        "full_name": "Katherine Johnson",
        "first": "Katherine",
        "last": "Johnson",
        "email": "katherine@example.com",
        "zip": "30301",
    },
}

_ADDRESSES = [
    {"line1": "42 Bay St", "city": "San Francisco", "state": "CA", "zip": "94105"},
    {"line1": "9 Loop Rd", "city": "Boston", "state": "MA", "zip": "02110"},
    {"line1": "77 Lake Ave", "city": "Chicago", "state": "IL", "zip": "60601"},
    {"line1": "15 Pine St", "city": "Seattle", "state": "WA", "zip": "98101"},
    {"line1": "600 Market St", "city": "Philadelphia", "state": "PA", "zip": "19106"},
    {"line1": "88 Congress Ave", "city": "Austin", "state": "TX", "zip": "78701"},
]

_CANCEL_PRODUCTS = ["headphones", "tablet", "printer", "desk lamp", "keyboard", "webcam", "backpack", "speaker"]
_RETURN_PRODUCTS = ["blue jacket", "coffee grinder", "running shoes", "winter coat", "red scarf", "camera strap", "travel mug", "linen shirt"]
_AMBIGUOUS_PAIRS = [
    ("mug", "lamp"),
    ("red shirt", "blue shirt"),
    ("small charger", "large charger"),
    ("notebook", "planner"),
    ("black shoes", "brown shoes"),
    ("glass vase", "ceramic vase"),
]
_REASONS = ["ordered by mistake", "no longer needed"]
_REFUNDS = ["original_payment", "gift_card"]


def install_expanded_tasks() -> None:
    for split_ids in SPLITS.values():
        split_ids.clear()
    for task_id, task, split in _build_tasks():
        TASKS[task_id] = task
        SPLITS[split].append(task_id)


def _build_tasks() -> list[tuple[int, dict[str, Any], str]]:
    rows: list[tuple[int, dict[str, Any], str]] = []
    plan = {
        "train": (10_000, 6),
        "dev": (11_000, 12),
        "holdout": (12_000, 12),
    }
    for split, (base_id, per_category) in plan.items():
        for index in range(per_category):
            rows.append((*_cancel_task(base_id + index, split, index), split))
            rows.append((*_address_task(base_id + 100 + index, split, index), split))
            rows.append((*_return_task(base_id + 200 + index, split, index), split))
            rows.append((*_ambiguity_task(base_id + 300 + index, split, index), split))
    return rows


def _cancel_task(task_id: int, split: str, index: int) -> tuple[int, dict[str, Any]]:
    user = _user(index)
    product = _CANCEL_PRODUCTS[index % len(_CANCEL_PRODUCTS)]
    other = _RETURN_PRODUCTS[(index + 2) % len(_RETURN_PRODUCTS)]
    order_id = f"#{split[0].upper()}C{index:03d}"
    other_id = f"#{split[0].upper()}C{index:03d}B"
    reason = _REASONS[index % len(_REASONS)]
    auth = _auth_phrase(user, index)
    request = (
        f"{auth} Please cancel pending order {order_id} for the {product}; "
        f"the reason is {reason}. Yes, go ahead with the cancellation."
    )
    return _task(
        task_id,
        category="cancel",
        request=request,
        confirmed=True,
        orders=[
            _order(order_id, user_id=user["user_id"], status="pending", product=product, item_id=f"it_{split}_c_{index}", created_at=f"2026-04-{(index % 20) + 1:02d}"),
            _order(other_id, user_id=user["user_id"], status="delivered", product=other, item_id=f"it_{split}_c_{index}_b", created_at=f"2026-03-{(index % 20) + 1:02d}"),
        ],
        expected={"action": "cancel_order", "order_id": order_id, "reason": reason},
    )


def _address_task(task_id: int, split: str, index: int) -> tuple[int, dict[str, Any]]:
    user = _user(index + 1)
    product = _CANCEL_PRODUCTS[(index + 3) % len(_CANCEL_PRODUCTS)]
    address = deepcopy(_ADDRESSES[index % len(_ADDRESSES)])
    order_id = f"#{split[0].upper()}A{index:03d}"
    request = (
        f"{_auth_phrase(user, index + 1)} Update pending order {order_id} for the {product} "
        f"to {address['line1']}, {address['city']}, {address['state']} {address['zip']}. Yes, make that shipping change."
    )
    return _task(
        task_id,
        category="address",
        request=request,
        confirmed=True,
        orders=[
            _order(order_id, user_id=user["user_id"], status="pending", product=product, item_id=f"it_{split}_a_{index}", created_at=f"2026-04-{(index % 20) + 2:02d}"),
        ],
        expected={"action": "modify_address", "order_id": order_id, "address": address},
    )


def _return_task(task_id: int, split: str, index: int) -> tuple[int, dict[str, Any]]:
    user = _user(index + 2)
    product = _RETURN_PRODUCTS[index % len(_RETURN_PRODUCTS)]
    order_id = f"#{split[0].upper()}R{index:03d}"
    item_id = f"it_{split}_r_{index}"
    refund = _REFUNDS[index % len(_REFUNDS)]
    refund_text = "my original payment method" if refund == "original_payment" else "a gift card"
    request = (
        f"{_auth_phrase(user, index + 2)} I need to return the {product} from delivered order {order_id} "
        f"and refund it to {refund_text}. Yes, process the return."
    )
    return _task(
        task_id,
        category="return",
        request=request,
        confirmed=True,
        orders=[
            _order(order_id, user_id=user["user_id"], status="delivered", product=product, item_id=item_id, created_at=f"2026-03-{(index % 20) + 1:02d}"),
        ],
        expected={"action": "return_item", "order_id": order_id, "item_id": item_id, "refund_method": refund},
    )


def _ambiguity_task(task_id: int, split: str, index: int) -> tuple[int, dict[str, Any]]:
    user = _user(index)
    first, second = _AMBIGUOUS_PAIRS[index % len(_AMBIGUOUS_PAIRS)]
    left_id = f"#{split[0].upper()}Q{index:03d}A"
    right_id = f"#{split[0].upper()}Q{index:03d}B"
    if index % 2 == 0:
        request = (
            f"{_auth_phrase(user, index)} Cancel my pending order, but I have two pending orders "
            f"and cannot tell whether it was the {first} or the {second}."
        )
        status = "pending"
    else:
        request = (
            f"{_auth_phrase(user, index)} Return the {first.split()[-1]} from my delivered orders, "
            f"but I am not sure which order contains it."
        )
        status = "delivered"
    return _task(
        task_id,
        category="ambiguity",
        request=request,
        confirmed=False,
        orders=[
            _order(left_id, user_id=user["user_id"], status=status, product=first, item_id=f"it_{split}_q_{index}_a", created_at=f"2026-03-{(index % 20) + 1:02d}"),
            _order(right_id, user_id=user["user_id"], status=status, product=second, item_id=f"it_{split}_q_{index}_b", created_at=f"2026-03-{(index % 20) + 2:02d}"),
        ],
        expected={"action": "clarify"},
    )


def _user(index: int) -> dict[str, str]:
    return list(_USERS.values())[index % len(_USERS)]


def _auth_phrase(user: dict[str, str], index: int) -> str:
    if index % 3 == 0:
        return f"I am {user['full_name']} at {user['email']}."
    if index % 3 == 1:
        return f"{user['first']} {user['last']}, zip {user['zip']}."
    return f"This is {user['first']}; my email is {user['email']}."
