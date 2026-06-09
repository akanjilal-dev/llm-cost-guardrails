"""
guardrails/budgets.py
=====================
Per-tenant cost budgets with reserve-then-reconcile accounting.

The hard part of budgeting an LLM call is that you do not know the output length
until the call returns, so you cannot know the exact cost in advance. The ledger
handles this the way a payments authorisation handles a card swipe: it reserves
the worst-case amount before the call (the authorisation hold), then reconciles
to the actual amount after the call settles. A tenant can never be authorised
past its cap, even while several of its calls are in flight at once.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class BudgetExceeded(Exception):
    """Raised when a call would push a tenant past its budget cap."""

    def __init__(self, tenant: str, cap: float, committed: float, reserved: float, requested: float):
        self.tenant = tenant
        self.cap = cap
        self.committed = committed
        self.reserved = reserved
        self.requested = requested
        available = cap - committed - reserved
        super().__init__(
            f"budget exceeded for tenant {tenant!r}: requested ${requested:.4f} but only "
            f"${max(available, 0):.4f} of the ${cap:.2f} cap remains "
            f"(${committed:.4f} spent, ${reserved:.4f} reserved in flight)"
        )


@dataclass
class _Account:
    cap: float
    committed: float = 0.0  # cost of calls that have settled
    reserved: float = 0.0   # worst-case holds for calls in flight


@dataclass
class BudgetLedger:
    """Tracks committed and in-flight spend per tenant against a cap.

    `default_cap` applies to any tenant without an explicit cap. Set a tenant's
    cap with `set_cap`. All amounts are USD.
    """

    default_cap: float = 1.0
    _accounts: dict[str, _Account] = field(default_factory=dict)

    def set_cap(self, tenant: str, cap: float) -> None:
        self._account(tenant).cap = cap

    def _account(self, tenant: str) -> _Account:
        if tenant not in self._accounts:
            self._accounts[tenant] = _Account(cap=self.default_cap)
        return self._accounts[tenant]

    def available(self, tenant: str) -> float:
        a = self._account(tenant)
        return round(a.cap - a.committed - a.reserved, 6)

    def spent(self, tenant: str) -> float:
        return round(self._account(tenant).committed, 6)

    def reserve(self, tenant: str, amount: float) -> float:
        """Place a hold for the worst-case cost. Raises BudgetExceeded if it won't fit."""
        a = self._account(tenant)
        if a.committed + a.reserved + amount > a.cap + 1e-9:
            raise BudgetExceeded(tenant, a.cap, a.committed, a.reserved, amount)
        a.reserved = round(a.reserved + amount, 6)
        return amount

    def reconcile(self, tenant: str, reserved: float, actual: float) -> None:
        """Settle a held call: release the hold, record the real cost."""
        a = self._account(tenant)
        a.reserved = round(max(a.reserved - reserved, 0.0), 6)
        a.committed = round(a.committed + actual, 6)

    def release(self, tenant: str, reserved: float) -> None:
        """Release a hold without charging (the call never executed)."""
        a = self._account(tenant)
        a.reserved = round(max(a.reserved - reserved, 0.0), 6)

    def report(self) -> dict[str, dict[str, float]]:
        """Per-tenant attribution: spend, cap, and remaining budget."""
        return {
            tenant: {
                "spent": round(a.committed, 6),
                "cap": a.cap,
                "remaining": round(a.cap - a.committed - a.reserved, 6),
            }
            for tenant, a in sorted(self._accounts.items(), key=lambda kv: kv[1].committed, reverse=True)
        }
