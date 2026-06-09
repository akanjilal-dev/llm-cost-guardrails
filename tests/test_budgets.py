"""
tests/test_budgets.py
=====================
Reserve-then-reconcile must never let a tenant authorise past its cap, even with
several calls in flight, and must settle to the real cost afterwards.
"""

import pytest

from guardrails.budgets import BudgetExceeded, BudgetLedger


def test_reserve_and_reconcile_settles_actual():
    ledger = BudgetLedger(default_cap=1.0)
    hold = ledger.reserve("t1", 0.40)        # worst case
    ledger.reconcile("t1", hold, 0.10)       # actual came in lower
    assert ledger.spent("t1") == 0.10
    assert ledger.available("t1") == pytest.approx(0.90)


def test_cannot_exceed_cap():
    ledger = BudgetLedger(default_cap=0.50)
    ledger.reserve("t1", 0.40)
    ledger.reconcile("t1", 0.40, 0.40)
    with pytest.raises(BudgetExceeded):
        ledger.reserve("t1", 0.20)           # 0.40 + 0.20 > 0.50


def test_in_flight_holds_count_against_cap():
    ledger = BudgetLedger(default_cap=1.0)
    ledger.reserve("t1", 0.60)               # in flight, not yet settled
    # A concurrent call must see only 0.40 available, even though nothing settled.
    assert ledger.available("t1") == pytest.approx(0.40)
    with pytest.raises(BudgetExceeded):
        ledger.reserve("t1", 0.50)


def test_release_frees_the_hold():
    ledger = BudgetLedger(default_cap=1.0)
    hold = ledger.reserve("t1", 0.60)
    ledger.release("t1", hold)               # call failed, no charge
    assert ledger.available("t1") == pytest.approx(1.0)
    assert ledger.spent("t1") == 0.0


def test_per_tenant_caps_are_independent():
    ledger = BudgetLedger(default_cap=1.0)
    ledger.set_cap("small", 0.10)
    ledger.reserve("small", 0.10)
    ledger.reconcile("small", 0.10, 0.10)
    with pytest.raises(BudgetExceeded):
        ledger.reserve("small", 0.01)
    # A different tenant is unaffected.
    ledger.reserve("big", 0.90)


def test_report_is_sorted_by_spend():
    ledger = BudgetLedger(default_cap=10.0)
    for t, amt in [("a", 0.10), ("b", 0.50), ("c", 0.30)]:
        ledger.reconcile(t, 0.0, amt)
    assert list(ledger.report()) == ["b", "c", "a"]
