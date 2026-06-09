# llm-cost-guardrails

**A circuit breaker for your language model spend.** Provider-agnostic middleware
that meters every LLM call and refuses the ones you can't afford: per-tenant
**cost budgets**, **rate limits** on requests and tokens, and a global **kill
switch** that trips when spend runs away. It is the concrete defense behind
OWASP's *Unbounded Consumption* risk, built with the discipline of a payments
authorization system.

> **Runs offline with zero API keys.** Every control is exact arithmetic over a
> token-usage shape that matches what providers report back, so the whole demo
> and test suite run on a deterministic mock model. Wire it to a real Anthropic
> or OpenAI client by passing a callable that returns `(result, Usage)` — the
> guard code is identical either way.

---

## Quickstart

```bash
git clone https://github.com/akanjilal-dev/llm-cost-guardrails
cd llm-cost-guardrails
pip install -r requirements.txt        # core needs nothing; this adds pytest
python -m guardrails.main              # watch every guardrail fire, offline
pytest -q                              # the budget/limit/breaker claims, as tests
```

## What you'll see

```
llm-cost-guardrails — every LLM call, metered and bounded
  globex  opus-4.8   in=2000 out=800   -> $0.03000  (allowed)
    same call on claude-sonnet-4-6    would cost $0.01800
    same call on claude-haiku-4-5     would cost $0.00600

Per-tenant budget: acme has a $0.20 cap
  call 4: $0.04500  (allowed, acme spent $0.18000)
  call 5: REFUSED — budget exceeded for tenant 'acme': only $0.0200 of the $0.20 cap remains

Rate limit: globex capped at 5 requests/min
  request 5: REFUSED — tenant 'globex' over request rate (5/min)
  after 60s the bucket refilled — next request admitted again

Kill switch: global $2.00 / minute spend breaker
  burst call 2: $1.5000  (window spend $3.2163)
  burst call 3: REFUSED — spend circuit breaker tripped: >$2.00 in 60s.
```

## The four controls

| Control | Module | What it stops |
|---|---|---|
| **Cost estimation** | [`pricing.py`](guardrails/pricing.py) | The foundation: turns token usage into dollars, including the cache-read discount and cache-write premium. Every other control asks it "what will this cost?" first. |
| **Per-tenant budgets** | [`budgets.py`](guardrails/budgets.py) | A tenant spending past its cap. Uses *reserve-then-reconcile* accounting (below). |
| **Rate limits** | [`limits.py`](guardrails/limits.py) | A single client flooding you with requests or tokens. Token-bucket, per tenant, for both requests-per-minute and tokens-per-minute. |
| **Kill switch** | [`limits.py`](guardrails/limits.py) | A runaway agent loop that slips past per-tenant budgets. A rolling-window spend breaker that fails closed until a human resets it. |

[`guard.py`](guardrails/guard.py) wires them into one `CostGuard.guarded_call(...)`:
kill switch → rate limit → budget reserve → run the call → reconcile to the real
cost → record and attribute. **Every gate fails closed** — if a control says no,
the underlying model is never called, so you never pay for a request you weren't
allowed to make.

## Reserve, then reconcile (the payments-grade part)

You cannot know an LLM call's cost in advance, because you do not know the output
length until the response returns. So the budget handles a call the way a card
network handles a swipe: it places an **authorization hold** for the worst-case
cost (full input plus the `max_output_tokens` ceiling) *before* the call, then
**settles** to the actual cost afterwards.

```python
hold   = worst_case_cost(model, input_tokens, max_output_tokens)  # the auth hold
budgets.reserve(tenant, hold)        # raises BudgetExceeded if it won't fit
result, usage = call()               # the real request
actual = estimate_cost(model, usage) # what it actually cost
budgets.reconcile(tenant, hold, actual)   # release the hold, record the truth
```

A tenant can never be authorized past its cap, even with several of its calls in
flight at once. A call that errors releases its hold and is never charged.

## Wiring it to a real model

The guard is provider-agnostic. Hand it a callable that performs the request and
returns `(result, Usage)` — the `Usage` shape mirrors what providers report:

```python
from guardrails.guard import default_guard
from guardrails.pricing import Usage

guard = default_guard(
    per_tenant_budget=5.00, requests_per_min=60,
    tokens_per_min=200_000, kill_switch_limit=50.00,
)

def call_claude():
    msg = client.messages.create(model="claude-opus-4-8", max_tokens=1024,
                                  messages=[{"role": "user", "content": prompt}])
    u = msg.usage
    return msg, Usage(input_tokens=u.input_tokens, output_tokens=u.output_tokens,
                      cache_read_tokens=u.cache_read_input_tokens,
                      cache_write_tokens=u.cache_creation_input_tokens)

out = guard.guarded_call(tenant="acme", model="claude-opus-4-8",
                         input_tokens=estimated_in, max_output_tokens=1024,
                         call=call_claude)
```

## Roadmap

- [x] Cost estimation with cache-read / cache-write pricing
- [x] Per-tenant budgets (reserve-then-reconcile), rate limits, spend kill switch
- [x] Per-tenant cost attribution report
- [ ] Persistent ledger backend (Redis / Postgres) for multi-process deployments
- [ ] Spend forecasting and alerting before the cap is hit (pairs with `ai-finops-agent`)
- [ ] Prompt-cache-aware budgeting that rewards cache hits
- [ ] A FastAPI / ASGI middleware wrapper

## Caveats

- **Teaching-grade, in-memory.** The ledger and limiters live in process memory;
  a real multi-process deployment needs a shared backend (see the roadmap). The
  control logic is the point, not the storage.
- **Pricing drifts.** The rate table carries published Anthropic Claude rates;
  re-verify against the provider's pricing page before you rely on a number.
- **Estimating input tokens.** Budgeting before a call needs an input-token
  estimate; use the provider's token-counting endpoint for an exact figure
  rather than a character heuristic.

---

*Part of [akanjilal.dev](https://akanjilal.dev) — frontier compute, made secure, cost-governed, and production-real.*
