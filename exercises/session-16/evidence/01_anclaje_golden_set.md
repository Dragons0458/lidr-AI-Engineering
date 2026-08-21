# Session 16 — golden-set anchoring

The numbers in `evals/golden_set_s16.json` are derived from
`data/budgets_sample.json` with `HOURS_PER_DAY = 8`. They are **not** LLM
outputs and they are **not** guessed.

Rule (documented in the golden-set `description`):

```
expected_engineer_days = round(sum(analog_component.estimated_hours) / 8)
acceptable_range       = [round(0.6 × expected), round(1.5 × expected)]
```

The band is asymmetric: under-estimating hurts more than over-estimating.

Live retrieval (`POST /v1/retrieval/search`) was **not** available while this
file was written (local API down). Anchors use the budgets the brief is
written to retrieve. Re-run retrieval before the live session and amend this
table if the returned `budget_id`s differ.

| Case | Brief | Budgets | Analog components | Hours | Days | Range | Sources |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S16-01 | Portal de proveedores (S14 happy path). No SAP analog in the sample corpus. | S07-FIN-001, S07-HLT-001, S07-FIN-002 | `auth-api` 120h; `appointment-booking` 120h + `secure-messaging` 135h (closest portal); `advisor-dashboard` 115h (reports) | 490 | 61 | [37, 92] | S07-FIN-001, S07-HLT-001, S07-FIN-002 |
| S16-02 | Telemedicine: appointments, secure messaging, device ingestion, clinical alerts | S07-HLT-001, S07-HLT-002 | full HLT-001 255h + full HLT-002 290h | 545 | 68 | [41, 102] | S07-HLT-001, S07-HLT-002 |
| S16-03 | Marketplace: catalog search, checkout, recommendations, campaign landings | S07-ECO-001, S07-ECO-004 | `catalog-search` 140h + `checkout-ui` 150h + `recommendations` 135h + `campaign-builder` 110h | 535 | 67 | [40, 101] | S07-ECO-001, S07-ECO-004 |
| S16-04 | Wallet + PSD2 (from `TR-2024-001`): OAuth/SCA + double-entry ledger | S07-FIN-001 | `auth-api` 120h + `payment-ledger` 160h | 280 | 35 | [21, 53] | S07-FIN-001 |
| S16-05 | TITÁN — login + reporting at a scale the corpus never saw. Must still **estimate**. | S07-FIN-001, S07-FIN-002 | `auth-api` 120h + `advisor-dashboard` 115h (do not scale hours to 10M sessions) | 235 | 29 | [17, 44] | S07-FIN-001 |
| S16-06 | Quantum metrology, no analog | — | — | — | 0 | [0, 0] | (abstention) |
| S16-07 | Empty brief | — | — | — | 0 | [0, 0] | (abstention) |

## Why S16-01 is not a clean "direct precedent"

The happy-path transcript asks for a **supplier portal + SAP integration**.
`data/budgets_sample.json` has no ERP/SAP component. Related rows:

- S07-HLT-001 — "Patient portal" (closest *portal*)
- S07-FIN-002 — "Loan origination portal"
- S07-FIN-001 — OAuth/JWT auth
- S07-ECO-003 — `returns-portal` 90h (not used: returns ≠ supplier invoices)

The 61-day figure is therefore an analogical floor, not a SAP quote. If the
system abstains here, that is a finding (over-cautious). If it invents a
six-month SAP programme, the band catches it.

## Hours already in engineer-days (spot check)

| Hours | / 8 | round |
| --- | --- | --- |
| 490 | 61.25 | 61 |
| 545 | 68.125 | 68 |
| 535 | 66.875 | 67 |
| 280 | 35.00 | 35 |
| 235 | 29.375 | 29 |
