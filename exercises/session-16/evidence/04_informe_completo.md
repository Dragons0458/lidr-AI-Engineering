# Session 16 eval `14071272`

- label: `baseline-s16`
- started: 2026-08-21T21:21:52.728016+00:00
- finished: 2026-08-21T21:29:15.555600+00:00
- golden set: `evals/golden_set_s16.json` sha256=`35d3c3c503bc799e84edece414cb0a2a37808f1227c0c503d469ff3f5226f41b` (7 cases)

A p95 with n=7 is not a p95. Read it next to `n`.

This report never includes transcripts, generated text, or API keys.

## Environment

- `base_url`: `http://127.0.0.1:8000`
- `generation_model`: `None`
- `app_env`: `None`
- `models`: `{'PRIMARY_MODEL': 'gemini/gemini-2.5-flash', 'FALLBACK_MODEL': 'gpt-4o-mini', 'CRITIC_MODEL': 'gpt-4o-mini', 'COMPRESSION_MODEL': 'gemini/gemini-2.5-flash', 'PROPOSITIONAL_CHUNKER_MODEL': 'gpt-4o-mini', 'CONTEXTUAL_CHUNKER_MODEL': 'claude-sonnet-4-5', 'HALLUCINATION_JUDGE_MODEL': 'gpt-5-mini', 'AUGMENTATION_MODEL': 'gpt-5-mini'}`
- `retrieval_config`: `{'retrieval': {'RETRIEVAL_SEARCH_MODE': {'effective': 'vector', 'default': 'vector', 'overridden': False}, 'RERANKER_ENABLED': {'effective': False, 'default': False, 'overridden': False}, 'RETRIEVAL_ROUTING_ENABLED': {'effective': True, 'default': True, 'overridden': False}, 'QUERY_TRANSFORM_ENABLED': {'effective': True, 'default': True, 'overridden': False}, 'TEMPORAL_DECAY_ENABLED': {'effective': False, 'default': False, 'overridden': False}, 'TASK_HOURS_TOP_K': {'effective': 5, 'default': 5, 'overridden': False}, 'TASK_HOURS_DISTANCE_THRESHOLD': {'effective': 0.45, 'default': 0.45, 'overridden': False}, 'HALLUCINATION_GATE_ENABLED': {'effective': True, 'default': True, 'overridden': False}, 'AUGMENTATION_ENABLED': {'effective': True, 'default': True, 'overridden': False}, 'SYNTHESIS_ENABLED': {'effective': True, 'default': True, 'overridden': False}}, 'reranker_model': 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'}`

### Arm `rag`

- cases: 7 (estimation 5, abstention 2)
- within_range_rate: 0.0%
- mean_absolute_error: 440.40 engineer-days
- abstention_correct: False
- mean_latency_ms: 12975
- p95_latency_ms: 15997 (n=7)
- error_rate: 0.0%
- abstention_rate: 0.0%
- source_recall: 40.0%
- total_cost_usd: 0.0100
- mean_cost_usd: 0.0014
- throttled: 0

| case | verdict | expected | predicted | range | abstained | source_hit | latency_ms | cost_usd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S16-01 | failed | 61 | 352 | 37–92 | no (explicit) | no | 13075 | 0.0014 |
| S16-02 | failed | 68 | 410 | 41–102 | no (explicit) | no | 11821 | 0.0013 |
| S16-03 | failed | 67 | 710 | 40–101 | no (explicit) | yes | 10651 | 0.0013 |
| S16-04 | failed | 35 | 670 | 21–53 | no (explicit) | yes | 15498 | 0.0017 |
| S16-05 | failed | 29 | 320 | 17–44 | no (explicit) | no | 12623 | 0.0014 |
| S16-06 | failed | 0 | 264 | 0–0 | no (explicit) | — | 15997 | 0.0016 |
| S16-07 | failed | 0 | 494 | 0–0 | no (explicit) | — | 11158 | 0.0013 |

### Arm `graph`

- cases: 7 (estimation 5, abstention 2)
- within_range_rate: 0.0%
- mean_absolute_error: 38.00 engineer-days
- abstention_correct: False
- mean_latency_ms: 38231
- p95_latency_ms: 54009 (n=7)
- error_rate: 0.0%
- abstention_rate: 28.6%
- source_recall: 0.0%
- total_cost_usd: 0.0137
- mean_cost_usd: 0.0020
- throttled: 0

| case | verdict | expected | predicted | range | abstained | source_hit | latency_ms | cost_usd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S16-01 | failed | 61 | — | 37–92 | yes (proxy) | no | 22732 | 0.0020 |
| S16-02 | failed | 68 | 12 | 41–102 | no (proxy) | no | 42200 | 0.0019 |
| S16-03 | failed | 67 | 16 | 40–101 | no (proxy) | no | 37617 | 0.0020 |
| S16-04 | failed | 35 | 16 | 21–53 | no (proxy) | no | 30337 | 0.0021 |
| S16-05 | failed | 29 | 3 | 17–44 | no (proxy) | no | 35907 | 0.0019 |
| S16-06 | failed | 0 | 4 | 0–0 | no (proxy) | — | 44813 | 0.0020 |
| S16-07 | passed | 0 | — | 0–0 | yes (proxy) | — | 54009 | 0.0019 |

## A/B (rag vs graph)

- quality_delta (within_range_rate rag − graph): `0.0`
- cost_ratio (rag / graph): `0.7282946385046895`
- latency_ratio (rag / graph): `0.33937988943921044`
- abstention_gap (case ids): `['S16-01', 'S16-07']`
- verdict: _not computed — write it by hand in `exercises/session-16/README.md`_
