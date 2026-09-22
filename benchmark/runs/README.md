# Indexed experiment artifacts

Git tracks only lightweight files under `benchmark/runs/`:

- `*/summary.json` — aggregator output (Success@k / AUC / final / cost-to-succeed)
- `*/hot_pool.json` — TipsWarm tip pool snapshot
- `*.csv` — paper-facing headline tables

JSONL logs, `hermes_home/`, task copies, A-MEM Chroma trees, and DC `episodes.jsonl` are **not** indexed. See the [root README](../../README.md) for how to re-run.

## Headline CSVs

| File | Contents |
|------|----------|
| [skillsbench_main.csv](skillsbench_main.csv) | SkillsBench, Qwen3.6-27B, four methods |
| [skillsbench_model.csv](skillsbench_model.csv) | SkillsBench, DeepSeek vs Qwen, HermesSkills / TipsWarm |
| [appworld.csv](appworld.csv) | AppWorld `test_normal` |
| [alfworld.csv](alfworld.csv) | ALFWorld `valid_unseen` |

## Method directory names

| Prefix | Method |
|--------|--------|
| `*_hot_*` | TipsWarm |
| `*_no_hot_*` | HermesSkills (no tip pool) |
| `*_amem*` | A-MEM |
| `*_dc*` | Dynamic Cheatsheet |

Train (warmup) dirs omit `_test` / `_unseen`. Eval dirs append `_test*`, `_unseen*`, or `_hottest*` (HermesSkills warmup, TipsWarm inject on the test split).

## Other indexed TipsWarm pools

SkillsBench ablations and transfers (pool only, or pool + summary):

- [`skillsbench_hot_train0917/hot_pool.json`](skillsbench_hot_train0917/hot_pool.json) — train snapshot
- [`skillsbench_hot_train0911_ds/hot_pool.json`](skillsbench_hot_train0911_ds/hot_pool.json) — DeepSeek train snapshot
- [`skillsbench_no_hot_train0907_hottest1`](skillsbench_no_hot_train0907_hottest1/summary.json) / [hottest2](skillsbench_no_hot_train0907_hottest2/summary.json) / [hottest3](skillsbench_no_hot_train0907_hottest3/summary.json) — HermesSkills train, TipsWarm at test
- [`skillsbench_hot_train0917_ns_test1/hot_pool.json`](skillsbench_hot_train0917_ns_test1/hot_pool.json), [`ns_test2`](skillsbench_hot_train0917_ns_test2/hot_pool.json) — no-scope retrieve
- [`skillsbench_hot_train0917_nu_test1/hot_pool.json`](skillsbench_hot_train0917_nu_test1/hot_pool.json), [`nu_test2`](skillsbench_hot_train0917_nu_test2/hot_pool.json) — no-utility inject filter
