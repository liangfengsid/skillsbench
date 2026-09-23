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
| [skillsbench_ablation.csv](skillsbench_ablation.csv) | SkillsBench ablations: NONE, NW, NS, NU, BP, ALL |
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

## SkillsBench ablations

Per-seed `summary.json` / `hot_pool.json` links are in [`../README.md`](../README.md#skillsbench-ablations-qwen-test).

| Label | Setting | Directories |
|-------|---------|-------------|
| **NONE** | Default HermesSkills | `skillsbench_no_hot_train0907_test*` |
| **ALL** | Default TipsWarm | `skillsbench_hot_train0917_test*` |
| **NW** | Inherits the HermesSkills warmup skill bundle | `skillsbench_no_hot_train0907_hottest*` |
| **NS** | No scope matching at inject | `skillsbench_hot_train0917_ns_test*` |
| **NU** | No utility attribution or screening | `skillsbench_hot_train0917_nu_test*` |
| **BP** | Store cap 16, inject quota 8 | `skillsbench_hot_train0917_bp_test*` |

Other indexed pools: [`skillsbench_hot_train0917/hot_pool.json`](skillsbench_hot_train0917/hot_pool.json) (Qwen train snapshot), [`skillsbench_hot_train0911_ds/hot_pool.json`](skillsbench_hot_train0911_ds/hot_pool.json) (DeepSeek train snapshot).
