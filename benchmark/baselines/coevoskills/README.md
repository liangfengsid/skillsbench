# CoEvoSkills baseline (SkillsBench + Terminal-Bench)

Isolated reimplementation of **CoEvoSkills** (Zhang et al., [arXiv:2604.01687](https://arxiv.org/abs/2604.01687)) for use as a SkillsBench baseline inside Hermes.

Official upstream code (`Zhang-Henry/CoEvoSkills`) was not released when this package was added; this follows **Algorithm 1** and appendix prompts as closely as practical, using Hermes `AIAgent` only as an LLM/tool backend.

## Design goals

| Goal | How |
|------|-----|
| Minimal core churn | Lives under `benchmark/baselines/coevoskills/`; shared metrics only in `benchmark/scripts/` |
| Faithful Alg. 1 | Skill Generator ↔ Surrogate Verifier; opaque ground-truth oracle |
| Train → freeze → test | `run_split_protocol.py` + `skillsbench_splits/stratified_v1.json` |
| Shared metrics | Same JSONL fields as Hermes → `aggregate_skillsbench_runs.py` |

## Layout

```
benchmark/baselines/coevoskills/
  algorithm.py           # Alg. 1 loop
  generator.py / verifier.py / oracle.py / prompts.py / skill_io.py
  hermes_backend.py
  run_skillsbench.py     # single-task evolve / oracle / hermes-eval
  run_split_protocol.py  # SkillsBench split-aware evolve + frozen library + frozen eval
  run_terminalbench_protocol.py  # Terminal-Bench Harbor evolve + frozen eval
```

Per-task workspaces (default without `--experiment-dir`):
`benchmark/runs/coevoskills/<task_id>/`  
Frozen library default: `…/coevoskills/_frozen_library/`

With **`--experiment-dir DIR`** (recommended):

| Artifact | Path |
|----------|------|
| Evolved skills / workspaces | `DIR/coevoskills/<task_id>/` |
| Frozen library (default) | `DIR/coevoskills/_frozen_library/` |
| Frozen-eval task copies | `DIR/skillsbench/tasks/<task_id>/` |
| Hermes home (opt-in) | `DIR/hermes_home/` with `--isolate-hermes-home` |

There is no separate `--work-root`; evolution storage is always under the
experiment dir (or the default `benchmark/runs/coevoskills` when
`--experiment-dir` is omitted).

## Recommended protocol (cross-task)

1. **Evolve** on `stratified_v1` **train** (skills may change).
2. **Build** a frozen library from train workspaces.
3. **Evaluate** a fresh agent with **frozen** skills on **test** (no further evolution).
4. **Aggregate** macro/micro success@k and cost-to-succeed.

### 1) Evolve on train

```bash
source venv/bin/activate

python -m benchmark.baselines.coevoskills.run_split_protocol \
  --evolve \
  --experiment-dir benchmark/runs/coevo_exp1 \
  --split-file benchmark/skillsbench_splits/stratified_v1.json \
  --split-part train \
  --model Qwen/Qwen3.6-27B \
  --log-jsonl benchmark/runs/coevo_exp1/evolve_train.jsonl

# Agent activity uses a fixed --console-window (default 12 lines) that overwrites
# in place; batch progress lines stay permanent. Errors go to stderr.
# Full scrolling log: add --console-window 0
# Silent agent chatter: add --quiet

# After an accidental interrupt — same command + --resume
# (skips tasks already finished in the JSONL; retries error/interrupted tasks)
python -m benchmark.baselines.coevoskills.run_split_protocol \
  --evolve --experiment-dir benchmark/runs/coevo_exp1 \
  --split-part train --resume \
  --log-jsonl benchmark/runs/coevo_exp1/evolve_train.jsonl

# Optional: resume a manual slice instead
python -m benchmark.baselines.coevoskills.run_split_protocol \
  --evolve --experiment-dir benchmark/runs/coevo_exp1 \
  --split-part train \
  --start-task-index 10 --end-task-index 20 \
  --log-jsonl benchmark/runs/coevo_exp1/evolve_train.jsonl
```

Progress lines look like:
`[evolve] [3/65] START task-id … elapsed=… eta≈…` and per Alg.1 step
`[evolve:task-id] surrogate_iter=2/15 → generator`.

### 2) Build frozen library from train skills

```bash
python -m benchmark.baselines.coevoskills.run_split_protocol \
  --build-library \
  --experiment-dir benchmark/runs/coevo_exp1 \
  --split-file benchmark/skillsbench_splits/stratified_v1.json \
  --library-source-part train
# Library defaults to benchmark/runs/coevo_exp1/coevoskills/_frozen_library
```

### 3) Frozen eval on test (headline transfer metric)

```bash
python -m benchmark.baselines.coevoskills.run_split_protocol \
  --frozen-eval \
  --experiment-dir benchmark/runs/coevo_exp1 \
  --split-file benchmark/skillsbench_splits/stratified_v1.json \
  --split-part test \
  --pass-k 1,5,10,70 \
  --max-iterations 90 \
  --model Qwen/Qwen3.6-27B \
  --install-skills-into-task \
  --log-jsonl benchmark/runs/coevo_exp1/frozen_test.jsonl \
  --aggregate-out benchmark/runs/coevo_exp1/frozen_test_summary.json
```

`--experiment-dir` holds evolve workspaces under `DIR/coevoskills/`, copies
frozen-eval tasks under `DIR/skillsbench/tasks/`, and defaults the frozen
library to `DIR/coevoskills/_frozen_library`. Hermes continues to use
`~/.hermes` unless you pass `--isolate-hermes-home` (seeded from repo
`skills/`, not `~/.hermes/skills`).

Same-task quality (evolve and score the same task’s skills, no pooled library):

```bash
python -m benchmark.baselines.coevoskills.run_split_protocol \
  --frozen-eval --per-task-skills \
  --experiment-dir benchmark/runs/coevo_exp1 \
  --split-part train \
  --pass-k 1,5,10,70 --max-iterations 90 \
  --log-jsonl benchmark/runs/coevo_exp1/frozen_train_same_task.jsonl \
  --aggregate-out benchmark/runs/coevo_exp1/frozen_train_same_task_summary.json
```

### 4) Aggregate / compare metrics (shared across baselines)

```bash
# Core metrics: macro/micro @k + final, cost-to-succeed mean±std
python benchmark/scripts/aggregate_skillsbench_runs.py \
  benchmark/runs/coevo_frozen_test.jsonl \
  --pass-k 1,5,10,70 \
  --max-user-iterations 90 \
  --method coevoskills --phase frozen_eval --split-part test \
  -o benchmark/runs/coevo_frozen_test_summary.json \
  --print-summary

# Hermes hot-pool run (same aggregator)
python benchmark/scripts/run_skillsbench_with_hermes.py --all \
  --split-file benchmark/skillsbench_splits/stratified_v1.json \
  --split-part test \
  --pass-k 1,5,10,70 \
  --hot-pool --hot-pool-persist benchmark/runs/stratified_train_pool.json \
  --log-jsonl benchmark/runs/hermes_hot_test.jsonl

python benchmark/scripts/aggregate_skillsbench_runs.py \
  benchmark/runs/hermes_hot_test.jsonl \
  --pass-k 1,5,10,70 --max-user-iterations 90 \
  --split-part test -o benchmark/runs/hermes_hot_test_summary.json --print-summary

# Hot-pool procedure proxies + embedded core_metrics
python benchmark/scripts/analyze_hot_pool_runs.py \
  benchmark/runs/hermes_hot_test.jsonl \
  --pass-k 1,5,10,70 --max-user-iterations 90 \
  --print-core-summary \
  -o benchmark/runs/hermes_hot_test_hotpool.json
```

## Shared metrics schema

Produced by `benchmark/scripts/skillsbench_aggregate_core.py` (`schema: skillsbench.metrics_summary.v1`):

| Metric | Meaning |
|--------|---------|
| `pass_k[k].macro_success_rate` | Tasks with `task_success` at user iteration *k* |
| `pass_k[k].micro_success_rate` | Pytest cases passed / total at iteration *k* |
| `final.macro_success_rate` / `micro_success_rate` | Same at end of run (within `--max-iterations`) |
| `final.duration_sec` | mean ± std wall-clock seconds per task (all tasks) |
| `final.duration_sec_sum` | total wall time summed across tasks |
| `cost_to_succeed.duration_sec` | mean ± std wall-clock among successful tasks |
| `cost_to_succeed.tokens` | mean ± std among successes (first success checkpoint) |
| `cost_to_succeed.user_iterations` | mean ± std of API turns among successes |
| `final.tokens` / `final.user_iterations` | mean ± std over all tasks (not only successes) |

Per-task JSONL rows include `duration_sec`. Batch runs also append a
`skillsbench.batch_timing.v1` row with wall-clock / mean / sum for that session.

A **user iteration** = one completed Hermes API turn (`api_calls` / `pass_at_turn` index).

JSONL envelopes from this baseline use `schema: skillsbench.baseline_run.v1` with `method: coevoskills` and `phase: evolve|frozen_eval`, plus the same `evaluation` / `pass_at_turn` / `run_conversation_result` fields as Hermes so future baselines can plug into the same aggregators.

## Terminal-Bench

Official Harbor eval (same sandbox + `tests/test.sh` as Hermes
`run_terminalbench_with_harbor.py`). No pass@k.

```bash
# Evolve on stratified train (Harbor sandbox + tests/test.sh oracle)
python -m benchmark.baselines.coevoskills.run_terminalbench_protocol \
  --evolve \
  --experiment-dir benchmark/runs/coevo_tb_exp1 \
  --split-file benchmark/terminalbench_splits/stratified_v1.json \
  --split-part train \
  --model Qwen/Qwen3.6-27B \
  --max-iterations 60 \
  --log-jsonl benchmark/runs/coevo_tb_exp1/evolve_train.jsonl \
  --exclude-task-name math-eval-grader --exclude-task-name jax-speedrun-gpu

# Freeze library, then frozen eval on test (HermesHarborAgent + skill overlay)
python -m benchmark.baselines.coevoskills.run_terminalbench_protocol \
  --build-library --frozen-eval \
  --experiment-dir benchmark/runs/coevo_tb_exp1 \
  --split-part test \
  --model Qwen/Qwen3.6-27B \
  --log-jsonl benchmark/runs/coevo_tb_exp1/frozen_test.jsonl \
  --aggregate-out benchmark/runs/coevo_tb_exp1/frozen_test_summary.json
```

Compare to Hermes (same split, `--env`, isolate, `--no-hot-pool`, no pass@k):

```bash
python3 benchmark/scripts/run_terminalbench_with_harbor.py --all \
  --split-file benchmark/terminalbench_splits/stratified_v1.json \
  --split-part test \
  --experiment-dir benchmark/runs/tb_harbor_test \
  --isolate-hermes-home --no-hot-pool \
  --log-jsonl benchmark/runs/tb_harbor_test/runs.jsonl --print-summary \
  --exclude-task-name math-eval-grader --exclude-task-name jax-speedrun-gpu
```

Frozen eval uses the same `HermesHarborAgent` as the Hermes driver; only isolated
`HERMES_HOME` skills differ (CoEvo frozen library overlay). Aggregate **without**
`--pass-k`.

## Hyperparameters (paper Table A1 defaults)

| Flag | Default | Paper |
|------|---------|--------|
| `--max-oracle-iters` | 5 | N / K = 5 |
| `--max-surrogate-iters` | 15 | M = 15 |
| `--max-iterations` | 90 | Frozen eval agent budget |

## Citation

```
Zhang et al. CoEvoSkills: Self-Evolving Agent Skills via Co-Evolutionary Verification.
arXiv:2604.01687, 2026.
```
