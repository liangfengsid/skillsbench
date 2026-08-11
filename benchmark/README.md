# Hermes benchmark drivers

Hermes ships evaluation harnesses under `benchmark/` for running **`AIAgent`** against public benchmarks and comparing runs. All **Hermes-specific drivers** live in [`scripts/`](scripts/). Vendored benchmark trees (`skillsbench/`, `hle/`, `appworld/`, `alfworld/`) keep their upstream docs.

**Run commands from the Hermes repo root** unless noted otherwise.

## Prerequisites

```bash
source .venv/bin/activate   # or: source venv/bin/activate
```

- **Hermes config:** `~/.hermes/config.yaml` and API keys in `~/.hermes/.env` (see [Configuration](https://hermes-agent.nousresearch.com/docs/user-guide/configuration)).
- **Import path:** drivers prepend `--hermes-root` (default: this repo) to `sys.path` and import `run_agent.AIAgent`.
- **SkillsBench / BenchFlow (optional):** `pip install -e ".[skillsbench]"` from repo root — see [`skillsbench/README.md`](skillsbench/README.md).
- **HLE judge deps:** `pip install -r benchmark/hle/requirements.txt`.
- **AppWorld:** `pip install -e benchmark/appworld` — see [`appworld/README.md`](appworld/README.md).

## Layout

| Path | Role |
|------|------|
| [`scripts/`](scripts/) | Hermes batch drivers and analysis tools |
| [`baselines/`](baselines/) | Isolated third-party / paper baselines (e.g. CoEvoSkills) |
| [`skillsbench/`](skillsbench/) | SkillsBench tasks + BenchFlow (nested project) |
| [`hle/`](hle/) | Humanity's Last Exam dataset / upstream eval notes |
| [`appworld/`](appworld/) | AppWorld environment (vendored) |
| [`alfworld/`](alfworld/) | ALFWorld environment (vendored) |

### CoEvoSkills baseline (SkillsBench)

Isolated Alg. 1 reimplementation under [`baselines/coevoskills/`](baselines/coevoskills/).

**Train → freeze → test** (recommended for cross-task transfer):

```bash
# Evolve on stratified train
python -m benchmark.baselines.coevoskills.run_split_protocol \
  --evolve \
  --split-file benchmark/skillsbench_splits/stratified_v1.json \
  --split-part train \
  --model qwen/qwen3.6-plus \
  --log-jsonl benchmark/runs/coevo_evolve_train.jsonl

# Freeze library, then evaluate on test (no further evolution)
python -m benchmark.baselines.coevoskills.run_split_protocol \
  --build-library --library-source-part train \
  --frozen-eval --split-part test \
  --pass-k 1,5,10,70 --max-iterations 90 \
  --log-jsonl benchmark/runs/coevo_frozen_test.jsonl \
  --aggregate-out benchmark/runs/coevo_frozen_test_summary.json
```

Full details: [`baselines/coevoskills/README.md`](baselines/coevoskills/README.md).

### Shared metrics (Hermes, CoEvoSkills, future baselines)

All drivers should log JSONL with `evaluation`, optional `pass_at_turn`, and
`run_conversation_result` (tokens / `api_calls`). Then:

```bash
python benchmark/scripts/aggregate_skillsbench_runs.py RUN.jsonl \
  --pass-k 1,5,10,70 --max-user-iterations 90 \
  -o RUN_summary.json --print-summary
```

Reports **macro/micro success@k**, **final** rates within max iterations, and
**cost-to-succeed** mean±std (tokens + user iterations). See
[`scripts/skillsbench_aggregate_core.py`](scripts/skillsbench_aggregate_core.py).

Hot-pool-specific proxies (plus the same core block as `core_metrics`):

```bash
python benchmark/scripts/analyze_hot_pool_runs.py RUN.jsonl \
  --pass-k 1,5,10,70 --max-user-iterations 90 \
  --print-core-summary -o RUN_hotpool.json
```


## Scripts overview

| Script | Purpose |
|--------|---------|
| [`run_skillsbench_with_hermes.py`](scripts/run_skillsbench_with_hermes.py) | Run Hermes on SkillsBench tasks; JSONL logs + host eval |
| [`evaluate_skillsbench_task.py`](scripts/evaluate_skillsbench_task.py) | Host pytest verifier for one task (used by driver) |
| [`skillsbench_metrics.py`](scripts/skillsbench_metrics.py) | Build compact `metrics` blocks for JSONL rows |
| [`aggregate_skillsbench_runs.py`](scripts/aggregate_skillsbench_runs.py) | Shared macro/micro success@k + cost-to-succeed mean±std |
| [`skillsbench_aggregate_core.py`](scripts/skillsbench_aggregate_core.py) | Core metrics library used by aggregators + baselines |
| [`make_skillsbench_splits.py`](scripts/make_skillsbench_splits.py) | Generate train/val/test split JSON (stratified or category holdout) |
| [`compare_skillsbench_runs.py`](scripts/compare_skillsbench_runs.py) | Compare two SkillsBench JSONL runs (tokens, cost, API calls) |
| [`analyze_hot_pool_runs.py`](scripts/analyze_hot_pool_runs.py) | Hot skill pool telemetry + procedure proxies + core metrics |
| [`read_skillsbench_jsonl.py`](scripts/read_skillsbench_jsonl.py) | Load SkillsBench JSONL into Python |
| [`run_model_predictions_hermes.py`](scripts/run_model_predictions_hermes.py) | HLE predictions via in-process Hermes |
| [`run_judge_results_hermes.py`](scripts/run_judge_results_hermes.py) | HLE judge pass on predictions |
| [`hle_hermes_inprocess.py`](scripts/hle_hermes_inprocess.py) | Shared HLE helper (imported by drivers) |
| [`run_appworld_with_hermes.py`](scripts/run_appworld_with_hermes.py) | AppWorld ReAct loop with Hermes code generation |

Each script supports `--help`.

---

## SkillsBench

Task authoring and BenchFlow CLI: [`skillsbench/README.md`](skillsbench/README.md), [`skillsbench/AGENTS.md`](skillsbench/AGENTS.md).

### Run tasks with Hermes

Defaults: `--hermes-root` = repo root; `--skillsbench-root` = `benchmark/skillsbench/`; `--prompt-tasks-base` = absolute path to `benchmark/skillsbench/tasks/`.

```bash
# List task ids (no Hermes import)
python3 benchmark/scripts/run_skillsbench_with_hermes.py --list-tasks

# Single task — hot pool on + persist (treatment)
python3 benchmark/scripts/run_skillsbench_with_hermes.py \
  --task adaptive-cruise-control \
  --model qwen/qwen3.6-plus \
  --skill-nudge-interval 10 \
  --memory-nudge-interval 10 \
  --hot-pool \
  --hot-pool-persist benchmark/skillsbench_hot_pool.json \
  --log-jsonl benchmark/hermes_skillsbench_runs.jsonl \
  --print-summary

# Single task — hot pool off (control)
python3 benchmark/scripts/run_skillsbench_with_hermes.py \
  --task adaptive-cruise-control \
  --model qwen/qwen3.6-plus \
  --skill-nudge-interval 10 \
  --memory-nudge-interval 10 \
  --no-hot-pool \
  --log-jsonl benchmark/hermes_skillsbench_runs.jsonl \
  --print-summary

# Batch (sorted task order; continues after errors)
python3 benchmark/scripts/run_skillsbench_with_hermes.py --all \
  --model qwen/qwen3.6-plus \
  --hot-pool \
  --hot-pool-persist benchmark/skillsbench_hot_pool.json \
  --log-jsonl benchmark/hermes_skillsbench_runs.jsonl

# Batch — hot pool off
python3 benchmark/scripts/run_skillsbench_with_hermes.py --all \
  --model qwen/qwen3.6-plus \
  --no-hot-pool \
  --log-jsonl benchmark/hermes_skillsbench_runs.jsonl

# Isolated experiment workspace (recommended for hot-pool / multi-run):
# copies selected tasks under DIR/skillsbench/tasks/, defaults hot-pool JSON
# to DIR/hot_pool.json, and isolates HERMES_HOME under DIR/hermes_home.
python3 benchmark/scripts/run_skillsbench_with_hermes.py --all \
  --experiment-dir benchmark/runs/exp_hot_train \
  --split-file benchmark/skillsbench_splits/stratified_v1.json \
  --split-part train \
  --model qwen/qwen3.6-plus \
  --hot-pool \
  --log-jsonl benchmark/runs/exp_hot_train/runs.jsonl \
  --print-summary

# Slice of tasks: [start, end) in sorted order
python3 benchmark/scripts/run_skillsbench_with_hermes.py --all \
  --start-task-index 0 --end-task-index 10 \
  --model qwen/qwen3.6-plus \
  --hot-pool \
  --hot-pool-persist benchmark/skillsbench_hot_pool.json \
  --log-jsonl benchmark/hermes_skillsbench_runs.jsonl
```

**Useful flags**

| Flag | Default | Notes |
|------|---------|--------|
| `--model` | from Hermes config | OpenRouter-style id, e.g. `qwen/qwen3.6-plus` |
| `--max-iterations` | 90 | Tool-calling cap per conversation |
| `--no-quiet` | off | More console output from Hermes |
| `--skip-context-files` | off | Skip AGENTS.md-style context injection |
| `--skip-memory` | off | Disable persistent memory |
| `--skill-nudge-interval` / `--memory-nudge-interval` | from config | Override nudge counters after agent init |
| `--hot-pool` / `--no-hot-pool` | follow config | Force hot skill pool on or off for this run (see below) |
| `--hot-pool-persist PATH` | off | Load/save hot pool across tasks; requires pool enabled; incompatible with `--no-hot-pool` |
| `--experiment-dir DIR` | off | Per-experiment workspace: task copies + default hot pool + isolated `HERMES_HOME` |
| `--reset-task-workspaces` | off | With `--experiment-dir`: refresh task copies / wipe prior agent outputs |
| `--no-isolate-hermes-home` | off | With `--experiment-dir`: keep using the real `HERMES_HOME` |
| `--no-batch-review-prompt` | off | Use default Hermes review prompt instead of SkillsBench batch appendix |
| `--no-wait-background-review` | off | Exit without waiting for end-of-turn skill/memory review |
| `--background-review-timeout SEC` | 180 | Max wait for background review per task |
| `--stop-on-error` | off | Abort `--all` on first exception |
| `--evaluate-after-run` / `--no-evaluate-after-run` | on | Host pytest verifier after each agent run |
| `--evaluate-only` | off | Skip agent; score existing outputs on disk |
| `--eval-timeout-sec` | 600 | Max seconds per host pytest eval |
| `--pass-k TURNS` | off | Conversation turns for pass@k (e.g. `1,5,10,70`) |
| `--print-batch-summary` / `--no-print-batch-summary` | on for `--all` | Aggregate metrics after a multi-task run |

**Background review:** By default the driver waits up to **180s** after each task for end-of-turn skill/memory review (so `skill_manage` and hot-pool updates are not killed when the process exits). JSONL rows include `background_review: {spawned, completed, timeout, actions, telemetry}` where `telemetry.tools` lists review-agent tool calls (e.g. `skill_manage`). The driver appends a SkillsBench-specific review prompt by default (host/container path pitfalls, verification thrashing); use `--no-batch-review-prompt` to disable. Use `--no-wait-background-review` to skip waiting; `--background-review-timeout SEC` to change the limit.

### Performance metrics (eval + pass@k)

**Multi-task runs** — use `--all` with an optional split file. Each task gets its own JSONL row (same file, appended). After the batch finishes, a **batch summary** prints automatically when `--all` is used (disable with `--no-print-batch-summary`).

```bash
# Train split (65 tasks): pass@k + hot pool learning + full metrics logging
python3 benchmark/scripts/run_skillsbench_with_hermes.py --all \
  --split-file benchmark/skillsbench_splits/stratified_v1.json \
  --split-part train \
  --pass-k 1,5,10,70 \
  --model qwen/qwen3.6-plus \
  --hot-pool \
  --hot-pool-persist benchmark/runs/stratified_train_pool.json \
  --log-jsonl benchmark/runs/stratified_train.jsonl \
  --print-summary

# Aggregate the same JSONL later (or combine multiple run files)
python3 benchmark/scripts/aggregate_skillsbench_runs.py \
  benchmark/runs/stratified_train.jsonl \
  --pass-k 1,5,10,70 \
  --split-part train \
  --print-summary \
  -o benchmark/runs/stratified_train_summary.json
```

**Final evaluation** — after the conversation ends, host pytest runs against `tests/test_outputs.py` (with `/root/` paths remapped). Logged as `evaluation` on each JSONL row:

| Field | Meaning |
|-------|---------|
| `evaluation.task_success` | **Macro** — all verifier tests passed at end of run |
| `evaluation.reward` | `tests_passed / tests_total` (1.0 when macro pass) |
| `evaluation.tests_passed` / `tests_total` | **Micro** — per-test-case counts at end of run |
| `run_conversation_result.total_tokens` | Tokens consumed by the full conversation |
| `run_conversation_result.cache_read_tokens` | Prompt-cache read tokens |
| `run_conversation_result.reasoning_tokens` | Reasoning tokens (when provider reports them) |
| `run_conversation_result.estimated_cost_usd` | Estimated session cost |
| `metrics.final.tool_calls` | Per-tool invocation counts from message history |
| `metrics.final.tool_rounds` | Total tool calls in the conversation |

**pass@k (conversation turn k)** — within **one** agent run, the driver evaluates task outputs at the end of agent conversation **turn** *k* (one completed API iteration + tool execution). No reruns and no early termination. Logged as `pass_at_turn` and summarized in `metrics.pass_at_turn`:

```json
"pass_k_turns": [1, 5, 10, 70],
"pass_at_turn": {
  "5": {
    "turn": 5,
    "evaluation": { "task_success": false, "tests_passed": 8, "tests_total": 12, "reward": 0.667 },
    "api_calls": 5,
    "tokens": { "total": 420000, "input": 380000, "output": 40000, "cache_read": 100000, "reasoning": 5000 },
    "estimated_cost_usd": 0.12
  }
},
"metrics": {
  "final": { "task_success": true, "micro_pass_rate": 1.0, "tokens": { "total": 1400000 }, ... },
  "pass_at_turn": { "5": { "task_success": false, "micro_pass_rate": 0.667, "tokens": { "total": 420000 }, ... } }
}
```

```bash
# Single task — final eval (default) + pass@k at turns 1,5,10,70
python3 benchmark/scripts/run_skillsbench_with_hermes.py \
  --task adaptive-cruise-control \
  --model qwen/qwen3.6-plus \
  --pass-k 1,5,10,70 \
  --hot-pool --hot-pool-persist benchmark/skillsbench_hot_pool.json \
  --log-jsonl benchmark/hermes_skillsbench_runs.jsonl \
  --print-summary

# Batch with pass@k (one row per task; each row has pass_at_turn snapshots)
python3 benchmark/scripts/run_skillsbench_with_hermes.py --all \
  --split-file benchmark/skillsbench_splits/stratified_v1.json \
  --split-part test \
  --pass-k 1,5,10,70 \
  --no-hot-pool \
  --model qwen/qwen3.6-plus \
  --log-jsonl benchmark/runs/stratified_test.jsonl

# Aggregate across tasks: macro task pass rate + micro test pass rate at each turn
python3 benchmark/scripts/aggregate_skillsbench_runs.py \
  benchmark/runs/stratified_test.jsonl \
  --pass-k 1,5,10,70 \
  --print-summary \
  -o benchmark/runs/stratified_test_summary.json

# Re-score final outputs without re-running the agent
python3 benchmark/scripts/run_skillsbench_with_hermes.py \
  --task adaptive-cruise-control --evaluate-only \
  --log-jsonl benchmark/hermes_skillsbench_runs.jsonl --print-summary

# Agent run only (no pytest)
python3 benchmark/scripts/run_skillsbench_with_hermes.py \
  --task adaptive-cruise-control --no-evaluate-after-run \
  --log-jsonl benchmark/hermes_skillsbench_runs.jsonl
```

**Aggregate output** (`aggregate_skillsbench_runs.py`):

- `pass_k.<turn>.macro_task_pass_rate` — fraction of tasks passing all tests at turn *k*
- `pass_k.<turn>.micro_test_pass_rate` — Σ passed test cases / Σ total at turn *k*
- `pass_k.<turn>.tokens_mean_at_turn` / `input_tokens_mean_at_turn` / `api_calls_mean_at_turn`
- `pass_k.<turn>.estimated_cost_usd_mean_at_turn` / `reward_mean`
- `final.macro_task_pass_rate` / `micro_test_pass_rate` / `reward_mean`
- `final.tokens_mean` / `input_tokens_mean` / `api_calls_mean` / `tool_rounds_mean`
- `final.estimated_cost_usd_mean` / `estimated_cost_usd_sum` / `duration_sec_mean`
- `final.completion_rate` — fraction of tasks with `completed=true`

### Train / test splits

SkillsBench has **no upstream train/test partition**. Hermes ships reproducible splits under [`skillsbench_splits/`](skillsbench_splits/):

| File | Protocol | Partitions (88 tasks) |
|------|----------|------------------------|
| [`stratified_v1.json`](skillsbench_splits/stratified_v1.json) | Difficulty-stratified (75/25, seed 42) | 65 train / 23 test |
| [`category_holdout_v1.json`](skillsbench_splits/category_holdout_v1.json) | Whole categories (≥3 tasks) held out to test | 64 train / 24 test |

Regenerate or customize:

```bash
python3 benchmark/scripts/make_skillsbench_splits.py --write-defaults

python3 benchmark/scripts/make_skillsbench_splits.py \
  --protocol difficulty_stratified --seed 42 \
  -o benchmark/skillsbench_splits/stratified_v1.json
```

Run a partition with the Hermes driver:

```bash
# Train split — skill updates / hot pool learning
python3 benchmark/scripts/run_skillsbench_with_hermes.py --all \
  --split-file benchmark/skillsbench_splits/stratified_v1.json \
  --split-part train \
  --hot-pool \
  --hot-pool-persist benchmark/runs/stratified_train_pool.json \
  --log-jsonl benchmark/runs/stratified_train.jsonl \
  --model qwen/qwen3.6-plus

# Test split — inject key points from train pool file
python3 benchmark/scripts/run_skillsbench_with_hermes.py --all \
  --split-file benchmark/skillsbench_splits/stratified_v1.json \
  --split-part test \
  --hot-pool \
  --hot-pool-persist benchmark/runs/stratified_train_pool.json \
  --log-jsonl benchmark/runs/stratified_test.jsonl \
  --model qwen/qwen3.6-plus
```

Category-holdout test (unseen domains):

```bash
python3 benchmark/scripts/run_skillsbench_with_hermes.py --all \
  --split-file benchmark/skillsbench_splits/category_holdout_v1.json \
  --split-part test \
  --hot-pool \
  --hot-pool-persist benchmark/runs/category_holdout_train_pool.json \
  --log-jsonl benchmark/runs/category_holdout_test.jsonl
```

### Hot skill key points (cross-task pool)

Hot pool injects recently learned skill **key points** ephemerally each API turn (see [`agent/hot_skills.py`](../agent/hot_skills.py)). The driver can override `skills.hot_pool.enabled` in `~/.hermes/config.yaml` without editing config:

| CLI | Effect |
|-----|--------|
| *(omit both flags)* | Follow `skills.hot_pool.enabled` in config |
| `--hot-pool` | Force enable for this run |
| `--no-hot-pool` | Force disable for this run (also disables persistence) |
| `--hot-pool-persist PATH` | Load/save pool JSON across tasks (implies pool must be enabled) |

**Treatment** — pool enabled with persistence across a split or batch:

```bash
python3 benchmark/scripts/run_skillsbench_with_hermes.py --all \
  --hot-pool \
  --hot-pool-persist benchmark/skillsbench_hot_pool.json \
  --log-jsonl benchmark/runs_hot_pool_treatment.jsonl \
  --model qwen/qwen3.6-plus
```

**Control** — hot pool off for this run (ignores config and `--hot-pool-persist`):

```bash
python3 benchmark/scripts/run_skillsbench_with_hermes.py --all \
  --no-hot-pool \
  --model qwen/qwen3.6-plus \
  --log-jsonl benchmark/runs_hot_pool_control.jsonl
```

Single-task examples:

```bash
# Hot pool on + persist (A/B treatment arm)
python3 benchmark/scripts/run_skillsbench_with_hermes.py \
  --task adaptive-cruise-control \
  --hot-pool --hot-pool-persist benchmark/skillsbench_hot_pool.json \
  --log-jsonl benchmark/hermes_skillsbench_runs.jsonl --print-summary

# Hot pool off (control arm)
python3 benchmark/scripts/run_skillsbench_with_hermes.py \
  --task adaptive-cruise-control \
  --no-hot-pool \
  --log-jsonl benchmark/hermes_skillsbench_runs.jsonl --print-summary
```

JSONL rows record `hot_pool_enabled` (`true` / `false` / `null`) and `hot_pool_persist`. When the pool is active, `hot_pool_telemetry.inject` shows `point_count`, `skills_injected`, and `points_injected`.

Env vars (set by the driver): `HERMES_HOT_POOL_ENABLED=0|1`, `HERMES_HOT_POOL_PERSIST=1`, `HERMES_HOT_POOL_PATH=/path/to/pool.json`.

### Clean task trees before A/B runs

Hermes runs **write into** `benchmark/skillsbench/tasks/<task-id>/` (solution files, `mass_report.json`, local verify scripts, etc.). Leftover artifacts make iteration counts unreliable — a later run may “verify existing output” in fewer steps while a dirty tree inflates or deflates comparisons.

`benchmark/skillsbench/` is a **nested git checkout**. Reset one task or the whole tree before a clean comparison:

```bash
cd benchmark/skillsbench

# One task — discard tracked edits and remove untracked agent outputs
git restore tasks/adaptive-cruise-control
git clean -fd tasks/adaptive-cruise-control/

# All tasks — same, repo-wide (review untracked list first)
git restore .
git clean -fdn    # dry run: shows what would be deleted
git clean -fd       # destructive: removes agent-generated files under tasks/
```

Then run from the Hermes repo root. **Copy-paste A/B** on one task after cleanup:

```bash
# 1) Reset task tree (from Hermes repo root)
cd benchmark/skillsbench
git restore tasks/adaptive-cruise-control
git clean -fd tasks/adaptive-cruise-control/
cd ../..

# 2) Control — hot pool off
python3 benchmark/scripts/run_skillsbench_with_hermes.py \
  --task adaptive-cruise-control \
  --model qwen/qwen3.6-plus \
  --skill-nudge-interval 10 \
  --memory-nudge-interval 10 \
  --no-hot-pool \
  --log-jsonl benchmark/hermes_skillsbench_runs.jsonl \
  --print-summary

# 3) Reset again
cd benchmark/skillsbench
git restore tasks/adaptive-cruise-control
git clean -fd tasks/adaptive-cruise-control/
cd ../..

# 4) Treatment — hot pool on + persist
python3 benchmark/scripts/run_skillsbench_with_hermes.py \
  --task adaptive-cruise-control \
  --model qwen/qwen3.6-plus \
  --skill-nudge-interval 10 \
  --memory-nudge-interval 10 \
  --hot-pool \
  --hot-pool-persist benchmark/skillsbench_hot_pool.json \
  --log-jsonl benchmark/hermes_skillsbench_runs.jsonl \
  --print-summary
```

Do **not** commit agent outputs under `tasks/`; treat them as ephemeral benchmark state.

### Analyze runs

```bash
# Token / API / cost comparison (two JSONL logs)
python3 benchmark/scripts/compare_skillsbench_runs.py \
  benchmark/runs_hot_pool_control.jsonl \
  benchmark/runs_hot_pool_treatment.jsonl \
  --label-a control --label-b hot-pool \
  --html benchmark/skillsbench_compare_report.html

# Hot pool telemetry + alignment / guardrail proxies
python3 benchmark/scripts/analyze_hot_pool_runs.py \
  benchmark/runs_hot_pool_treatment.jsonl \
  -o benchmark/hot_pool_summary.json

python3 benchmark/scripts/analyze_hot_pool_runs.py \
  benchmark/runs_hot_pool_control.jsonl \
  benchmark/runs_hot_pool_treatment.jsonl \
  --label-a control --label-b hot-pool \
  -o benchmark/hot_pool_compare.json
```

Load JSONL in Python: `from read_skillsbench_jsonl import load_skillsbench_run_records` (run from `benchmark/scripts/` or add that directory to `sys.path`).

---

## HLE (Humanity's Last Exam)

Upstream dataset docs: [`hle/README.md`](hle/README.md). Hermes drivers call in-process **`AIAgent` with tools disabled**.

```bash
pip install -r benchmark/hle/requirements.txt   # once, in Hermes venv

DATASET="cais/hle"
MODEL="gpt-4o-2024-11-20"

# Predictions → benchmark/hle/hle_<model>_hermes.json (slashes in model → underscores)
python3 benchmark/scripts/run_model_predictions_hermes.py \
  --dataset "$DATASET" --model "$MODEL" \
  --num_workers 10 --max_completion_tokens 8192

# Quick smoke test with local JSON
python3 benchmark/scripts/run_model_predictions_hermes.py \
  --dataset_file benchmark/hle/_smoke_first_question.json \
  --model "$MODEL" --max_samples 1

# Judge → benchmark/hle/judged_<basename>_hermes.json
python3 benchmark/scripts/run_judge_results_hermes.py \
  --dataset "$DATASET" \
  --predictions "benchmark/hle/hle_${MODEL}_hermes.json" \
  --judge "$MODEL" \
  --num_workers 10
```

Optional: set `HERMES_AGENT_REPO` if `run_agent` is not importable from the current checkout.

---

## AppWorld

Install AppWorld from the vendored tree, then run the Hermes driver. Pass **`--model`** explicitly (OpenRouter-style id); omitting it only works when `~/.hermes/config.yaml` already resolves a default model.

```bash
pip install -e benchmark/appworld

python3 benchmark/scripts/run_appworld_with_hermes.py --list-tasks
python3 benchmark/scripts/run_appworld_with_hermes.py --list-tasks --dataset dev

python3 benchmark/scripts/run_appworld_with_hermes.py \
  --dataset dev --task 50e1ac9_1 \
  --model qwen/qwen3.6-plus \
  --log-jsonl benchmark/appworld_hermes_runs.jsonl

python3 benchmark/scripts/run_appworld_with_hermes.py \
  --dataset dev --all --start-task-index 0 --end-task-index 5 \
  --model qwen/qwen3.6-plus \
  --experiment-name hermes-dev \
  --log-jsonl benchmark/appworld_runs.jsonl \
  --print-summary
```

**AppWorld data:** from `benchmark/appworld/`, run `appworld download` once (sets up `data/` under `APPWORLD_ROOT`; default is the AppWorld repo root). See [`appworld/README.md`](appworld/README.md) for `APPWORLD_ROOT` and full setup.

**Hermes tools:** the driver exposes all configured Hermes toolsets by default (terminal, files, web search, skills, etc.). AppWorld API actions still run via ` ```python ` blocks executed in the AppWorld REPL. Use `--no-tools` to restore the original code-only ReAct mode. With tools enabled, each AppWorld step allows up to 90 Hermes tool-calling iterations by default (`--max-hermes-iterations` overrides).

### Evaluate task results

The Hermes driver calls AppWorld’s `evaluate_task()` after each agent run **by default**. Task state is saved under:

```text
benchmark/appworld/experiments/outputs/<experiment-name>/tasks/<task_id>/dbs/
```

Use the same **`--experiment-name`** for run and eval (default: `hermes-agent`).

**Run agent + evaluate** (evaluation block is included in the JSONL row):

```bash
python3 benchmark/scripts/run_appworld_with_hermes.py \
  --dataset dev --task 50e1ac9_1 \
  --model qwen/qwen3.6-plus \
  --experiment-name hermes-agent \
  --log-jsonl benchmark/appworld_hermes_runs.jsonl \
  --print-summary
```

With `--print-summary`, stdout includes `eval_success`, step count, and tokens. The JSONL record has an `evaluation` object, for example:

| Field | Meaning |
|-------|---------|
| `evaluation.success` | Task passed all tests |
| `evaluation.pass_count` / `fail_count` / `num_tests` | Test counts |
| `evaluation.pass_percentage` | Pass rate for the task |
| `evaluation.report_path` | Markdown report on disk |

Human-readable report (when evaluation succeeds):

```text
benchmark/appworld/experiments/outputs/<experiment-name>/tasks/<task_id>/evaluation/report.md
```

**Re-evaluate without re-running the agent** (uses saved `dbs/` from a prior run):

```bash
python3 benchmark/scripts/run_appworld_with_hermes.py \
  --evaluate-only \
  --dataset dev --task 50e1ac9_1 \
  --experiment-name hermes-agent \
  --log-jsonl benchmark/appworld_hermes_eval.jsonl \
  --print-summary
```

Batch re-eval prints a summary at the end (task success rate, test pass ratio, skipped count). Token totals appear when you pass the agent-run JSONL via `--run-log-jsonl` (or reuse `--log-jsonl` if that file already contains run records):

```bash
python3 benchmark/scripts/run_appworld_with_hermes.py \
  --evaluate-only --dataset dev --all \
  --experiment-name hermes-dev \
  --run-log-jsonl benchmark/appworld_runs.jsonl \
  --log-jsonl benchmark/appworld_hermes_eval.jsonl \
  --print-summary
```

Example summary:

```text
=== AppWorld evaluate summary (hermes-dev / dev) ===
tasks: 168 total | 5 evaluated | 163 skipped | 0 errors
task success: 4/5 (80.0%)
test passes: 18/20 (90.0%)
tokens: 1,435,564 total | 287,113 avg (5/5 evaluated tasks from run log)
```

**Skip evaluation** during the agent run (e.g. debug codegen only):

```bash
python3 benchmark/scripts/run_appworld_with_hermes.py \
  --dataset dev --task 50e1ac9_1 \
  --model qwen/qwen3.6-plus \
  --no-evaluate
```

**Upstream CLI** (from `benchmark/appworld/` after `pip install -e .`):

```bash
cd benchmark/appworld
appworld evaluate hermes-agent dev    # experiment name ^   dataset ^
```

See [`appworld/README.md`](appworld/README.md) for environment setup and leaderboard packing.

---

## ALFWorld

Vendored at [`alfworld/`](alfworld/). Install from source (Python 3.9+ recommended upstream; 3.12 works with Hermes venv):

```bash
cd benchmark/alfworld
uv pip install -e ".[full]"
```

If `visdom` fails to build under `uv`, `benchmark/alfworld/pyproject.toml` pins `[tool.uv.extra-build-dependencies]` for that package. See [`alfworld/README.md`](alfworld/README.md) for data download and play scripts.

There is no Hermes batch driver in `benchmark/scripts/` for ALFWorld yet; use upstream training/play scripts or add a driver following the SkillsBench pattern.

---

## Output files

JSONL logs are usually written under `benchmark/`:

| Driver | Schema | Notes |
|--------|--------|--------|
| SkillsBench | `skillsbench.hermes_run.v1` | `run_conversation_result`, `hot_pool_enabled`, `hot_pool_persist`, optional `hot_pool_telemetry`, `background_review` (`actions`, `telemetry`) |
| AppWorld (run) | `appworld.hermes_run.v1` | `steps`, `hermes_stats`, `evaluation` (unless `--no-evaluate`) |
| AppWorld (eval only) | `appworld.hermes_eval.v1` | `evaluation`; optional `run_hermes_stats` when `--run-log-jsonl` matches |
| AppWorld (eval skip) | `appworld.hermes_eval_skip.v1` | `skip_reason` when task was not run (`--evaluate-only`) |

Errors use `*.hermes_run_error.v1` schemas on failure lines.
