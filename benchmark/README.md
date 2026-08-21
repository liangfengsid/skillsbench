# Hermes benchmark drivers

Hermes ships evaluation harnesses under `benchmark/` for running **`AIAgent`** against public benchmarks and comparing runs. All **Hermes-specific drivers** live in [`scripts/`](scripts/). Vendored benchmark trees (`skillsbench/`, `terminal-bench/`, `alfworld/`, `appworld/`) keep their upstream docs.

**Run commands from the Hermes repo root** unless noted otherwise.

## Prerequisites

```bash
source .venv/bin/activate   # or: source venv/bin/activate
```

- **Hermes config:** `~/.hermes/config.yaml` and API keys in `~/.hermes/.env` (see [Configuration](https://hermes-agent.nousresearch.com/docs/user-guide/configuration)).
- **Import path:** drivers prepend `--hermes-root` (default: this repo) to `sys.path` and import `run_agent.AIAgent`.
- **SkillsBench / BenchFlow (optional):** `pip install -e ".[skillsbench]"` from repo root — see [`skillsbench/README.md`](skillsbench/README.md).
- **Harbor / official Terminal-Bench (optional):** `pip install -e ".[harbor]"` plus Docker (or `pip install "harbor[modal]"` + Modal). Not included in `[all]`.
- **ALFWorld (optional):** `pip install -e ".[alfworld]"` plus PDDL/game files in `ALFWORLD_DATA`. Not included in `[all]`.
- **AppWorld:** `pip install -e benchmark/appworld` — see [`appworld/README.md`](appworld/README.md).

## Layout

| Path | Role |
|------|------|
| [`scripts/`](scripts/) | Hermes batch drivers and analysis tools |
| [`harbor_adapter/`](harbor_adapter/) | Hermes `BaseAgent` for official Harbor / Terminal-Bench eval |
| [`alfworld_adapter/`](alfworld_adapter/) | ALFWorld TextWorld discovery + env wrapper for Hermes |
| [`baselines/`](baselines/) | Isolated third-party / paper baselines (e.g. CoEvoSkills) |
| [`skillsbench/`](skillsbench/) | SkillsBench tasks + BenchFlow (nested project) |
| [`terminal-bench/`](terminal-bench/) | Terminal-Bench tasks (Harbor dataset tree) |
| [`alfworld/`](alfworld/) | Vendored ALFWorld (TextWorld / ALFRED) |
| [`appworld/`](appworld/) | AppWorld environment (vendored) |

### CoEvoSkills baseline (SkillsBench)

Isolated Alg. 1 reimplementation under [`baselines/coevoskills/`](baselines/coevoskills/).

**Train → freeze → test** (recommended for cross-task transfer):

```bash
# Evolve on stratified train
python -m benchmark.baselines.coevoskills.run_split_protocol \
  --evolve \
  --split-file benchmark/skillsbench_splits/stratified_v1.json \
  --split-part train \
  --model Qwen/Qwen3.6-27B \
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

CoEvoSkills on **Terminal-Bench** uses Harbor for both the evolve oracle and frozen eval (same verifier as Hermes). See [`baselines/coevoskills/README.md`](baselines/coevoskills/README.md#terminal-bench).

```bash
python -m benchmark.baselines.coevoskills.run_terminalbench_protocol \
  --evolve --experiment-dir benchmark/runs/coevo_tb_exp1 \
  --split-file benchmark/terminalbench_splits/stratified_v1.json \
  --split-part train --model Qwen/Qwen3.6-27B \
  --max-iterations 60 \
  --log-jsonl benchmark/runs/coevo_tb_exp1/evolve_train.jsonl \
  --exclude-task-name math-eval-grader --exclude-task-name jax-speedrun-gpu

python -m benchmark.baselines.coevoskills.run_terminalbench_protocol \
  --build-library --frozen-eval \
  --experiment-dir benchmark/runs/coevo_tb_exp1 --split-part test \
  --model Qwen/Qwen3.6-27B \
  --log-jsonl benchmark/runs/coevo_tb_exp1/frozen_test.jsonl
```

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
| [`run_terminalbench_with_harbor.py`](scripts/run_terminalbench_with_harbor.py) | Terminal-Bench: Hermes agent + official Harbor verifier |
| [`run_alfworld_with_hermes.py`](scripts/run_alfworld_with_hermes.py) | ALFWorld TextWorld: Hermes picks admissible commands |
| [`run_terminalbench_protocol.py`](baselines/coevoskills/run_terminalbench_protocol.py) | CoEvoSkills on Terminal-Bench (Harbor evolve oracle + frozen eval) |
| [`evaluate_skillsbench_task.py`](scripts/evaluate_skillsbench_task.py) | Host pytest / `test.sh` verifier for SkillsBench |
| [`skillsbench_metrics.py`](scripts/skillsbench_metrics.py) | Build compact `metrics` blocks for JSONL rows |
| [`aggregate_skillsbench_runs.py`](scripts/aggregate_skillsbench_runs.py) | Shared macro/micro success@k + cost-to-succeed mean±std |
| [`skillsbench_aggregate_core.py`](scripts/skillsbench_aggregate_core.py) | Core metrics library used by aggregators + baselines |
| [`make_skillsbench_splits.py`](scripts/make_skillsbench_splits.py) | Generate SkillsBench train/val/test split JSON |
| [`make_terminalbench_splits.py`](scripts/make_terminalbench_splits.py) | Generate Terminal-Bench category-stratified / holdout splits |
| [`compare_skillsbench_runs.py`](scripts/compare_skillsbench_runs.py) | Compare two SkillsBench JSONL runs (tokens, cost, API calls) |
| [`analyze_hot_pool_runs.py`](scripts/analyze_hot_pool_runs.py) | Hot skill pool telemetry + procedure proxies + core metrics |
| [`read_skillsbench_jsonl.py`](scripts/read_skillsbench_jsonl.py) | Load SkillsBench JSONL into Python |
| [`run_appworld_with_hermes.py`](scripts/run_appworld_with_hermes.py) | AppWorld ReAct loop with Hermes code generation; hot-pool / experiment-dir / resume parity with SkillsBench & ALFWorld |

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
  --model Qwen/Qwen3.6-27B \
  --skill-nudge-interval 10 \
  --memory-nudge-interval 10 \
  --hot-pool \
  --hot-pool-persist benchmark/skillsbench_hot_pool.json \
  --log-jsonl benchmark/hermes_skillsbench_runs.jsonl \
  --print-summary

# Single task — hot pool off (control)
python3 benchmark/scripts/run_skillsbench_with_hermes.py \
  --task adaptive-cruise-control \
  --model Qwen/Qwen3.6-27B \
  --skill-nudge-interval 10 \
  --memory-nudge-interval 10 \
  --no-hot-pool \
  --log-jsonl benchmark/hermes_skillsbench_runs.jsonl \
  --print-summary

# Batch (sorted task order; continues after errors)
python3 benchmark/scripts/run_skillsbench_with_hermes.py --all \
  --model Qwen/Qwen3.6-27B \
  --hot-pool \
  --hot-pool-persist benchmark/skillsbench_hot_pool.json \
  --log-jsonl benchmark/hermes_skillsbench_runs.jsonl

# Batch — hot pool off
python3 benchmark/scripts/run_skillsbench_with_hermes.py --all \
  --model Qwen/Qwen3.6-27B \
  --no-hot-pool \
  --log-jsonl benchmark/hermes_skillsbench_runs.jsonl

# Isolated experiment workspace (recommended for multi-run / A/B):
# copies selected tasks under DIR/skillsbench/tasks/ so agent outputs do not
# pollute the shared SkillsBench tree; defaults hot-pool JSON to DIR/hot_pool.json.
# Hermes keeps using ~/.hermes unless you pass --isolate-hermes-home
# (then repo skills/ is copied into DIR/hermes_home/skills — not ~/.hermes/skills;
# config/.env/SOUL symlink to ~/.hermes).
python3 benchmark/scripts/run_skillsbench_with_hermes.py --all \
  --experiment-dir benchmark/runs/exp_hot_train \
  --split-file benchmark/skillsbench_splits/stratified_v1.json \
  --split-part train \
  --model Qwen/Qwen3.6-27B \
  --hot-pool \
  --log-jsonl benchmark/runs/exp_hot_train/runs.jsonl \
  --print-summary

# Slice of tasks: [start, end) in sorted order
python3 benchmark/scripts/run_skillsbench_with_hermes.py --all \
  --start-task-index 0 --end-task-index 10 \
  --model Qwen/Qwen3.6-27B \
  --hot-pool \
  --hot-pool-persist benchmark/skillsbench_hot_pool.json \
  --log-jsonl benchmark/hermes_skillsbench_runs.jsonl
```

**Useful flags**

| Flag | Default | Notes |
|------|---------|--------|
| `--model` | from Hermes config | Model id, e.g. `Qwen/Qwen3.6-27B` |
| `--max-iterations` | 90 | Tool-calling cap per conversation |
| `--no-quiet` | off | More console output from Hermes |
| `--skip-context-files` | off | Skip AGENTS.md-style context injection |
| `--skip-memory` | off | Disable persistent memory |
| `--skill-nudge-interval` / `--memory-nudge-interval` | from config | Override nudge counters after agent init |
| `--hot-pool` / `--no-hot-pool` | follow config | Force hot skill pool on or off for this run (see below) |
| `--hot-pool-persist PATH` | off | Load/save hot pool across tasks; requires pool enabled; incompatible with `--no-hot-pool` |
| `--experiment-dir DIR` | off | Per-experiment **task** workspace (`DIR/skillsbench/tasks/`); does not isolate Hermes by default |
| `--reset-task-workspaces` | off | With `--experiment-dir`: refresh task copies / wipe prior agent outputs (also re-seeds isolated skills from repo `skills/`) |
| `--isolate-hermes-home` | off | Sandbox `HERMES_HOME=DIR/hermes_home`: **copy** repo `skills/` (not `~/.hermes/skills`); **symlink** `config.yaml` / `.env` / `SOUL.md` to real `~/.hermes` |
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
  --model Qwen/Qwen3.6-27B \
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

**Final evaluation** — after the conversation ends, host pytest runs against `tests/test_outputs.py` (container paths `/root`, `/app`, `/tests`, `/logs` remapped into the staged workspace). Logged as `evaluation` on each JSONL row:

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
  --model Qwen/Qwen3.6-27B \
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
  --model Qwen/Qwen3.6-27B \
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
  --model Qwen/Qwen3.6-27B

# Test split — inject key points from train pool file
python3 benchmark/scripts/run_skillsbench_with_hermes.py --all \
  --split-file benchmark/skillsbench_splits/stratified_v1.json \
  --split-part test \
  --hot-pool \
  --hot-pool-persist benchmark/runs/stratified_train_pool.json \
  --log-jsonl benchmark/runs/stratified_test.jsonl \
  --model Qwen/Qwen3.6-27B
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

Capacity eviction is **admission-time** (`skills.hot_pool.eviction_policy`): `llm` (default — side-channel judge on the running agent's LLM client when the pool overflows; falls back to `oldest`) or `oldest` (FIFO of extract on a persisted `global_turn` clock — one tick per `run_conversation` / SkillsBench task; deprecated as primary). Because retain = inject, the judge keeps a broadcast-worthy subset (transferable abstraction over context-local relevance); see [Hot skills](../README.md#hot-skills-ephemeral-key-point-pool). `max_entries` is both retain and inject size — the prompt dumps the whole pool. JSONL `hot_pool_telemetry.eviction` reports `capacity` drops.

**Outcome feedback** (`skills.hot_pool.outcome_feedback`, default **on**): after host evaluation (SkillsBench `--evaluate-after-run`), episode end (ALFWorld), or AppWorld `evaluate_task()`, a side-channel LLM attributes exposed tips using multi-dimensional metrics (success, reward, **iterations/steps**, tests). Utilities appear on eviction point payloads and in `hot_pool_telemetry.outcome_feedback`. Set `outcome_feedback: false` to A/B without attribution cost.

**Treatment** — pool enabled with persistence across a split or batch:

```bash
python3 benchmark/scripts/run_skillsbench_with_hermes.py --all \
  --hot-pool \
  --hot-pool-persist benchmark/skillsbench_hot_pool.json \
  --log-jsonl benchmark/runs_hot_pool_treatment.jsonl \
  --model Qwen/Qwen3.6-27B
```

**Control** — hot pool off for this run (ignores config and `--hot-pool-persist`):

```bash
python3 benchmark/scripts/run_skillsbench_with_hermes.py --all \
  --no-hot-pool \
  --model Qwen/Qwen3.6-27B \
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

JSONL rows record `hot_pool_enabled` (`true` / `false` / `null`) and `hot_pool_persist`. When the pool is active, `hot_pool_telemetry.inject` shows `point_count`, `skills_injected`, and `points_injected`. With `skip_if_in_history` (default on), pool skills already opened via `skill_view` are omitted from the inject block to avoid duplicating the full skill body; that is **not** “hot off.” Read `inject.skills_excluded_in_history` and `inject.points_excluded_in_history` for how many retained tips were skipped for that reason (`point_count==0` with nonzero exclusions still means the tips lived in the transcript via `skill_view`).

Env vars (set by the driver): `HERMES_HOT_POOL_ENABLED=0|1`, `HERMES_HOT_POOL_PERSIST=1`, `HERMES_HOT_POOL_PATH=/path/to/pool.json`.

### Clean task trees before A/B runs

Hermes runs **write into** `benchmark/skillsbench/tasks/<task-id>/` (solution files, `mass_report.json`, local verify scripts, etc.). Leftover artifacts make iteration counts unreliable — a later run may “verify existing output” in fewer steps while a dirty tree inflates or deflates comparisons.

**Preferred:** use `--experiment-dir DIR` so each experiment gets its own copy under `DIR/skillsbench/tasks/` (shared SkillsBench tree stays clean). Hermes still uses `~/.hermes` unless you pass `--isolate-hermes-home` (writable **copy** of repo `skills/` under `DIR/hermes_home/skills/` — not the live `~/.hermes/skills` tree; `config.yaml` / `.env` / `SOUL.md` stay linked to the real home).

```bash
python3 benchmark/scripts/run_skillsbench_with_hermes.py --all \
  --experiment-dir benchmark/runs/exp_control \
  --split-file benchmark/skillsbench_splits/stratified_v1.json \
  --split-part train \
  --no-hot-pool \
  --log-jsonl benchmark/runs/exp_control/runs.jsonl
```

Alternatively, `benchmark/skillsbench/` is a **nested git checkout**. Reset one task or the whole tree before a clean comparison:

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
  --model Qwen/Qwen3.6-27B \
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
  --model Qwen/Qwen3.6-27B \
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

## Terminal-Bench

Vendored Harbor task tree: [`terminal-bench/`](terminal-bench/).

**Official eval** uses Harbor: Hermes `AIAgent` stays on the host, `terminal` / file tools exec inside the trial sandbox (`/app`), then Harbor runs `tests/test.sh` (verifier image). Pass@k does not apply (one verifier score per trial). Default concurrency is 1 (local vLLM does not parallelize well).

There is no host-pytest path for Terminal-Bench. Do **not** set `terminal.backend: harbor` in `config.yaml`. `TERMINAL_ENV=harbor` is trial-scoped and only valid while a Harbor session is bound.

```bash
pip install -e ".[harbor]"   # plus Docker; or: pip install "harbor[modal]"

# Oracle sanity check (no LLM)
python3 benchmark/scripts/run_terminalbench_with_harbor.py \
    --task cad-model --oracle --env docker --dry-run

python3 benchmark/scripts/run_terminalbench_with_harbor.py --all \
    --split-file benchmark/terminalbench_splits/stratified_v1.json \
    --split-part train \
    --model Qwen/Qwen3.6-27B --env docker --n-concurrent 1 \
    --experiment-dir benchmark/runs/tb_harbor_train \
    --isolate-hermes-home --no-hot-pool \
    --max-iterations 60 \
    --log-jsonl benchmark/runs/tb_harbor_train/runs.jsonl \
    --print-summary --resume
```

`--resume` skips tasks that already have a finished (non-error) JSONL row **or** a Harbor trial `result.json` under `--jobs-dir` (defaults to `DIR/jobs`). Re-run the same command after an interrupt: finished Harbor trials are not re-queued even if JSONL was never flushed, and those artifacts are backfilled into `--log-jsonl`. In-progress trials (no `result.json` yet) start over. Ctrl-C on the driver also salvages finished trials into JSONL before exit.

`--isolate-hermes-home` requires `--experiment-dir` (same as SkillsBench): `HERMES_HOME` becomes `DIR/hermes_home` (writable copy of repo `skills/`, not `~/.hermes/skills`; `config.yaml` / `.env` / `SOUL.md` symlink to `~/.hermes`). `--no-hot-pool` / `--hot-pool` override `config.yaml` for the Harbor child. `--jobs-dir` defaults to `DIR/jobs` when `--experiment-dir` is set. Skip-context and skip-memory stay on.

Harbor CLI equivalent (PYTHONPATH must include the Hermes repo root):

```bash
harbor run -p benchmark/terminal-bench/tasks \
    --agent-import-path benchmark.harbor_adapter.hermes_agent:HermesHarborAgent \
    -m Qwen/Qwen3.6-27B --env docker --jobs-dir benchmark/runs/harbor_jobs \
    --include-task-name cad-model
```

Aggregate Harbor JSONL with the same SkillsBench aggregator (`skillsbench_task_id` is the task id field):

```bash
python3 benchmark/scripts/aggregate_skillsbench_runs.py \
  benchmark/runs/tb_harbor_train/runs.jsonl --print-summary
```

### Train / test splits

Terminal-Bench has no upstream train/test partition and typically no `difficulty` in `task.toml`. Hermes ships splits under [`terminalbench_splits/`](terminalbench_splits/):

| File | Protocol | Notes |
|------|----------|--------|
| [`stratified_v1.json`](terminalbench_splits/stratified_v1.json) | Category-stratified 75/25, seed 42 | Default train/test |
| [`category_holdout_v1.json`](terminalbench_splits/category_holdout_v1.json) | Hold out Hardware + Media + Security | Domain OOD test |

```bash
python3 benchmark/scripts/make_terminalbench_splits.py --write-defaults

python3 benchmark/scripts/make_terminalbench_splits.py \
  --protocol category_stratified --seed 42 \
  -o benchmark/terminalbench_splits/stratified_v1.json
```

---

## ALFWorld

ALFWorld is an **interactive text household** (TextWorld), not a SkillsBench-style file workspace. Hermes does **not** get a terminal sandbox; the driver owns the env loop and asks Hermes for one admissible command per step (same pattern as AppWorld). Success is TextWorld `won`.

**You do need a small adapter** (now in [`alfworld_adapter/`](alfworld_adapter/) + [`scripts/run_alfworld_with_hermes.py`](scripts/run_alfworld_with_hermes.py)). You do **not** need MaskRCNN, seq2seq checkpoints, or AI2-THOR for this driver — those are for embodied `AlfredThorEnv`. PDDL / `game.tw-pddl` files are enough.

```bash
pip install -e ".[alfworld]"
# optional, if you want `import alfworld` without the driver path hack:
pip install -e benchmark/alfworld

export ALFWORLD_DATA=/data/liangfeng/alfwordData   # this machine's download path

python3 benchmark/scripts/run_alfworld_with_hermes.py --list-tasks --split valid_unseen

# Smoke: one unseen game
python3 benchmark/scripts/run_alfworld_with_hermes.py \
  --split valid_unseen --limit 1 \
  --model Qwen/Qwen3.6-27B \
  --log-jsonl benchmark/runs/alfworld_smoke/runs.jsonl \
  --print-summary

# Official-style eval split (valid_unseen, 50 env steps)
python3 benchmark/scripts/run_alfworld_with_hermes.py --all --split valid_unseen \
  --model Qwen/Qwen3.6-27B --max-steps 50 \
  --experiment-dir benchmark/runs/alfworld_unseen \
  --log-jsonl benchmark/runs/alfworld_unseen/runs.jsonl \
  --resume --print-summary
```

`--task` ids look like `pick_and_place_simple-Mug-None-Desk-1/trial_T…`. Default is **no Hermes tools** (pure text policy). Pass `--tools` / `--hot-pool` if you want skills in the loop. With `--isolate-hermes-home`, skills are seeded from repo `skills/` (not `~/.hermes/skills`). JSONL uses `evaluation.task_success` so `aggregate_skillsbench_runs.py` still works.

If `ALFWORLD_DATA` is unset, the driver also looks at `/data/liangfeng/alfwordData` and `/data/liangfeng/alfworldData`.

---

## AppWorld

Install AppWorld from the vendored tree, then run the Hermes driver. Pass **`--model`** explicitly (OpenRouter-style id); omitting it only works when `~/.hermes/config.yaml` already resolves a default model.

Hot-pool / experiment isolation mirrors ALFWorld and SkillsBench: `--hot-pool` / `--no-hot-pool` / `--hot-pool-persist`, `--experiment-dir`, `--isolate-hermes-home`, `--resume`. `--experiment-name` remains AppWorld’s output namespace under `experiments/outputs/` (orthogonal to `--experiment-dir`).

**Hot-skill experiment paradigm** (official AppWorld splits): build the pool on **`train`** (89), evaluate transfer on **`test_normal`** (167); optionally replay **`train`** with the frozen pool for train-repeat. Use **`dev`** only for smoke tests. Optional harder transfer: **`test_challenge`**.

```bash
pip install -e benchmark/appworld
# Fix broken editable IPython if import fails: pip install --force-reinstall ipython

python3 benchmark/scripts/run_appworld_with_hermes.py --list-tasks --dataset train

# 1) Train — generate / grow hot pool
python3 benchmark/scripts/run_appworld_with_hermes.py \
  --dataset train --all \
  --model Qwen/Qwen3.6-27B \
  --experiment-dir benchmark/runs/appworld_hot_train \
  --isolate-hermes-home --hot-pool \
  --experiment-name hermes-train \
  --log-jsonl benchmark/runs/appworld_hot_train/runs.jsonl \
  --resume --print-summary

# 2) Unseen test — inject frozen train pool (do not rebuild on test_*)
python3 benchmark/scripts/run_appworld_with_hermes.py \
  --dataset test_normal --all \
  --model Qwen/Qwen3.6-27B \
  --experiment-dir benchmark/runs/appworld_hot_test \
  --isolate-hermes-home \
  --hot-pool --hot-pool-persist benchmark/runs/appworld_hot_train/hot_pool.json \
  --experiment-name hermes-test \
  --log-jsonl benchmark/runs/appworld_hot_test/runs.jsonl \
  --resume --print-summary

# 3) Optional train-repeat — same frozen pool on train again
python3 benchmark/scripts/run_appworld_with_hermes.py \
  --dataset train --all \
  --model Qwen/Qwen3.6-27B \
  --experiment-dir benchmark/runs/appworld_hot_train_repeat \
  --isolate-hermes-home \
  --hot-pool --hot-pool-persist benchmark/runs/appworld_hot_train/hot_pool.json \
  --experiment-name hermes-train-repeat \
  --log-jsonl benchmark/runs/appworld_hot_train_repeat/runs.jsonl \
  --resume --print-summary

# Control (no hot pool) on unseen test
python3 benchmark/scripts/run_appworld_with_hermes.py \
  --dataset test_normal --all \
  --model Qwen/Qwen3.6-27B \
  --experiment-dir benchmark/runs/appworld_no_hot_test \
  --isolate-hermes-home --no-hot-pool \
  --experiment-name hermes-no-hot-test \
  --log-jsonl benchmark/runs/appworld_no_hot_test/runs.jsonl \
  --resume --print-summary
```

**AppWorld data:** from `benchmark/appworld/`, run `appworld download data` once (sets up `data/` under `APPWORLD_ROOT`; default is the AppWorld repo root). See [`appworld/README.md`](appworld/README.md) for `APPWORLD_ROOT` and full setup.

**Hermes tools:** the driver exposes all configured Hermes toolsets by default (terminal, files, web search, skills, etc.). AppWorld API actions still run via ` ```python ` blocks executed in the AppWorld REPL. Use `--no-tools` to restore the original code-only ReAct mode. With tools enabled, each AppWorld step allows up to 90 Hermes tool-calling iterations by default (`--max-hermes-iterations` overrides).

**Hot pool:** same CLI as SkillsBench/ALFWorld. With `--experiment-dir` + `--hot-pool` and no `--hot-pool-persist`, the pool defaults to `DIR/hot_pool.json`. After evaluation, outcome feedback updates tip utilities (iterations = AppWorld steps when present). JSONL includes `hot_pool_enabled`, `hot_pool_persist`, optional `hot_pool_telemetry`, and `evaluation.task_success` (alias of AppWorld `success`) for shared aggregators.

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
  --model Qwen/Qwen3.6-27B \
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
  --model Qwen/Qwen3.6-27B \
  --no-evaluate
```

**Upstream CLI** (from `benchmark/appworld/` after `pip install -e .`):

```bash
cd benchmark/appworld
appworld evaluate hermes-agent dev    # experiment name ^   dataset ^
```

See [`appworld/README.md`](appworld/README.md) for environment setup and leaderboard packing.

---

## Output files

JSONL logs are usually written under `benchmark/`:

| Driver | Schema | Notes |
|--------|--------|--------|
| SkillsBench | `skillsbench.hermes_run.v1` | `run_conversation_result`, `hot_pool_enabled`, `hot_pool_persist`, optional `hot_pool_telemetry`, `background_review` (`actions`, `telemetry`) |
| Terminal-Bench | `terminalbench.hermes_run.v1` | `eval_mode: harbor`; verifier reward from Harbor `result.json` |
| AppWorld (run) | `appworld.hermes_run.v1` | `steps`, `hermes_stats`, `run_conversation_result`, `evaluation` (unless `--no-evaluate`), `hot_pool_enabled` / `hot_pool_persist`, optional `hot_pool_telemetry` |
| AppWorld (eval only) | `appworld.hermes_eval.v1` | `evaluation` (`success` + `task_success`); optional `run_hermes_stats` when `--run-log-jsonl` matches |
| AppWorld (eval skip) | `appworld.hermes_eval_skip.v1` | `skip_reason` when task was not run (`--evaluate-only`) |

Errors use `*.hermes_run_error.v1` schemas on failure lines.
