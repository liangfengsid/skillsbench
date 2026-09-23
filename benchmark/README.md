# Hermes benchmark drivers

Hermes ships evaluation harnesses under `benchmark/` for running **`AIAgent`** against public benchmarks and comparing runs. All **Hermes-specific drivers** live in [`scripts/`](scripts/). Vendored benchmark trees (`skillsbench/`, `alfworld/`, `appworld/`) keep their upstream docs.

**Paper reproduction (TipsWarm / HermesSkills / A-MEM / DC)** — isolated `$expDir`, copy-workspace test, and reported tables — is in the root [`README.md`](../README.md). This file is the **flag and option reference** plus extra experiments (single-task smokes, split generators, telemetry).

**Run commands from the Hermes repo root** unless noted otherwise.

**Default for any experiment you intend to keep:** `--experiment-dir $expDir --isolate-hermes-home`, and persist method state under `$expDir` (`hot_pool.json`, `amem/`, `dcheatsheet/`). Do not pass a live `~/.hermes` skills tree into an A/B run.

## Prerequisites

```bash
source .venv/bin/activate   # or: source venv/bin/activate
```

- **Hermes config:** `~/.hermes/config.yaml` and API keys in `~/.hermes/.env` (see [Configuration](https://hermes-agent.nousresearch.com/docs/user-guide/configuration)).
- **Import path:** drivers prepend `--hermes-root` (default: this repo) to `sys.path` and import `run_agent.AIAgent`.
- **SkillsBench / BenchFlow (optional):** `pip install -e ".[skillsbench]"` from repo root — see [`skillsbench/README.md`](skillsbench/README.md).
- **ALFWorld (optional):** `pip install -e ".[alfworld]"` plus PDDL/game files in `ALFWORLD_DATA`. Not included in `[all]`.
- **A-Mem baseline (optional):** `pip install -e ".[amem]"` — [A-Mem](https://github.com/WujiangXu/A-mem-sys) vs hot-skill on the Hermes drivers. See [`baselines/amem/README.md`](baselines/amem/README.md).
- **Dynamic Cheatsheet baseline:** [Suzgun et al. 2025](https://arxiv.org/abs/2504.07952) vs hot-skill on the same drivers (`--dc`). See [`baselines/dcheatsheet/README.md`](baselines/dcheatsheet/README.md).
- **AppWorld:** `pip install -e benchmark/appworld` — see [`appworld/README.md`](appworld/README.md).

## Layout

| Path | Role |
|------|------|
| [`scripts/`](scripts/) | Hermes batch drivers and analysis tools |
| [`alfworld_adapter/`](alfworld_adapter/) | ALFWorld TextWorld discovery + env wrapper for Hermes |
| [`baselines/`](baselines/) | Isolated third-party / paper baselines (A-Mem, Dynamic Cheatsheet) |
| [`skillsbench/`](skillsbench/) | SkillsBench tasks + BenchFlow (nested project) |
| [`alfworld/`](alfworld/) | Vendored ALFWorld (TextWorld / ALFRED) |
| [`appworld/`](appworld/) | AppWorld environment (vendored) |

### A-Mem baseline (SkillsBench, ALFWorld, AppWorld)

A-Mem is **not** a skill factory. It is a long-term episode-memory baseline vs TipsWarm: same Hermes agent, skill tools on, hot-skill off, notes injected via `<memory-context>`. Implementation: memory plugin [`plugins/memory/amem/`](../plugins/memory/amem/) + `--amem` on the three Hermes drivers.

```bash
pip install -e ".[amem]"
# Prefetch MiniLM first (export HF_ENDPOINT=https://hf-mirror.com if huggingface.co is blocked).
# Steps: baselines/amem/README.md  →  Prefetch MiniLM
```

Copy-paste train/eval (including `$expDir` and `--amem-sync-every 20` on test): [`baselines/amem/README.md`](baselines/amem/README.md). Root [`README.md`](../README.md) has the same pattern next to TipsWarm.

### Dynamic Cheatsheet baseline (SkillsBench, ALFWorld, AppWorld)

Dynamic Cheatsheet is **not** a skill factory. It is a self-curated solution notebook vs TipsWarm: same Hermes agent (the *generator*), skill tools on, hot-skill off, official curator prompts, cheatsheet injected via `<memory-context>`. Implementation: memory plugin [`plugins/memory/dcheatsheet/`](../plugins/memory/dcheatsheet/) + `--dc` on the three Hermes drivers.

Paper protocol: warmup `--dc --dc-freeze`; after copying `$expDir`, test `--dc --dc-sync-every 0`. `--dc` cannot combine with `--hot-pool` or `--amem`. Copy-paste: [`baselines/dcheatsheet/README.md`](baselines/dcheatsheet/README.md).

### Shared metrics (Hermes and baselines)

All drivers should log JSONL with `evaluation`, optional `pass_at_turn`, and
`run_conversation_result` (tokens / `api_calls`). Then:

```bash
# SkillsBench — budget k = Hermes API turns (user iterations)
python benchmark/scripts/aggregate_skillsbench_runs.py RUN.jsonl \
  --pass-k 1,5,10,70 --max-user-iterations 90 \
  -o RUN_summary.json --print-summary

# ALFWorld / AppWorld — same aggregator; budget k = env / execute() steps
python benchmark/scripts/aggregate_skillsbench_runs.py RUN.jsonl \
  --pass-k 1,5,10,30,50 --max-user-iterations 50 \
  --auc-max-k 50 \
  -o RUN_summary.json --print-summary
```

Reports **macro/micro Success@k** (success **within** budget *k*, not exact-at-*k*)
from `pass_at_turn` snapshots **and** in-budget final host eval (**default**;
opt into snapshots-only with `--pass-k-checkpoints-only`), **AUC** of that
curve over `k=1..--auc-max-k`, **final** rates within max iterations, and
**cost-to-succeed** mean±std among tasks that first succeed within
`--max-user-iterations` (or `max(--pass-k)` when that flag is omitted under
checkpoints-only).

Budget axis: SkillsBench uses Hermes API turns; ALFWorld / AppWorld use
environment / `execute()` steps (`evaluation.steps` / `steps_taken`). See
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
| [`run_alfworld_with_hermes.py`](scripts/run_alfworld_with_hermes.py) | ALFWorld TextWorld: Hermes picks admissible commands |
| [`evaluate_skillsbench_task.py`](scripts/evaluate_skillsbench_task.py) | Host pytest / `test.sh` verifier for SkillsBench |
| [`skillsbench_metrics.py`](scripts/skillsbench_metrics.py) | Build compact `metrics` blocks for JSONL rows |
| [`aggregate_skillsbench_runs.py`](scripts/aggregate_skillsbench_runs.py) | Shared Success@k / AUC / final / cost-to-succeed (SkillsBench, ALFWorld, AppWorld) |
| [`skillsbench_aggregate_core.py`](scripts/skillsbench_aggregate_core.py) | Core metrics library used by aggregators + baselines |
| [`make_skillsbench_splits.py`](scripts/make_skillsbench_splits.py) | Generate SkillsBench train/val/test split JSON |
| [`compare_skillsbench_runs.py`](scripts/compare_skillsbench_runs.py) | Compare two SkillsBench JSONL runs (tokens, cost, API calls) |
| [`analyze_hot_pool_runs.py`](scripts/analyze_hot_pool_runs.py) | Hot skill pool telemetry + procedure proxies + core metrics |
| [`read_skillsbench_jsonl.py`](scripts/read_skillsbench_jsonl.py) | Load SkillsBench JSONL into Python |
| [`run_appworld_with_hermes.py`](scripts/run_appworld_with_hermes.py) | AppWorld ReAct loop with Hermes code generation; hot-pool / A-Mem / Dynamic Cheatsheet / experiment-dir / resume parity with SkillsBench & ALFWorld |
| [`amem_baseline.py`](scripts/amem_baseline.py) | `--amem` CLI helpers shared by SkillsBench, ALFWorld, AppWorld |
| [`dc_baseline.py`](scripts/dc_baseline.py) | `--dc` CLI helpers shared by SkillsBench, ALFWorld, AppWorld |

Each script supports `--help`.

---

## SkillsBench

Task authoring and BenchFlow CLI: [`skillsbench/README.md`](skillsbench/README.md), [`skillsbench/AGENTS.md`](skillsbench/AGENTS.md).

### Run tasks with Hermes

Defaults: `--hermes-root` = repo root; `--skillsbench-root` = `benchmark/skillsbench/`; `--prompt-tasks-base` = absolute path to `benchmark/skillsbench/tasks/`.

**Canonical paper commands** (TipsWarm warmup → copy workspace → test → aggregate) are in the root [`README.md`](../README.md). The snippets below are extra options.

```bash
expDir=benchmark/runs/skillsbench_hot_train
model=Qwen/Qwen3.6-27B
provider=openrouter

# List task ids (no Hermes import)
python3 benchmark/scripts/run_skillsbench_with_hermes.py --list-tasks

# Isolated workspace is the default you should use for anything you keep.
python3 benchmark/scripts/run_skillsbench_with_hermes.py --all \
  --experiment-dir $expDir \
  --isolate-hermes-home \
  --split-file benchmark/skillsbench_splits/stratified_v1.json \
  --split-part train \
  --pass-k 1,5,10,30,60 \
  --model $model --provider $provider \
  --skip-context-files --skip-memory \
  --hot-pool --hot-pool-persist $expDir/hot_pool.json \
  --max-iterations 60 \
  --log-jsonl $expDir/runs.jsonl \
  --print-summary --resume

# Single task smoke (still isolated)
python3 benchmark/scripts/run_skillsbench_with_hermes.py \
  --task adaptive-cruise-control \
  --experiment-dir $expDir --isolate-hermes-home \
  --model $model --provider $provider \
  --skip-context-files --skip-memory \
  --hot-pool --hot-pool-persist $expDir/hot_pool.json \
  --log-jsonl $expDir/runs.jsonl --print-summary

# HermesSkills (no tip pool)
python3 benchmark/scripts/run_skillsbench_with_hermes.py --all \
  --experiment-dir $expDir --isolate-hermes-home \
  --split-file benchmark/skillsbench_splits/stratified_v1.json --split-part train \
  --model $model --no-hot-pool \
  --skip-context-files --skip-memory \
  --log-jsonl $expDir/runs.jsonl --print-summary --resume

# Slice of tasks: [start, end) in sorted order
python3 benchmark/scripts/run_skillsbench_with_hermes.py --all \
  --experiment-dir $expDir --isolate-hermes-home \
  --start-task-index 0 --end-task-index 10 \
  --model $model --hot-pool --hot-pool-persist $expDir/hot_pool.json \
  --log-jsonl $expDir/runs.jsonl
```

`--experiment-dir DIR` copies selected tasks under `DIR/skillsbench/tasks/` so agent outputs do not pollute the shared SkillsBench tree. `--isolate-hermes-home` sandboxes `HERMES_HOME=DIR/hermes_home`: **copy** repo `skills/` (not `~/.hermes/skills`); **symlink** `config.yaml` / `.env` / `SOUL.md` to real `~/.hermes`. With `--hot-pool` and no `--hot-pool-persist`, the pool defaults to `DIR/hot_pool.json`.

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
| `--amem` | off | A-Mem paper baseline: skill tools on, hot-skill off, notes via `<memory-context>` (incompatible with `--hot-pool` and `--dc`) |
| `--amem-persist DIR` | `DIR/amem` with `--experiment-dir` | A-Mem Chroma + `amem_notes.json` store |
| `--amem-k K` | 5 | Notes injected per turn |
| `--dc` | off | Dynamic Cheatsheet paper baseline: skill tools on, hot-skill off, cheatsheet via `<memory-context>` (incompatible with `--hot-pool` and `--amem`) |
| `--dc-persist DIR` | `DIR/dcheatsheet` with `--experiment-dir` | DC `cheatsheet.txt` + `episodes.jsonl` |
| `--dc-mode {cu,rs,curetr}` | `cu` | DC-Cu (generate then curate), DC-RS, or CU plus retrieved examples |
| `--dc-k K` | 3 | Retrieved prior episodes for `rs` / `curetr` |
| `--experiment-dir DIR` | off | Per-experiment **task** workspace (`DIR/skillsbench/tasks/`). Pair with `--isolate-hermes-home` (paper default). |
| `--reset-task-workspaces` | off | With `--experiment-dir`: refresh task copies / wipe prior agent outputs (also re-seeds isolated skills from repo `skills/`) |
| `--isolate-hermes-home` | off | **Pass this for any kept run.** Sandbox `HERMES_HOME=DIR/hermes_home`: **copy** repo `skills/` (not `~/.hermes/skills`); **symlink** `config.yaml` / `.env` / `SOUL.md` to real `~/.hermes` |
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
expDir=benchmark/runs/skillsbench_hot_train
python3 benchmark/scripts/run_skillsbench_with_hermes.py --all \
  --experiment-dir $expDir --isolate-hermes-home \
  --split-file benchmark/skillsbench_splits/stratified_v1.json \
  --split-part train \
  --pass-k 1,5,10,30,60 \
  --model Qwen/Qwen3.6-27B \
  --skip-context-files --skip-memory \
  --hot-pool --hot-pool-persist $expDir/hot_pool.json \
  --log-jsonl $expDir/runs.jsonl \
  --print-summary --resume

# Aggregate the same JSONL later (or combine multiple run files)
python3 benchmark/scripts/aggregate_skillsbench_runs.py \
  $expDir/runs.jsonl \
  --pass-k 1,5,10,30,60 --max-user-iterations 60 \
  --split-part train \
  --print-summary \
  -o $expDir/summary.json
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

# Aggregate across tasks: macro/micro pass@k (in-budget finals included by default)
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

- `pass_k.<turn>.macro_task_pass_rate` — fraction of tasks with success within budget *k* (any successful `pass_at_turn` ≤ *k*, or in-budget final by default; use `--pass-k-checkpoints-only` for snapshots only)
- `pass_k.<turn>.micro_test_pass_rate` — Σ passed test cases / Σ total at the terminal in-budget observation
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

Run a partition with the Hermes driver (copy the train workspace before the test split — see root [`README.md`](../README.md)):

```bash
expDir=benchmark/runs/skillsbench_hot_train
python3 benchmark/scripts/run_skillsbench_with_hermes.py --all \
  --experiment-dir $expDir --isolate-hermes-home \
  --split-file benchmark/skillsbench_splits/stratified_v1.json \
  --split-part train \
  --hot-pool --hot-pool-persist $expDir/hot_pool.json \
  --skip-context-files --skip-memory \
  --log-jsonl $expDir/runs.jsonl \
  --model Qwen/Qwen3.6-27B --resume

expDir2=benchmark/runs/skillsbench_hot_train_test
cp -r $expDir $expDir2
rm -f $expDir2/runs.jsonl
expDir=$expDir2
python3 benchmark/scripts/run_skillsbench_with_hermes.py --all \
  --experiment-dir $expDir --isolate-hermes-home \
  --split-file benchmark/skillsbench_splits/stratified_v1.json \
  --split-part test \
  --hot-pool --hot-pool-persist $expDir/hot_pool.json \
  --skip-context-files --skip-memory \
  --log-jsonl $expDir/runs.jsonl \
  --model Qwen/Qwen3.6-27B --resume
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
| `--hot-pool-inject-retrieve` / `--no-hot-pool-inject-retrieve` | Scope matching at inject (`inject_retrieve`). Default follows config (on). `--no-…` injects without query/scope retrieve ranking |
| `--hot-pool-outcome-feedback` / `--no-hot-pool-outcome-feedback` | Contribution attribution after host eval (`outcome_feedback`). `--no-…` skips the side-channel judge |
| `--hot-pool-inject-filter-utilities` / `--no-hot-pool-inject-filter-utilities` | Utility screen at inject (`inject_filter_utilities`). `--no-…` does not omit/rank by stored utilities. Pair with `--no-hot-pool-outcome-feedback` to A/B without credit assignment |
| `--hot-pool-max-entries N` | Store cap (`max_entries`). Default follows config (12). `N >= 0` |
| `--hot-pool-inject-k N` | Per-turn inject quota (`inject_k`). Default follows config (4). `0` injects every tip that passes the skill gate |
| `--amem` / `--amem-persist` / `--amem-k` | A-Mem baseline (see [baselines/amem](baselines/amem/README.md)); cannot combine with `--hot-pool` or `--dc` |
| `--dc` / `--dc-persist` / `--dc-mode` / `--dc-k` | Dynamic Cheatsheet baseline (see [baselines/dcheatsheet](baselines/dcheatsheet/README.md)); cannot combine with `--hot-pool` or `--amem` |

Pool curation is **admit-time** (`skills.hot_pool.eviction_policy`): `llm` (default — side-channel reconcile judge on each material admit/sync and when over `max_entries`; falls back to `oldest`) or `oldest` (FIFO only when over cap). Mechanical extract is followed by junk filters (`junk_filter`, `exclude_skills_from_pool`), a ritual filter (`ritual_filter`: episode closers / short-answer procedures stay out of the transfer pool), and a lexical domain gate (`admit_domain_gate`: AppWorld/ALFWorld omit distinctive off-domain skills; SkillsBench fails open except framework identities). Extract redacts structural identifiers; transfer vs episode-local is a side-channel prompt judgment. Inject selects up to `inject_k` tips by retrieve overlap with the frozen episode query, omits off-domain skills, ritual tips, and strongly harmful utilities, and ranks the rest. See [Hot skills](../README.md#hot-skills-ephemeral-key-point-pool). `max_entries` is the store cap; `inject_k` is the inject cap. JSONL `hot_pool_telemetry.eviction` reports `capacity` drops and `reconcile_ran`.

**Outcome feedback** (`skills.hot_pool.outcome_feedback`, default **on**): after host evaluation (SkillsBench `--evaluate-after-run`), episode end (ALFWorld), or AppWorld `evaluate_task()`, a side-channel LLM attributes exposed tips using multi-dimensional metrics (success, reward, **iterations/steps**, tests). Empty or unparseable judge JSON does **not** persist utilities (`skipped_reason` stays `unparseable_or_empty` / `no_complete_fn`). `outcome_feedback_heuristic` is off by default — cloned win/loss is not credit assignment. Results land in `hot_pool_telemetry.outcome_feedback` (`skipped_reason`, `attribution_source`, plus `judge_finish_reason` / `judge_raw_preview` / `judge_stripped_preview` when the side-channel ran). The judge call disables thinking, uses `outcome_judge_max_tokens` (default 2048), and receives a compressed tool/env action log (`outcome_judge_log`, no observations). SkillsBench CLI: `--no-hot-pool-outcome-feedback` (and `--no-hot-pool-inject-filter-utilities` to also skip the inject utility screen). Config: `outcome_feedback: false`.

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

Env vars (set by the driver): `HERMES_HOT_POOL_ENABLED=0|1`, `HERMES_HOT_POOL_PERSIST=1`, `HERMES_HOT_POOL_PATH=/path/to/pool.json`, and optionally `HERMES_HOT_POOL_INJECT_RETRIEVE`, `HERMES_HOT_POOL_OUTCOME_FEEDBACK`, `HERMES_HOT_POOL_INJECT_FILTER_UTILITIES` (`0`/`1`), `HERMES_HOT_POOL_MAX_ENTRIES`, and `HERMES_HOT_POOL_INJECT_K` (non-negative integers; `INJECT_K=0` means no inject cap).

A/B without scope matching or credit assignment (pool still injects):

```bash
python3 benchmark/scripts/run_skillsbench_with_hermes.py --all \
  --hot-pool --hot-pool-persist benchmark/runs/exp_hot/hot_pool.json \
  --no-hot-pool-inject-retrieve \
  --no-hot-pool-outcome-feedback --no-hot-pool-inject-filter-utilities \
  --log-jsonl benchmark/runs/exp_hot/runs.jsonl
```

### Clean task trees before A/B runs

Hermes runs **write into** `benchmark/skillsbench/tasks/<task-id>/` (solution files, `mass_report.json`, local verify scripts, etc.). Leftover artifacts make iteration counts unreliable — a later run may “verify existing output” in fewer steps while a dirty tree inflates or deflates comparisons.

**Preferred:** use `--experiment-dir $expDir --isolate-hermes-home` so each experiment gets its own copy under `$expDir/skillsbench/tasks/` (shared SkillsBench tree stays clean). Also reset the nested git between runs:

```bash
cd benchmark/skillsbench
git restore .
git clean -fdn
git clean -fd
cd ../..
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

## ALFWorld

ALFWorld is an **interactive text household** (TextWorld), not a SkillsBench-style file workspace. Hermes does **not** get a terminal sandbox; the driver owns the env loop and asks Hermes for one admissible command per step (same pattern as AppWorld). Success is TextWorld `won`.

**You do need a small adapter** (now in [`alfworld_adapter/`](alfworld_adapter/) + [`scripts/run_alfworld_with_hermes.py`](scripts/run_alfworld_with_hermes.py)). You do **not** need MaskRCNN, seq2seq checkpoints, or AI2-THOR for this driver — those are for embodied `AlfredThorEnv`. PDDL / `game.tw-pddl` files are enough.

```bash
pip install -e ".[alfworld]"
# optional, if you want `import alfworld` without the driver path hack:
pip install -e benchmark/alfworld

export ALFWORLD_DATA=/path/to/alfworld/data
expDir=benchmark/runs/alfworld_hot_train
model=Qwen/Qwen3.6-27B

python3 benchmark/scripts/run_alfworld_with_hermes.py --list-tasks --split valid_unseen

# Smoke: one unseen game
python3 benchmark/scripts/run_alfworld_with_hermes.py \
  --split valid_unseen --limit 1 \
  --experiment-dir $expDir --isolate-hermes-home \
  --model $model \
  --log-jsonl $expDir/runs.jsonl \
  --print-summary

# Paper-style warmup (train, 50 games). Full copy-to-unseen protocol: root README.md
python3 benchmark/scripts/run_alfworld_with_hermes.py --all \
  --experiment-dir $expDir --isolate-hermes-home \
  --split train --limit 50 \
  --model $model \
  --skip-context-files --skip-memory \
  --tools --hot-pool --hot-pool-persist $expDir/hot_pool.json \
  --max-steps 50 \
  --log-jsonl $expDir/runs.jsonl \
  --resume --print-summary

# Re-aggregate an existing ALFWorld JSONL
python3 benchmark/scripts/aggregate_skillsbench_runs.py \
  $expDir/runs.jsonl \
  --pass-k 1,5,10,30,50 --max-user-iterations 50 --auc-max-k 50 \
  -o $expDir/summary.json --print-summary
```

`--task` ids look like `pick_and_place_simple-Mug-None-Desk-1/trial_T…`. Default is **no Hermes tools** (pure text policy). Pass `--tools` / `--hot-pool` if you want skills in the loop. `--amem` and `--dc` are paper memory baselines (each forces `--no-hot-pool` and enables `--tools` so skill tools stay on); see [`baselines/amem/README.md`](baselines/amem/README.md) and [`baselines/dcheatsheet/README.md`](baselines/dcheatsheet/README.md). With `--isolate-hermes-home`, skills are seeded from repo `skills/` (not `~/.hermes/skills`). JSONL uses `evaluation.task_success` so `aggregate_skillsbench_runs.py` still works. Per-task lines use `--print-summary`; batch Success@k / AUC use `--print-batch-summary` (on by default; `--no-print-batch-summary` to silence). Override budgets with `--pass-k` / `--auc-max-k` / `--summary-output`.

Set `ALFWORLD_DATA` (or pass `--alfworld-data`) to the PDDL/game-file root. If that is unset, the driver also checks `~/.cache/alfworld`.

---

## AppWorld

Install AppWorld from the vendored tree, then run the Hermes driver. Pass **`--model`** explicitly (OpenRouter-style id); omitting it only works when `~/.hermes/config.yaml` already resolves a default model.

Hot-pool / experiment isolation mirrors ALFWorld and SkillsBench: `--hot-pool` / `--no-hot-pool` / `--hot-pool-persist`, `--amem` / `--amem-persist`, `--dc` / `--dc-persist`, `--experiment-dir`, `--isolate-hermes-home`, `--resume`. `--experiment-name` remains AppWorld’s output namespace under `experiments/outputs/` (orthogonal to `--experiment-dir`).

**Hot-skill experiment paradigm** (official AppWorld splits): build the pool on **`train`** (89), evaluate transfer on **`test_normal`** (167); optionally replay **`train`** with the frozen pool for train-repeat. Use **`dev`** only for smoke tests. Optional harder transfer: **`test_challenge`**.

```bash
pip install -e benchmark/appworld
# Fix broken editable IPython if import fails: pip install --force-reinstall ipython

exp=appworld_hot_train
expDir=benchmark/runs/$exp
model=Qwen/Qwen3.6-27B

python3 benchmark/scripts/run_appworld_with_hermes.py --list-tasks --dataset train

# 1) Train — generate / grow hot pool. Copy-to-test_normal: root README.md
python3 benchmark/scripts/run_appworld_with_hermes.py \
  --dataset train --all \
  --model $model \
  --experiment-dir $expDir \
  --isolate-hermes-home --skip-context-files --skip-memory \
  --hot-pool --hot-pool-persist $expDir/hot_pool.json \
  --experiment-name $exp \
  --log-jsonl $expDir/runs.jsonl \
  --resume --print-summary

# Re-aggregate an existing AppWorld JSONL
python3 benchmark/scripts/aggregate_skillsbench_runs.py \
  $expDir/runs.jsonl \
  --pass-k 1,10,20,40 --max-user-iterations 40 --auc-max-k 40 \
  -o $expDir/summary.json --print-summary
```

**AppWorld data:** from `benchmark/appworld/`, run `appworld download data` once (sets up `data/` under `APPWORLD_ROOT`; default is the AppWorld repo root). See [`appworld/README.md`](appworld/README.md) for `APPWORLD_ROOT` and full setup.

**Hermes tools:** the driver exposes all configured Hermes toolsets by default (terminal, files, web search, skills, etc.). AppWorld API actions still run via ` ```python ` blocks executed in the AppWorld REPL. Use `--no-tools` to restore the original code-only ReAct mode. With tools enabled, each AppWorld step allows up to 90 Hermes tool-calling iterations by default (`--max-hermes-iterations` overrides).

**Hot pool:** same CLI as SkillsBench/ALFWorld. With `--experiment-dir` + `--hot-pool` and no `--hot-pool-persist`, the pool defaults to `DIR/hot_pool.json`. After evaluation, outcome feedback updates tip utilities (iterations = AppWorld steps when present). JSONL includes `hot_pool_enabled`, `hot_pool_persist`, optional `hot_pool_telemetry`, and `evaluation.task_success` (alias of AppWorld `success`) for shared aggregators. Per-task lines use `--print-summary`; batch Success@k / AUC use `--print-batch-summary` (on by default). Override budgets with `--pass-k` / `--auc-max-k` / `--summary-output`.

**A-Mem:** `--amem` on the same driver (skill tools stay on; incompatible with `--hot-pool` and `--dc`). Persist dir defaults to `DIR/amem`. See [`baselines/amem/README.md`](baselines/amem/README.md).

**Dynamic Cheatsheet:** `--dc` on the same driver (skill tools stay on; incompatible with `--hot-pool` and `--amem`). Persist dir defaults to `DIR/dcheatsheet`. See [`baselines/dcheatsheet/README.md`](baselines/dcheatsheet/README.md).

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

## Reported run files

Headline numbers live in the [root README](../README.md#5-reported-results). This section is the file index. Git tracks `runs/*/summary.json`, `runs/*/hot_pool.json`, and `runs/*.csv`. JSONL logs and `hermes_home/` stay untracked. Naming rules: [`runs/README.md`](runs/README.md).

### SkillsBench (Qwen, test)

| Method | Seed 1 | Seed 2 | Seed 3 |
|--------|--------|--------|--------|
| A-MEM | [summary](runs/skillsbench_amem_train_test/summary.json) | — | — |
| DC | [summary](runs/skillsbench_dc_train_test1/summary.json) | [summary](runs/skillsbench_dc_train_test2/summary.json) | [summary](runs/skillsbench_dc_train_test3/summary.json) |
| HermesSkills (**NONE**) | [summary](runs/skillsbench_no_hot_train_test1/summary.json) | [summary](runs/skillsbench_no_hot_train_test2/summary.json) | [summary](runs/skillsbench_no_hot_train_test3/summary.json) |
| TipsWarm (**ALL**) | [summary](runs/skillsbench_hot_train_test1/summary.json) · [pool](runs/skillsbench_hot_train_test1/hot_pool.json) | [summary](runs/skillsbench_hot_train_test2/summary.json) · [pool](runs/skillsbench_hot_train_test2/hot_pool.json) | [summary](runs/skillsbench_hot_train_test3/summary.json) · [pool](runs/skillsbench_hot_train_test3/hot_pool.json) |

CSV: [skillsbench_main.csv](runs/skillsbench_main.csv).

### SkillsBench, DeepSeek vs Qwen

Headline numbers: [root README](../README.md#skillsbench-deepseek-vs-qwen). CSV: [skillsbench_model.csv](runs/skillsbench_model.csv). Qwen seeds are the HermesSkills and TipsWarm rows in the table above.

| Method | Seed 1 | Seed 2 | Seed 3 |
|--------|--------|--------|--------|
| HermesSkills-DS | [summary](runs/skillsbench_no_hot_train_ds_test1/summary.json) | [summary](runs/skillsbench_no_hot_train_ds_test2/summary.json) | [summary](runs/skillsbench_no_hot_train_ds_test3/summary.json) |
| TipsWarm-DS | [summary](runs/skillsbench_hot_train_ds_test1/summary.json) · [pool](runs/skillsbench_hot_train_ds_test1/hot_pool.json) | [summary](runs/skillsbench_hot_train_ds_test2/summary.json) · [pool](runs/skillsbench_hot_train_ds_test2/hot_pool.json) | [summary](runs/skillsbench_hot_train_ds_test3/summary.json) · [pool](runs/skillsbench_hot_train_ds_test3/hot_pool.json) |
| HermesSkills-Qwen | [summary](runs/skillsbench_no_hot_train_test1/summary.json) | [summary](runs/skillsbench_no_hot_train_test2/summary.json) | [summary](runs/skillsbench_no_hot_train_test3/summary.json) |
| TipsWarm-Qwen | [summary](runs/skillsbench_hot_train_test1/summary.json) · [pool](runs/skillsbench_hot_train_test1/hot_pool.json) | [summary](runs/skillsbench_hot_train_test2/summary.json) · [pool](runs/skillsbench_hot_train_test2/hot_pool.json) | [summary](runs/skillsbench_hot_train_test3/summary.json) · [pool](runs/skillsbench_hot_train_test3/hot_pool.json) |

### SkillsBench ablations (Qwen, test)

CSV: [skillsbench_ablation.csv](runs/skillsbench_ablation.csv). **NONE** and **ALL** are the HermesSkills and TipsWarm rows above.

| Label | Setting | Seed 1 | Seed 2 | Seed 3 |
|-------|---------|--------|--------|--------|
| **NW** | Inherits the HermesSkills warmup skill bundle | [summary](runs/skillsbench_no_hot_train_hottest1/summary.json) · [pool](runs/skillsbench_no_hot_train_hottest1/hot_pool.json) | [summary](runs/skillsbench_no_hot_train_hottest2/summary.json) · [pool](runs/skillsbench_no_hot_train_hottest2/hot_pool.json) | [summary](runs/skillsbench_no_hot_train_hottest3/summary.json) · [pool](runs/skillsbench_no_hot_train_hottest3/hot_pool.json) |
| **NS** | No scope matching (`--no-hot-pool-inject-retrieve`) | [summary](runs/skillsbench_hot_train_ns_test1/summary.json) · [pool](runs/skillsbench_hot_train_ns_test1/hot_pool.json) | [summary](runs/skillsbench_hot_train_ns_test2/summary.json) · [pool](runs/skillsbench_hot_train_ns_test2/hot_pool.json) | [summary](runs/skillsbench_hot_train_ns_test3/summary.json) · [pool](runs/skillsbench_hot_train_ns_test3/hot_pool.json) |
| **NU** | No utility attribution or screening (`--no-hot-pool-outcome-feedback --no-hot-pool-inject-filter-utilities`) | [summary](runs/skillsbench_hot_train_nu_test1/summary.json) · [pool](runs/skillsbench_hot_train_nu_test1/hot_pool.json) | [summary](runs/skillsbench_hot_train_nu_test2/summary.json) · [pool](runs/skillsbench_hot_train_nu_test2/hot_pool.json) | [summary](runs/skillsbench_hot_train_nu_test3/summary.json) · [pool](runs/skillsbench_hot_train_nu_test3/hot_pool.json) |
| **BP** | Store 16, inject 8 (`--hot-pool-max-entries 16 --hot-pool-inject-k 8`) | [summary](runs/skillsbench_hot_train_bp_test1/summary.json) · [pool](runs/skillsbench_hot_train_bp_test1/hot_pool.json) | [summary](runs/skillsbench_hot_train_bp_test2/summary.json) · [pool](runs/skillsbench_hot_train_bp_test2/hot_pool.json) | [summary](runs/skillsbench_hot_train_bp_test3/summary.json) · [pool](runs/skillsbench_hot_train_bp_test3/hot_pool.json) |

### AppWorld (`test_normal`)

| Method | Seed 1 | Seed 2 | Seed 3 |
|--------|--------|--------|--------|
| A-MEM | [summary](runs/appworld_amem_unseen/summary.json) | — | — |
| DC | [summary](runs/appworld_dc_test/summary.json) | [summary](runs/appworld_dc_test2/summary.json) | [summary](runs/appworld_dc_test3/summary.json) |
| HermesSkills | [summary](runs/appworld_no_hot_train_unseen1/summary.json) | [summary](runs/appworld_no_hot_train_unseen2/summary.json) | [summary](runs/appworld_no_hot_train_unseen3/summary.json) |
| TipsWarm | [summary](runs/appworld_hot_train_unseen/summary.json) · [pool](runs/appworld_hot_train_unseen/hot_pool.json) | [summary](runs/appworld_hot_train_unseen2/summary.json) · [pool](runs/appworld_hot_train_unseen2/hot_pool.json) | [summary](runs/appworld_hot_train_unseen3/summary.json) · [pool](runs/appworld_hot_train_unseen3/hot_pool.json) |

CSV: [appworld.csv](runs/appworld.csv).

### ALFWorld (`valid_unseen`)

| Method | Seed 1 | Seed 2 | Seed 3 |
|--------|--------|--------|--------|
| A-MEM | [summary](runs/alfworld_amem_train_unseen/summary.json) | — | — |
| DC | [summary](runs/alfworld_dc_train_unseen/summary.json) | [summary](runs/alfworld_dc_train_unseen2/summary.json) | [summary](runs/alfworld_dc_train_unseen3/summary.json) |
| HermesSkills | [summary](runs/alfworld_no_hot_train_unseen/summary.json) | [summary](runs/alfworld_no_hot_train_unseen2/summary.json) | [summary](runs/alfworld_no_hot_train_unseen3/summary.json) |
| TipsWarm | [summary](runs/alfworld_hot_train_unseen/summary.json) · [pool](runs/alfworld_hot_train_unseen/hot_pool.json) | [summary](runs/alfworld_hot_train_unseen2/summary.json) · [pool](runs/alfworld_hot_train_unseen2/hot_pool.json) | [summary](runs/alfworld_hot_train_unseen3/summary.json) · [pool](runs/alfworld_hot_train_unseen3/hot_pool.json) |

CSV: [alfworld.csv](runs/alfworld.csv).

## Output files

JSONL logs are usually written under `benchmark/`:

| Driver | Schema | Notes |
|--------|--------|--------|
| SkillsBench | `skillsbench.hermes_run.v1` | `run_conversation_result`, `hot_pool_enabled`, `hot_pool_persist`, optional `hot_pool_telemetry` / `amem_telemetry` / `dc_telemetry`, `background_review` (`actions`, `telemetry`) |
| ALFWorld | `alfworld.hermes_run.v1` | `evaluation.task_success` (`won`), `hot_pool_*`, optional `amem_telemetry` / `dc_telemetry` |
| AppWorld (run) | `appworld.hermes_run.v1` | `steps`, `hermes_stats`, `run_conversation_result`, `evaluation` (unless `--no-evaluate`), `hot_pool_*`, optional `amem_telemetry` / `dc_telemetry` |
| AppWorld (eval only) | `appworld.hermes_eval.v1` | `evaluation` (`success` + `task_success`); optional `run_hermes_stats` when `--run-log-jsonl` matches |
| AppWorld (eval skip) | `appworld.hermes_eval_skip.v1` | `skip_reason` when task was not run (`--evaluate-only`) |

Errors use `*.hermes_run_error.v1` schemas on failure lines.
