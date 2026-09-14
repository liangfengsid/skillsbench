# Dynamic Cheatsheet baseline (vs hot-skill)

[Dynamic Cheatsheet](https://github.com/suzgunmirac/dynamic-cheatsheet) (Suzgun et al., 2025, [arXiv:2504.07952](https://arxiv.org/abs/2504.07952)) is a **self-curated solution notebook** baseline for Hermes, not a skill-generation method. Compare it to the hot-skill pool on the same `AIAgent` backbone:

- **Same agent** (`run_agent.AIAgent`) — Hermes is the *generator*
- **Skill tools stay on** (`skill_view` / `skill_manage`)
- **Hot-skill pool off** (`--dc` forces `--no-hot-pool`)
- **MEMORY.md off** (`skip_memory=True`; DC still loads via `HERMES_DC_ENABLED`)
- **No extra DC tools** — the cheatsheet injects automatically in `<memory-context>`
- **Official curator prompts** (DC-Cu / DC-RS) from the paper repo; a side-channel LLM is the *curator*

Default `--dc-mode cu` is DC-Cu (generate, then curate). `--dc-mode rs` is DC-RS (retrieve similar episodes, curate, then generate). `--dc-mode curetr` injects the current sheet plus retrieved examples, then curates like CU.

Plugin: [`plugins/memory/dcheatsheet/`](../../../plugins/memory/dcheatsheet/).

## Install

No extra pip extra for DC-Cu. The curator uses the same OpenAI-compatible endpoint as the Hermes agent.

`--dc-mode rs` / `curetr` optionally use MiniLM via `sentence-transformers`. If Hugging Face is blocked, prefetch with the A-Mem mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com   # optional
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

Without MiniLM, retrieval falls back to the most recent episodes.

## Protocol

Train (grow the cheatsheet) on the training split, then evaluate transfer with the **same persist dir**:

```
DIR/dcheatsheet/cheatsheet.txt
DIR/dcheatsheet/episodes.jsonl
```

`--experiment-dir DIR` defaults `--dc-persist` to `DIR/dcheatsheet`. For a held-out eval, point `--dc-persist` at the **train** store (same pattern as `--hot-pool-persist PATH` / `--amem-persist`).

Use `--isolate-hermes-home` so skills are a copy of repo `skills/`, not `~/.hermes/skills`.

`--dc` + `--hot-pool` and `--dc` + `--amem` are rejected.

Curator writes are batched like A-Mem: `--dc-sync-every` (default **5**)
flushes every N buffered turns; `0` = one curation per episode; `1` = legacy
per-turn curator calls.

**Held-out eval must use `--dc-freeze`** (sets `HERMES_DC_READONLY=1`):
inject the train cheatsheet only — no curator writes. Point `--dc-persist`
at the train snapshot (or a copy). Without freeze, eval keeps curating and
pays the curator LLM tax.

## SkillsBench

```bash
# Train — grow the cheatsheet
python3 benchmark/scripts/run_skillsbench_with_hermes.py --all \
  --dc --experiment-dir benchmark/runs/exp_dc_train \
  --isolate-hermes-home \
  --split-file benchmark/skillsbench_splits/stratified_v1.json --split-part train \
  --log-jsonl benchmark/runs/exp_dc_train/runs.jsonl \
  --resume --print-summary

# Test — frozen cheatsheet from train
python3 benchmark/scripts/run_skillsbench_with_hermes.py --all \
  --dc --dc-freeze --dc-persist benchmark/runs/exp_dc_train/dcheatsheet \
  --experiment-dir benchmark/runs/exp_dc_test \
  --isolate-hermes-home \
  --split-file benchmark/skillsbench_splits/stratified_v1.json --split-part test \
  --log-jsonl benchmark/runs/exp_dc_test/runs.jsonl \
  --resume --print-summary
```

Hot-skill counterpart: same commands with `--hot-pool` / `--hot-pool-persist` instead of `--dc`.

## ALFWorld

DC comparison uses **`--tools`** (skill tools). `--dc` enables `--tools` automatically if you omit it.

```bash
export ALFWORLD_DATA=/data/liangfeng/alfwordData

# Train
python3 benchmark/scripts/run_alfworld_with_hermes.py --all --split train \
  --model Qwen/Qwen3.6-27B --dc --tools --max-steps 50 \
  --experiment-dir benchmark/runs/alfworld_dc_train \
  --isolate-hermes-home --resume --print-summary

# Unseen eval (frozen train cheatsheet)
python3 benchmark/scripts/run_alfworld_with_hermes.py --all --split valid_unseen \
  --model Qwen/Qwen3.6-27B --dc --dc-freeze --tools --max-steps 50 \
  --dc-persist benchmark/runs/alfworld_dc_train/dcheatsheet \
  --experiment-dir benchmark/runs/alfworld_dc_unseen \
  --isolate-hermes-home --resume --print-summary
```

## AppWorld

Official splits: grow the cheatsheet on **`train`**, transfer on **`test_normal`**. Each `--all` task runs in a subprocess; `cheatsheet.txt` + `episodes.jsonl` under `--dc-persist` carry state across workers.

```bash
# Train
python3 benchmark/scripts/run_appworld_with_hermes.py \
  --dataset train --all --model Qwen/Qwen3.6-27B --dc \
  --experiment-dir benchmark/runs/appworld_dc_train \
  --isolate-hermes-home --experiment-name hermes-dc-train \
  --log-jsonl benchmark/runs/appworld_dc_train/runs.jsonl \
  --resume --print-summary

# test_normal — frozen train store
python3 benchmark/scripts/run_appworld_with_hermes.py \
  --dataset test_normal --all --model Qwen/Qwen3.6-27B --dc --dc-freeze \
  --dc-persist benchmark/runs/appworld_dc_train/dcheatsheet \
  --experiment-dir benchmark/runs/appworld_dc_test \
  --isolate-hermes-home --experiment-name hermes-dc-test \
  --log-jsonl benchmark/runs/appworld_dc_test/runs.jsonl \
  --resume --print-summary
```

## JSONL

Rows include `dc_enabled`, `dc_persist`, `dc_mode`, and optional `dc_telemetry`
(`sheet_chars`, `n_episodes`, `n_curations`, `readonly`, `sync_every`,
`last_prefetch`). Aggregate with the same SkillsBench aggregator
(`evaluation.task_success`).
