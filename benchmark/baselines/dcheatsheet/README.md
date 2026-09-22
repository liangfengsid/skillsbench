# Dynamic Cheatsheet baseline (vs TipsWarm)

[Dynamic Cheatsheet](https://github.com/suzgunmirac/dynamic-cheatsheet) (Suzgun et al., 2025, [arXiv:2504.07952](https://arxiv.org/abs/2504.07952)) is a **self-curated solution notebook** baseline for Hermes, not a skill-generation method. Compare it to TipsWarm on the same `AIAgent` backbone:

- **Same agent** (`run_agent.AIAgent`) — Hermes is the *generator*
- **Skill tools stay on** (`skill_view` / `skill_manage`)
- **Hot-skill pool off** (`--dc` forces `--no-hot-pool`)
- **MEMORY.md off** (`skip_memory=True`; DC still loads via `HERMES_DC_ENABLED`)
- **No extra DC tools** — the cheatsheet injects automatically in `<memory-context>`
- **Official curator prompts** (DC-Cu / DC-RS) from the paper repo; a side-channel LLM is the *curator*

Default `--dc-mode cu` is DC-Cu (generate, then curate). `--dc-mode rs` is DC-RS (retrieve similar episodes, curate, then generate). `--dc-mode curetr` injects the current sheet plus retrieved examples, then curates like CU.

Paper protocol (all three benchmarks):

- **Warmup:** `--dc --dc-freeze` (inject only; no curator writes)
- **Test** after `cp -r $expDir $expDir2`: `--dc --dc-sync-every 0` (one curator pass per episode)

`--dc-freeze` sets `HERMES_DC_READONLY=1` and **ignores** `--dc-sync-every`.

Plugin: [`plugins/memory/dcheatsheet/`](../../../plugins/memory/dcheatsheet/).
Canonical copy-workspace commands also live in the root [`README.md`](../../../README.md).

## Install

No extra pip extra for DC-Cu. The curator uses the same OpenAI-compatible endpoint as the Hermes agent.

`--dc-mode rs` / `curetr` optionally use MiniLM via `sentence-transformers`. If Hugging Face is blocked, prefetch with the A-Mem mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com   # optional
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

Without MiniLM, retrieval falls back to the most recent episodes.

## Protocol

Use `--experiment-dir $expDir --isolate-hermes-home`. `--experiment-dir` defaults `--dc-persist` to `$expDir/dcheatsheet`.

```
$expDir/dcheatsheet/cheatsheet.txt
$expDir/dcheatsheet/episodes.jsonl
```

`--dc` + `--hot-pool` and `--dc` + `--amem` are rejected.

`--dc-sync-every` (default **5**) flushes every N buffered turns; `0` = one curation per episode; `1` = legacy per-turn curator calls.

## SkillsBench

```bash
model=Qwen/Qwen3.6-27B
provider=openrouter

expDir=benchmark/runs/skillsbench_dc_train
python3 benchmark/scripts/run_skillsbench_with_hermes.py --all \
  --dc --dc-freeze --experiment-dir $expDir --isolate-hermes-home \
  --split-file benchmark/skillsbench_splits/stratified_v1.json --split-part train \
  --pass-k 1,5,10,30,60 --model $model --provider $provider \
  --skip-context-files --skip-memory --max-iterations 60 \
  --log-jsonl $expDir/runs.jsonl --resume --print-summary

expDir2=benchmark/runs/skillsbench_dc_train_test
cp -r $expDir $expDir2
rm -f $expDir2/runs.jsonl
expDir=$expDir2
python3 benchmark/scripts/run_skillsbench_with_hermes.py --all \
  --dc --dc-sync-every 0 \
  --experiment-dir $expDir --isolate-hermes-home \
  --split-file benchmark/skillsbench_splits/stratified_v1.json --split-part test \
  --pass-k 1,5,10,30,60 --model $model --provider $provider \
  --skip-context-files --skip-memory --max-iterations 60 \
  --log-jsonl $expDir/runs.jsonl --resume --print-summary
```

## ALFWorld

`--dc` enables `--tools` if you omit it.

```bash
export ALFWORLD_DATA=/path/to/alfworld/data
model=Qwen/Qwen3.6-27B
expDir=benchmark/runs/alfworld_dc_train
python3 benchmark/scripts/run_alfworld_with_hermes.py --all --split train --limit 50 \
  --model $model --dc --dc-freeze --tools --max-steps 50 \
  --experiment-dir $expDir --isolate-hermes-home \
  --skip-context-files --skip-memory --resume --print-summary

expDir2=benchmark/runs/alfworld_dc_train_unseen
cp -r $expDir $expDir2
rm -f $expDir2/runs.jsonl
expDir=$expDir2
python3 benchmark/scripts/run_alfworld_with_hermes.py --all --split valid_unseen \
  --model $model --dc --dc-sync-every 0 --tools --max-steps 50 \
  --experiment-dir $expDir --isolate-hermes-home \
  --skip-context-files --skip-memory --resume --print-summary
```

## AppWorld

Official splits: **`train`** then **`test_normal`**.

```bash
model=Qwen/Qwen3.6-27B
exp=appworld_dc_train
expDir=benchmark/runs/$exp
python3 benchmark/scripts/run_appworld_with_hermes.py \
  --dataset train --all --model $model --dc --dc-freeze \
  --experiment-dir $expDir --isolate-hermes-home \
  --skip-context-files --skip-memory \
  --experiment-name $exp \
  --log-jsonl $expDir/runs.jsonl --resume --print-summary

exp2=appworld_dc_train_unseen
expDir2=benchmark/runs/$exp2
cp -r $expDir $expDir2
exp=$exp2
expDir=$expDir2
rm -f $expDir/runs.jsonl
python3 benchmark/scripts/run_appworld_with_hermes.py \
  --dataset test_normal --all --model $model \
  --dc --dc-sync-every 0 \
  --experiment-dir $expDir --isolate-hermes-home \
  --skip-context-files --skip-memory \
  --experiment-name $exp \
  --log-jsonl $expDir/runs.jsonl --resume --print-summary
```

## JSONL

Rows include `dc_enabled`, `dc_persist`, `dc_mode`, and optional `dc_telemetry`
(`sheet_chars`, `n_episodes`, `n_curations`, `readonly`, `sync_every`,
`last_prefetch`). Aggregate with the same SkillsBench aggregator
(`evaluation.task_success`).
