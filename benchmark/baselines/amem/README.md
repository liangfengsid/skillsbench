# A-Mem baseline (vs hot-skill)

[A-Mem](https://github.com/WujiangXu/A-mem-sys) (Xu et al., 2025) is a **long-term experience memory** baseline for Hermes, not a skill-generation method. Compare it to the hot-skill pool on the same `AIAgent` backbone:

- **Same agent** (`run_agent.AIAgent`)
- **Skill tools stay on** (`skill_view` / `skill_manage`)
- **Hot-skill pool off** (`--amem` forces `--no-hot-pool`)
- **MEMORY.md off** (`skip_memory=True`; A-Mem still loads via `HERMES_AMEM_ENABLED`)
- **No extra A-Mem tools** — notes inject automatically in `<memory-context>`
- **Writes are batched** — default `--amem-sync-every 5` flushes every 5
  buffered turns (~3 notes on a typical ~15-step AppWorld task). Use `0` for
  one note per episode/task; `1` for legacy per-turn writes (expensive: MiniLM
  embed + LLM evolution each step).

CoEvoSkills remains a *skill factory* baseline (different research question). A-Mem is the right row for “does structured episode memory match hot key-points?”

Plugin: [`plugins/memory/amem/`](../../../plugins/memory/amem/).

## Install

```bash
source .venv/bin/activate   # or: source venv/bin/activate
pip install -e ".[amem]"
```

That installs [A-mem-sys](https://github.com/WujiangXu/A-mem-sys) plus `sentence-transformers` / `chromadb`. The **Hermes chat LLM** is already used for A-Mem note analysis. MiniLM is only the **local embedding** model Chroma uses for nearest-neighbor retrieval — prefetch it once before a long `--amem` run so the first task does not stall on Hugging Face.

### Prefetch MiniLM (`all-MiniLM-L6-v2`)

Chroma loads `sentence-transformers/all-MiniLM-L6-v2` on first `HermesAMemStore` init (~80MB). If `huggingface.co` is slow or blocked, point Hugging Face Hub at a mirror **before** both the prefetch and every later `--amem` command (same shell, or put it in `~/.bashrc`):

```bash
source .venv/bin/activate   # or: source venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com

python -c "
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('all-MiniLM-L6-v2')
print('ok', m.get_sentence_embedding_dimension(), 'dims')
"
```

A successful prefetch prints `ok 384 dims` and caches weights under `~/.cache/huggingface/hub/` (or `$HF_HOME`). You do **not** pass MiniLM as `--model`; leave `HERMES_AMEM_EMBED_MODEL` unset unless you want a different encoder.

If the snippet fails with `No module named sentence_transformers`, the `[amem]` extra did not install — rerun `pip install -e ".[amem]"` in the same venv.

Keep `HF_ENDPOINT` exported when you start SkillsBench / ALFWorld / AppWorld `--amem` jobs. The first store open will reuse the cache and should not hit the network.

## Protocol

Train (or grow notes) on the training split, then evaluate transfer with the **same persist dir**:

```
DIR/amem/chroma/          # Chroma collection
DIR/amem/amem_notes.json  # note snapshot (required for AppWorld subprocesses)
```

`--experiment-dir DIR` defaults `--amem-persist` to `DIR/amem`. For a held-out eval, point `--amem-persist` at the **train** store (same pattern as `--hot-pool-persist PATH`).

Use `--isolate-hermes-home` so skills are a copy of repo `skills/`, not `~/.hermes/skills`.

`--amem` + `--hot-pool` is rejected. `--amem-k` (default 5) is how many notes to inject per turn.
`--amem-sync-every` (default **5**) controls write cadence: `N` = flush every N
buffered turns, `0` = one note per episode, `1` = legacy per-turn.

**Held-out eval must use `--amem-freeze`** (sets `HERMES_AMEM_READONLY=1`):
prefetch/retrieve only — no note writes, no embed/evolve. Point
`--amem-persist` at the train snapshot (or a copied store). Without freeze,
eval continues growing the store and pays write latency.

## SkillsBench

```bash
# Train — grow A-Mem notes
python3 benchmark/scripts/run_skillsbench_with_hermes.py --all \
  --amem --experiment-dir benchmark/runs/exp_amem_train \
  --isolate-hermes-home \
  --split-file benchmark/skillsbench_splits/stratified_v1.json --split-part train \
  --log-jsonl benchmark/runs/exp_amem_train/runs.jsonl \
  --resume --print-summary

# Test — frozen notes from train
python3 benchmark/scripts/run_skillsbench_with_hermes.py --all \
  --amem --amem-freeze --amem-persist benchmark/runs/exp_amem_train/amem \
  --experiment-dir benchmark/runs/exp_amem_test \
  --isolate-hermes-home \
  --split-file benchmark/skillsbench_splits/stratified_v1.json --split-part test \
  --log-jsonl benchmark/runs/exp_amem_test/runs.jsonl \
  --resume --print-summary
```

Hot-skill counterpart: same commands with `--hot-pool` / `--hot-pool-persist` instead of `--amem`.

## ALFWorld

A-Mem comparison uses **`--tools`** (skill tools). `--amem` enables `--tools` automatically if you omit it.

```bash
export ALFWORLD_DATA=/data/liangfeng/alfwordData

# Train
python3 benchmark/scripts/run_alfworld_with_hermes.py --all --split train \
  --model Qwen/Qwen3.6-27B --amem --tools --max-steps 50 \
  --experiment-dir benchmark/runs/alfworld_amem_train \
  --isolate-hermes-home --resume --print-summary

# Unseen eval (frozen train notes)
python3 benchmark/scripts/run_alfworld_with_hermes.py --all --split valid_unseen \
  --model Qwen/Qwen3.6-27B --amem --amem-freeze --tools --max-steps 50 \
  --amem-persist benchmark/runs/alfworld_amem_train/amem \
  --experiment-dir benchmark/runs/alfworld_amem_unseen \
  --isolate-hermes-home --resume --print-summary
```

## AppWorld

Official splits: grow notes on **`train`**, transfer on **`test_normal`**. Each `--all` task runs in a subprocess; the JSON snapshot under `--amem-persist` is what carries notes across workers.

```bash
# Train
python3 benchmark/scripts/run_appworld_with_hermes.py \
  --dataset train --all --model Qwen/Qwen3.6-27B --amem \
  --experiment-dir benchmark/runs/appworld_amem_train \
  --isolate-hermes-home --experiment-name hermes-amem-train \
  --log-jsonl benchmark/runs/appworld_amem_train/runs.jsonl \
  --resume --print-summary

# test_normal — frozen train store
python3 benchmark/scripts/run_appworld_with_hermes.py \
  --dataset test_normal --all --model Qwen/Qwen3.6-27B --amem --amem-freeze \
  --amem-persist benchmark/runs/appworld_amem_train/amem \
  --experiment-dir benchmark/runs/appworld_amem_test \
  --isolate-hermes-home --experiment-name hermes-amem-test \
  --log-jsonl benchmark/runs/appworld_amem_test/runs.jsonl \
  --resume --print-summary
```

## JSONL

Rows include `amem_enabled`, `amem_persist`, and optional `amem_telemetry`
(`n_notes`, `n_syncs`, `readonly`, `last_prefetch`). Aggregate with the same
SkillsBench aggregator (`evaluation.task_success`).
