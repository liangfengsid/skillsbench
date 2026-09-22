# A-Mem baseline (vs TipsWarm)

[A-Mem](https://github.com/WujiangXu/A-mem-sys) (Xu et al., 2025) is a **long-term experience memory** baseline for Hermes, not a skill-generation method. Compare it to TipsWarm on the same `AIAgent` backbone:

- **Same agent** (`run_agent.AIAgent`)
- **Skill tools stay on** (`skill_view` / `skill_manage`)
- **Hot-skill pool off** (`--amem` forces `--no-hot-pool`)
- **MEMORY.md off** (`skip_memory=True`; A-Mem still loads via `HERMES_AMEM_ENABLED`)
- **No extra A-Mem tools** — notes inject automatically in `<memory-context>`
- **Writes are batched** — default `--amem-sync-every 5` flushes every 5
  buffered turns. Use `0` for one note per episode/task; `1` for legacy
  per-turn writes (expensive: MiniLM embed + LLM evolution each step).

Paper **test** protocol: `--amem-sync-every 20` after copying the warmup workspace.
`--amem-freeze` is inject-only (no writes) and ignores `--amem-sync-every`.

Plugin: [`plugins/memory/amem/`](../../../plugins/memory/amem/).
Canonical copy-workspace commands also live in the root [`README.md`](../../../README.md).

## Install

```bash
source .venv/bin/activate   # or: source venv/bin/activate
pip install -e ".[amem]"
```

That installs [A-mem-sys](https://github.com/WujiangXu/A-mem-sys) plus `sentence-transformers` / `chromadb`. The **Hermes chat LLM** is already used for A-Mem note analysis. MiniLM is only the **local embedding** model Chroma uses for nearest-neighbor retrieval — prefetch it once before a long `--amem` run so the first task does not stall on Hugging Face.

### Prefetch MiniLM (`all-MiniLM-L6-v2`)

Chroma loads `sentence-transformers/all-MiniLM-L6-v2` on first `HermesAMemStore` init (~80MB). If `huggingface.co` is slow or blocked, point Hugging Face Hub at a mirror **before** both the prefetch and every later `--amem` command (same shell, or put it in `~/.bashrc`):

```bash
source .venv/bin/activate
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

Use `--experiment-dir $expDir --isolate-hermes-home`. `--experiment-dir` defaults `--amem-persist` to `$expDir/amem`. Warmup on train, **copy the whole workspace**, drop `runs.jsonl`, then evaluate with `--amem-sync-every 20`.

```
$expDir/amem/chroma/          # Chroma collection
$expDir/amem/amem_notes.json  # note snapshot (required for AppWorld subprocesses)
```

`--amem` + `--hot-pool` is rejected. `--amem-k` (default 5) is how many notes to inject per turn.

## SkillsBench

```bash
model=Qwen/Qwen3.6-27B
provider=openrouter

expDir=benchmark/runs/skillsbench_amem_train
python3 benchmark/scripts/run_skillsbench_with_hermes.py --all \
  --amem --experiment-dir $expDir --isolate-hermes-home \
  --split-file benchmark/skillsbench_splits/stratified_v1.json --split-part train \
  --pass-k 1,5,10,30,60 --model $model --provider $provider \
  --skip-context-files --skip-memory --max-iterations 60 \
  --log-jsonl $expDir/runs.jsonl --resume --print-summary

expDir2=benchmark/runs/skillsbench_amem_train_test
cp -r $expDir $expDir2
rm -f $expDir2/runs.jsonl
expDir=$expDir2
python3 benchmark/scripts/run_skillsbench_with_hermes.py --all \
  --amem --amem-sync-every 20 \
  --experiment-dir $expDir --isolate-hermes-home \
  --split-file benchmark/skillsbench_splits/stratified_v1.json --split-part test \
  --pass-k 1,5,10,30,60 --model $model --provider $provider \
  --skip-context-files --skip-memory --max-iterations 60 \
  --log-jsonl $expDir/runs.jsonl --resume --print-summary
```

## ALFWorld

`--amem` enables `--tools` if you omit it.

```bash
export ALFWORLD_DATA=/path/to/alfworld/data
model=Qwen/Qwen3.6-27B
expDir=benchmark/runs/alfworld_amem_train
python3 benchmark/scripts/run_alfworld_with_hermes.py --all --split train --limit 50 \
  --model $model --amem --tools --max-steps 50 \
  --experiment-dir $expDir --isolate-hermes-home \
  --skip-context-files --skip-memory --resume --print-summary

expDir2=benchmark/runs/alfworld_amem_train_unseen
cp -r $expDir $expDir2
rm -f $expDir2/runs.jsonl
expDir=$expDir2
python3 benchmark/scripts/run_alfworld_with_hermes.py --all --split valid_unseen \
  --model $model --amem --amem-sync-every 20 --tools --max-steps 50 \
  --experiment-dir $expDir --isolate-hermes-home \
  --skip-context-files --skip-memory --resume --print-summary
```

## AppWorld

Official splits: grow notes on **`train`**, transfer on **`test_normal`**.

```bash
model=Qwen/Qwen3.6-27B
exp=appworld_amem_train
expDir=benchmark/runs/$exp
python3 benchmark/scripts/run_appworld_with_hermes.py \
  --dataset train --all --model $model --amem \
  --experiment-dir $expDir --isolate-hermes-home \
  --skip-context-files --skip-memory \
  --experiment-name $exp \
  --log-jsonl $expDir/runs.jsonl --resume --print-summary

exp2=appworld_amem_train_unseen
expDir2=benchmark/runs/$exp2
cp -r $expDir $expDir2
exp=$exp2
expDir=$expDir2
rm -f $expDir/runs.jsonl
python3 benchmark/scripts/run_appworld_with_hermes.py \
  --dataset test_normal --all --model $model \
  --amem --amem-sync-every 20 \
  --experiment-dir $expDir --isolate-hermes-home \
  --skip-context-files --skip-memory \
  --experiment-name $exp \
  --log-jsonl $expDir/runs.jsonl --resume --print-summary
```

## JSONL

Rows include `amem_enabled`, `amem_persist`, and optional `amem_telemetry`
(`n_notes`, `n_syncs`, `readonly`, `last_prefetch`). Aggregate with the same
SkillsBench aggregator (`evaluation.task_success`).
