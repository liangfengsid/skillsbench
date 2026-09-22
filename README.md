# TipsWarm

**TipsWarm** is a transfer-oriented memory mechanism on top of [Hermes Agent](README_HERMES.md): a small pool of short, reusable tips (guardrails / pitfalls) is extracted from skills, stored across tasks, and injected ephemerally each turn. This repository is a Hermes Agent checkout plus the benchmark drivers, baselines, and reported runs used in the TipsWarm experiments.

This README is the reproduction guide. Hermes Agent itself (CLI, gateway, skills, config) is documented in **[`README_HERMES.md`](README_HERMES.md)** and at the [upstream docs](https://hermes-agent.nousresearch.com/docs/).

| Method in tables | What it is | Driver flags |
|------------------|------------|----------------|
| **TipsWarm** (ours) | Hot-tip pool (`hot_pool.json`) | `--hot-pool --hot-pool-persist $expDir/hot_pool.json` |
| **HermesSkills** | Same Hermes agent, no tip pool | `--no-hot-pool` |
| **A-MEM** | [A-Mem](https://github.com/WujiangXu/A-mem-sys) notes + Chroma | `--amem` (test: `--amem-sync-every 20`) |
| **DC** | [Dynamic Cheatsheet](https://arxiv.org/abs/2504.07952) (DC-Cu) | `--dc` (warmup: `--dc-freeze`; test: `--dc-sync-every 0`) |

`--hot-pool`, `--amem`, and `--dc` are mutually exclusive.

**Default experiment style:** isolated workspace home (`--experiment-dir` + `--isolate-hermes-home`) and a shell variable `$expDir` so warmup, test, and aggregation reuse the same command with only the path changed.

```
1. Hermes environment
2. Benchmark data (SkillsBench, AppWorld, ALFWorld)
3. Baseline extras (A-MEM, DC)
4. Run experiments (full commands below)
5. Reported results (CSVs + indexed summary.json / hot_pool.json)
```

---

## 1. Hermes Agent environment

TipsWarm runs through Hermes `AIAgent`. From the repo root:

```bash
# Python 3.12+ is required for SkillsBench / BenchFlow.
./setup-hermes.sh          # uv venv, install, optional wizard
# or:
uv venv .venv --python 3.12
source .venv/bin/activate  # or: source venv/bin/activate
uv pip install -e ".[all,dev]"
```

Then configure a provider and API keys (Hermes reads `~/.hermes/config.yaml` and `~/.hermes/.env`):

```bash
source .venv/bin/activate
hermes setup               # or: hermes model
```

Isolated experiment homes **copy** repo `skills/` into `$expDir/hermes_home/skills` and **symlink** `config.yaml`, `.env`, and `SOUL.md` to the real `~/.hermes`. You still need a working Hermes profile on the host.

Full CLI / config / skills documentation: [`README_HERMES.md`](README_HERMES.md). TipsWarm internals: [`agent/hot_skills.py`](agent/hot_skills.py). Extra driver flags: [`benchmark/README.md`](benchmark/README.md).

---

## 2. Prepare benchmarks

Install extras from the **repo root** (same venv). Task trees and official setup notes live in the vendored dirs.

```bash
source .venv/bin/activate
pip install -e ".[skillsbench]"          # BenchFlow
pip install -e ".[alfworld]"             # ALFWorld TextWorld driver
pip install -e benchmark/appworld        # AppWorld package
```

| Benchmark | What to prepare | Details |
|-----------|-----------------|---------|
| **SkillsBench** | Nested tree at `benchmark/skillsbench/` (already vendored). Optional: `cd benchmark/skillsbench && pip install -e .` | [`benchmark/skillsbench/README.md`](benchmark/skillsbench/README.md), splits in [`benchmark/skillsbench_splits/`](benchmark/skillsbench_splits/) |
| **AppWorld** | Download official data once | [`benchmark/appworld/README.md`](benchmark/appworld/README.md) |
| **ALFWorld** | PDDL / `game.tw-pddl` files; set `ALFWORLD_DATA` | [`benchmark/alfworld/README.md`](benchmark/alfworld/README.md) |

AppWorld data:

```bash
cd benchmark/appworld
appworld download data
cd ../..
```

ALFWorld data (path is machine-specific):

```bash
export ALFWORLD_DATA=/path/to/alfworld/data
# optional: alfworld-download   # see benchmark/alfworld/README.md
```

Train/test partitions used in the paper:

- SkillsBench: [`benchmark/skillsbench_splits/stratified_v1.json`](benchmark/skillsbench_splits/stratified_v1.json) — 65 train / 23 test
- AppWorld: official `train` → `test_normal`
- ALFWorld: `train` (limit 50) → `valid_unseen`

---

## 3. Configure other baseline methods

**A-MEM** needs the `[amem]` extra and a one-time MiniLM prefetch (Chroma embeddings). If `huggingface.co` is blocked, set a mirror first. See [`benchmark/baselines/amem/README.md`](benchmark/baselines/amem/README.md).

```bash
pip install -e ".[amem]"
export HF_ENDPOINT=https://hf-mirror.com   # optional
python -c "from sentence_transformers import SentenceTransformer as S; m=S('all-MiniLM-L6-v2'); print('ok', m.get_sentence_embedding_dimension())"
```

**Dynamic Cheatsheet (DC-Cu)** uses the same OpenAI-compatible endpoint as Hermes. No extra pip package. See [`benchmark/baselines/dcheatsheet/README.md`](benchmark/baselines/dcheatsheet/README.md).

Reported protocol:

- DC **warmup:** `--dc --dc-freeze` (inject only; no curator writes)
- DC **test** (after copying the workspace): `--dc --dc-sync-every 0` (one curator pass per episode)
- A-MEM **test:** `--amem --amem-sync-every 20`

`--dc-freeze` / `--amem-freeze` ignore `--*-sync-every` (readonly inject).

---

## 4. Run experiments

Always from the **repo root**, with the venv activated. Shared conventions:

1. Set `expDir=benchmark/runs/<name>` and pass `--experiment-dir $expDir --isolate-hermes-home`.
2. Persist method state under that directory (`hot_pool.json`, `amem/`, `dcheatsheet/`).
3. Warmup on the train split; **copy the whole workspace** to a test directory; delete only `runs.jsonl`; re-point `expDir`; run the test split.
4. Between SkillsBench experiments, reset the nested task git so leftover agent files cannot leak:

```bash
cd benchmark/skillsbench
git restore .
git clean -fdn    # dry run
git clean -fd
cd ../..
```

`--experiment-dir` already copies tasks into `$expDir/skillsbench/tasks/`. The `git clean` is extra insurance for the shared tree at `benchmark/skillsbench/tasks/`.

Set these once per shell:

```bash
source .venv/bin/activate
model=Qwen/Qwen3.6-27B
provider=openrouter          # or your Hermes provider id (e.g. a local vLLM)
```

OpenRouter-style ids are sometimes lowercase (`qwen/qwen3.6-27b`); use whatever `hermes model` resolved.

More flags (pass@k variants, single-task smoke tests, telemetry): [`benchmark/README.md`](benchmark/README.md).

### 4.1 SkillsBench — TipsWarm

**Warmup (train)**

```bash
expDir=benchmark/runs/skillsbench_hot_train
python3 benchmark/scripts/run_skillsbench_with_hermes.py --all \
  --experiment-dir $expDir \
  --isolate-hermes-home \
  --split-file benchmark/skillsbench_splits/stratified_v1.json \
  --split-part train \
  --pass-k 1,5,10,30,60 \
  --model $model --provider $provider \
  --skip-context-files \
  --skip-memory \
  --hot-pool \
  --hot-pool-persist $expDir/hot_pool.json \
  --max-iterations 60 \
  --log-jsonl $expDir/runs.jsonl \
  --print-summary \
  --resume
```

**Test**

```bash
expDir2=benchmark/runs/skillsbench_hot_train_test
cp -r $expDir $expDir2
rm -f $expDir2/runs.jsonl
expDir=$expDir2

python3 benchmark/scripts/run_skillsbench_with_hermes.py --all \
  --experiment-dir $expDir \
  --isolate-hermes-home \
  --split-file benchmark/skillsbench_splits/stratified_v1.json \
  --split-part test \
  --pass-k 1,5,10,30,60 \
  --model $model --provider $provider \
  --skip-context-files \
  --skip-memory \
  --hot-pool \
  --hot-pool-persist $expDir/hot_pool.json \
  --max-iterations 60 \
  --log-jsonl $expDir/runs.jsonl \
  --print-summary \
  --resume
```

**Summarize**

```bash
python3 benchmark/scripts/aggregate_skillsbench_runs.py \
  $expDir/runs.jsonl \
  --pass-k 1,5,10,30,60 \
  --max-user-iterations 60 \
  --split-part test \
  --print-summary \
  -o $expDir/summary.json
```

HermesSkills: same commands with `--no-hot-pool` and without `--hot-pool-persist`.  
A-MEM: replace the hot-pool flags with `--amem` on warmup and `--amem --amem-sync-every 20` on test.  
DC: replace them with `--dc --dc-freeze` on warmup and `--dc --dc-sync-every 0` on test.

### 4.2 AppWorld — TipsWarm

**Warmup (train)**

```bash
exp=appworld_hot_train
expDir=benchmark/runs/$exp
python3 benchmark/scripts/run_appworld_with_hermes.py \
  --dataset train --all \
  --model $model \
  --experiment-dir $expDir \
  --isolate-hermes-home --skip-context-files --skip-memory \
  --hot-pool --hot-pool-persist $expDir/hot_pool.json \
  --experiment-name $exp \
  --log-jsonl $expDir/runs.jsonl \
  --resume --print-summary
```

**Test (`test_normal`)**

```bash
exp2=appworld_hot_train_unseen
expDir2=benchmark/runs/$exp2
cp -r $expDir $expDir2
exp=$exp2
expDir=$expDir2
rm -f $expDir/runs.jsonl

python3 benchmark/scripts/run_appworld_with_hermes.py \
  --dataset test_normal --all \
  --model $model \
  --experiment-dir $expDir \
  --isolate-hermes-home --skip-context-files --skip-memory \
  --hot-pool --hot-pool-persist $expDir/hot_pool.json \
  --experiment-name $exp \
  --log-jsonl $expDir/runs.jsonl \
  --resume --print-summary
```

**Summarize** (budget *k* = AppWorld `execute()` steps)

```bash
python3 benchmark/scripts/aggregate_skillsbench_runs.py \
  $expDir/runs.jsonl \
  --pass-k 1,10,20,40 --max-user-iterations 40 --auc-max-k 40 \
  -o $expDir/summary.json --print-summary
```

Swap in `--no-hot-pool`, `--amem` / `--amem-sync-every 20`, or `--dc --dc-freeze` / `--dc --dc-sync-every 0` as in §4.1.

### 4.3 ALFWorld — TipsWarm

ALFWorld needs `--tools` so skill tools (and therefore TipsWarm) are in the loop.

**Warmup**

```bash
export ALFWORLD_DATA=/path/to/alfworld/data
expDir=benchmark/runs/alfworld_hot_train
python3 benchmark/scripts/run_alfworld_with_hermes.py --all \
  --experiment-dir $expDir \
  --isolate-hermes-home \
  --split train --limit 50 \
  --model $model \
  --skip-context-files --skip-memory \
  --tools --hot-pool \
  --hot-pool-persist $expDir/hot_pool.json \
  --max-steps 50 \
  --log-jsonl $expDir/runs.jsonl \
  --print-summary --resume
```

**Test (`valid_unseen`)**

```bash
expDir2=benchmark/runs/alfworld_hot_train_unseen
cp -r $expDir $expDir2
rm -f $expDir2/runs.jsonl
expDir=$expDir2
python3 benchmark/scripts/run_alfworld_with_hermes.py --all \
  --experiment-dir $expDir \
  --isolate-hermes-home \
  --split valid_unseen \
  --model $model \
  --skip-context-files --skip-memory \
  --tools --hot-pool \
  --hot-pool-persist $expDir/hot_pool.json \
  --max-steps 50 \
  --log-jsonl $expDir/runs.jsonl \
  --print-summary --resume
```

The driver prints Success@k / AUC / final / cost at the end of a batch and writes `$expDir/summary.json`. To re-aggregate later:

```bash
python3 benchmark/scripts/aggregate_skillsbench_runs.py \
  $expDir/runs.jsonl \
  --pass-k 1,5,10,30,50 --max-user-iterations 50 --auc-max-k 50 \
  -o $expDir/summary.json --print-summary
```

`--amem` and `--dc` enable `--tools` automatically if you omit it. Use the same freeze / `sync-every` flags as the other benchmarks.

---

## 5. Reported results

Git indexes **`benchmark/runs/*/summary.json`**, **`benchmark/runs/*/hot_pool.json`** (TipsWarm), and **`benchmark/runs/*.csv`**. JSONL logs, `hermes_home/`, and task copies stay untracked.

Headline numbers below are copied from those CSVs (one row per seed; empty method names in the CSV are the same method as the previous named row). Token / iteration / time columns are **cost-to-succeed** among tasks that finish within the budget (SkillsBench, AppWorld) unless the CSV header says otherwise.

### SkillsBench (Qwen3.6-27B)

Source: [`benchmark/runs/skillsbench_main.csv`](benchmark/runs/skillsbench_main.csv). Pass@k is Hermes API turns; *k* ∈ {1,10,30,60}.

| Method | Seed | macro AUC | micro AUC | P@1 | P@10 | P@30 | P@60 | tokens | iters | time (s) |
|--------|------|-----------|-----------|-----|------|------|------|--------|-------|----------|
| A-MEM | 1 | 0.5232 | 0.6304 | 0.0000 | 0.2174 | 0.6087 | 0.7826 | 1.65e6±1.69e6 | 20.9±13.4 | 8044±9769 |
| DC | 1 | 0.5268 | 0.6331 | 0.1304 | 0.2609 | 0.6957 | 0.6957 | 6.10e5±4.01e5 | 15.6±9.0 | 1169±1081 |
| DC | 2 | 0.4587 | 0.5861 | 0.0435 | 0.2609 | 0.6087 | 0.6957 | 9.85e5±7.84e5 | 21.4±15.1 | 1656±1077 |
| DC | 3 | 0.2659 | 0.4324 | 0.0435 | 0.0435 | 0.3043 | 0.6087 | 1.58e6±9.44e5 | 34.8±18.1 | 2256±1828 |
| HermesSkills | 1 | 0.4803 | 0.5289 | 0.0455 | 0.2273 | 0.6818 | 0.8182 | 1.10e6±6.18e5 | 25.8±13.8 | 1977±1077 |
| HermesSkills | 2 | 0.4167 | 0.5379 | 0.0435 | 0.2174 | 0.6087 | 0.8261 | 1.33e6±8.66e5 | 30.7±18.0 | 2549±1456 |
| HermesSkills | 3 | 0.4703 | 0.6665 | 0.0435 | 0.2174 | 0.6957 | 0.8261 | 1.18e6±7.54e5 | 26.8±15.4 | 1699±1026 |
| TipsWarm | 1 | 0.5101 | 0.5785 | 0.0435 | 0.2174 | 0.7391 | 0.8261 | 1.24e6±8.02e5 | 25.3±14.3 | 1283±876 |
| TipsWarm | 2 | 0.5312 | 0.6057 | 0.0435 | 0.3913 | 0.6957 | 0.8261 | 1.38e6±1.36e6 | 22.4±18.3 | 1452±951 |
| TipsWarm | 3 | 0.5565 | 0.6083 | 0.0870 | 0.3043 | 0.7391 | 0.8696 | 1.33e6±1.09e6 | 24.2±14.9 | 1574±1085 |

DeepSeek vs Qwen (SkillsBench only): [`benchmark/runs/skillsbench_model.csv`](benchmark/runs/skillsbench_model.csv).

### AppWorld (`test_normal`)

Source: [`benchmark/runs/appworld.csv`](benchmark/runs/appworld.csv). Success@k is `execute()` steps; *k* ∈ {1,10,20,40}.

| Method | Seed | macro AUC | micro AUC | S@1 | S@10 | S@20 | S@40 |
|--------|------|-----------|-----------|-----|------|------|------|
| A-MEM | 1 | 0.3757 | 0.5071 | 0.0000 | 0.2275 | 0.4551 | 0.5629 |
| DC | 1 | 0.3804 | 0.4668 | 0.0000 | 0.2036 | 0.4611 | 0.5808 |
| DC | 2 | 0.3350 | 0.4563 | 0.0000 | 0.2024 | 0.3988 | 0.5000 |
| DC | 3 | 0.3516 | 0.4793 | 0.0000 | 0.2083 | 0.4167 | 0.5238 |
| HermesSkills | 1 | 0.4390 | 0.5076 | 0.0000 | 0.1905 | 0.5298 | 0.6726 |
| HermesSkills | 2 | 0.3978 | 0.4687 | 0.0000 | 0.1726 | 0.4583 | 0.6250 |
| HermesSkills | 3 | 0.4277 | 0.4718 | 0.0000 | 0.1845 | 0.4821 | 0.6845 |
| TipsWarm | 1 | 0.4616 | 0.4854 | 0.0000 | 0.2202 | 0.5476 | 0.7202 |
| TipsWarm | 2 | 0.4280 | 0.4849 | 0.0000 | 0.1845 | 0.4821 | 0.6786 |
| TipsWarm | 3 | 0.4413 | 0.4866 | 0.0000 | 0.1497 | 0.5030 | 0.7305 |

### ALFWorld (`valid_unseen`)

Source: [`benchmark/runs/alfworld.csv`](benchmark/runs/alfworld.csv). Success@k is env steps; *k* ∈ {1,10,30,50}.

| Method | Seed | macro AUC | S@1 | S@10 | S@30 | S@50 |
|--------|------|-----------|-----|------|------|------|
| A-MEM | 1 | 0.6507 | 0.0000 | 0.4403 | 0.8284 | 0.8731 |
| DC | 1 | 0.7407 | 0.0000 | 0.5672 | 0.9030 | 0.9701 |
| DC | 2 | 0.7345 | 0.0000 | 0.5896 | 0.9179 | 0.9552 |
| DC | 3 | 0.7378 | 0.0000 | 0.5597 | 0.9104 | 0.9701 |
| HermesSkills | 1 | 0.7061 | 0.0000 | 0.5000 | 0.8731 | 0.9403 |
| HermesSkills | 2 | 0.7284 | 0.0000 | 0.5373 | 0.8881 | 0.9851 |
| HermesSkills | 3 | 0.7090 | 0.0000 | 0.5075 | 0.8657 | 0.9776 |
| TipsWarm | 1 | 0.7181 | 0.0000 | 0.5000 | 0.8955 | 0.9701 |
| TipsWarm | 2 | 0.7133 | 0.0000 | 0.5149 | 0.8955 | 0.9851 |
| TipsWarm | 3 | 0.6901 | 0.0000 | 0.4552 | 0.8657 | 0.9776 |

### Indexed run directories

Each cell is the git-tracked `summary.json`. TipsWarm cells also have `hot_pool.json` in the same directory.

**SkillsBench (test)**

| Method | Seed 1 | Seed 2 | Seed 3 |
|--------|--------|--------|--------|
| A-MEM | [summary](benchmark/runs/skillsbench_amem_train0824_test/summary.json) | — | — |
| DC | [summary](benchmark/runs/skillsbench_dc_train0906_test1/summary.json) | [summary](benchmark/runs/skillsbench_dc_train0906_test2/summary.json) | [summary](benchmark/runs/skillsbench_dc_train0906_test3/summary.json) |
| HermesSkills | [summary](benchmark/runs/skillsbench_no_hot_train0907_test1/summary.json) | [summary](benchmark/runs/skillsbench_no_hot_train0907_test2/summary.json) | [summary](benchmark/runs/skillsbench_no_hot_train0907_test3/summary.json) |
| TipsWarm | [summary](benchmark/runs/skillsbench_hot_train0917_test1/summary.json) · [pool](benchmark/runs/skillsbench_hot_train0917_test1/hot_pool.json) | [summary](benchmark/runs/skillsbench_hot_train0917_test2/summary.json) · [pool](benchmark/runs/skillsbench_hot_train0917_test2/hot_pool.json) | [summary](benchmark/runs/skillsbench_hot_train0917_test3/summary.json) · [pool](benchmark/runs/skillsbench_hot_train0917_test3/hot_pool.json) |

DeepSeek SkillsBench: HermesSkills [`ds_test1`](benchmark/runs/skillsbench_no_hot_train0911_ds_test1/summary.json) / [`ds_test2`](benchmark/runs/skillsbench_no_hot_train0911_ds_test2/summary.json) / [`ds_test3`](benchmark/runs/skillsbench_no_hot_train0911_ds_test3/summary.json); TipsWarm [`ds_test1`](benchmark/runs/skillsbench_hot_train0911_ds_test1/summary.json) · [pool](benchmark/runs/skillsbench_hot_train0911_ds_test1/hot_pool.json) / [`ds_test2`](benchmark/runs/skillsbench_hot_train0911_ds_test2/summary.json) · [pool](benchmark/runs/skillsbench_hot_train0911_ds_test2/hot_pool.json) / [`ds_test3`](benchmark/runs/skillsbench_hot_train0911_ds_test3/summary.json) · [pool](benchmark/runs/skillsbench_hot_train0911_ds_test3/hot_pool.json).

**AppWorld (`test_normal`)**

| Method | Seed 1 | Seed 2 | Seed 3 |
|--------|--------|--------|--------|
| A-MEM | [summary](benchmark/runs/appworld_amem0824_unseen/summary.json) | — | — |
| DC | [summary](benchmark/runs/appworld_dc0909_test/summary.json) | [summary](benchmark/runs/appworld_dc0909_test2/summary.json) | [summary](benchmark/runs/appworld_dc0909_test3/summary.json) |
| HermesSkills | [summary](benchmark/runs/appworld_no_hot_train0821_unseen1/summary.json) | [summary](benchmark/runs/appworld_no_hot_train0821_unseen2/summary.json) | [summary](benchmark/runs/appworld_no_hot_train0821_unseen3/summary.json) |
| TipsWarm | [summary](benchmark/runs/appworld_hot_train0901_unseen/summary.json) · [pool](benchmark/runs/appworld_hot_train0901_unseen/hot_pool.json) | [summary](benchmark/runs/appworld_hot_train0901_unseen2/summary.json) · [pool](benchmark/runs/appworld_hot_train0901_unseen2/hot_pool.json) | [summary](benchmark/runs/appworld_hot_train0901_unseen3/summary.json) · [pool](benchmark/runs/appworld_hot_train0901_unseen3/hot_pool.json) |

**ALFWorld (`valid_unseen`)**

| Method | Seed 1 | Seed 2 | Seed 3 |
|--------|--------|--------|--------|
| A-MEM | [summary](benchmark/runs/alfworld_amem_train0824_unseen0828/summary.json) | — | — |
| DC | [summary](benchmark/runs/alfworld_dc_train0906_unseen/summary.json) | [summary](benchmark/runs/alfworld_dc_train0906_unseen2/summary.json) | [summary](benchmark/runs/alfworld_dc_train0906_unseen3/summary.json) |
| HermesSkills | [summary](benchmark/runs/alfworld_no_hot_train0820_unseen/summary.json) | [summary](benchmark/runs/alfworld_no_hot_train0820_unseen2/summary.json) | [summary](benchmark/runs/alfworld_no_hot_train0820_unseen3/summary.json) |
| TipsWarm | [summary](benchmark/runs/alfworld_hot_train0825_unseen/summary.json) · [pool](benchmark/runs/alfworld_hot_train0825_unseen/hot_pool.json) | [summary](benchmark/runs/alfworld_hot_train0825_unseen2/summary.json) · [pool](benchmark/runs/alfworld_hot_train0825_unseen2/hot_pool.json) | [summary](benchmark/runs/alfworld_hot_train0825_unseen3/summary.json) · [pool](benchmark/runs/alfworld_hot_train0825_unseen3/hot_pool.json) |

A fuller directory list (train snapshots, ablations, hottest-pool transfers) is in [`benchmark/runs/README.md`](benchmark/runs/README.md).

---

## License

Hermes Agent is MIT — see [LICENSE](LICENSE). Vendored benchmarks keep their own licenses (SkillsBench Apache-2.0, AppWorld Apache-2.0, ALFWorld as upstream).
