# A-Mem memory provider

[A-Mem](https://github.com/WujiangXu/A-mem-sys) (Xu et al., [arXiv:2502.12110](https://arxiv.org/abs/2502.12110)) as a Hermes **memory-provider plugin**. Used as a paper baseline against the hot-skill pool.

This is **not** a skill factory. Skill tools (`skill_view` / `skill_manage`) stay on. A-Mem only stores and retrieves episode notes.

## Fair comparison vs hot-skill

| Path | Skills | Hot-skill pool | Builtin MEMORY.md | Extra tools |
|------|--------|----------------|-------------------|-------------|
| Hot-skill | on | on | off (benchmarks) | none |
| A-Mem | on | **off** | **off** | **none** |

Retrieved notes inject via the existing `<memory-context>` fence on the current user message (`prefetch()` once per user turn). Turns are **buffered**; by default a note is written every **5** turns (`HERMES_AMEM_SYNC_EVERY=5`), not on every `sync_turn`. Set `0` for one note per episode.

## Enable

Benchmarks: `--amem` on the SkillsBench / ALFWorld / AppWorld drivers (sets `HERMES_AMEM_ENABLED=1`).

```bash
pip install -e ".[amem]"
```

That pulls [A-mem-sys](https://github.com/WujiangXu/A-mem-sys). Prefetch MiniLM before the first `--amem` run (mirror if Hugging Face is blocked):

```bash
export HF_ENDPOINT=https://hf-mirror.com   # optional; required when huggingface.co is unreachable
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

Step-by-step: [`benchmark/baselines/amem/README.md`](../../../benchmark/baselines/amem/README.md#prefetch-minilm-all-minilm-l6-v2).

Interactive Hermes (`memory.provider: amem` in config.yaml) is supported but intended for experiments, not daily use. Prefer the driver flags so `skip_memory=True` still loads this plugin without MEMORY.md.

## Persistence

Upstream `AgenticMemorySystem` resets an in-memory Chroma client on construct. This plugin:

- Uses a **disk-backed** Chroma client under `$HERMES_AMEM_PATH/chroma`
- Snapshots notes to `$HERMES_AMEM_PATH/amem_notes.json` (AppWorld task subprocesses reload this)
- Reuses a **process singleton** for in-process drivers (SkillsBench, ALFWorld)

Env vars (set by the drivers):

- `HERMES_AMEM_ENABLED=1`
- `HERMES_AMEM_PATH` — store directory (default: `$HERMES_HOME/amem`)
- `HERMES_AMEM_K` — notes per prefetch (default 5)
- `HERMES_AMEM_SYNC_EVERY` — flush every N buffered turns (default **5**);
  `0`/`episode` = one write per episode; `1` = legacy per-turn
- `HERMES_AMEM_READONLY=1` — frozen eval (prefetch only; no writes). Drivers:
  `--amem-freeze`
- `HERMES_AMEM_LLM_MODEL` / `HERMES_AMEM_API_KEY` / `HERMES_AMEM_API_BASE`
- `HERMES_AMEM_EMBED_MODEL` (default `all-MiniLM-L6-v2`)

See [`benchmark/baselines/amem/README.md`](../../../benchmark/baselines/amem/README.md) for copy-workspace train → test (`--amem-sync-every 20`) commands.
