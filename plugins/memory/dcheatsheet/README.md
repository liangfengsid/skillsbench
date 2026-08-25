# Dynamic Cheatsheet memory provider

[Dynamic Cheatsheet](https://github.com/suzgunmirac/dynamic-cheatsheet) (Suzgun et al., [arXiv:2504.07952](https://arxiv.org/abs/2504.07952)) as a Hermes **memory-provider plugin**. Used as a paper baseline against the hot-skill pool.

This is **not** a skill factory and does **not** wrap the official DC generator. Hermes remains the generator (`AIAgent`). Official **curator** prompts (DC-Cu / DC-RS) update a persistent cheatsheet after (or, for RS, before) each turn.

## Fair comparison vs hot-skill

| Path | Skills | Hot-skill pool | Builtin MEMORY.md | Extra tools |
|------|--------|----------------|-------------------|-------------|
| Hot-skill | on | on | off (benchmarks) | none |
| Dynamic Cheatsheet | on | **off** | **off** | **none** |

The current cheatsheet injects via the existing `<memory-context>` fence on the current user message (`prefetch()`). After each turn, `sync_turn()` appends the episode and (DC-Cu) runs the official curator prompt.

## Enable

Benchmarks: `--dc` on the SkillsBench / ALFWorld / AppWorld drivers (sets `HERMES_DC_ENABLED=1`).

No extra pip extra is required for the default **DC-Cu** mode. `--dc-mode rs` / `curetr` use MiniLM if `sentence-transformers` is installed (same prefetch as A-Mem); otherwise retrieval falls back to recency.

Interactive Hermes (`memory.provider: dcheatsheet` in config.yaml) is supported but intended for experiments. Prefer the driver flags so `skip_memory=True` still loads this plugin without MEMORY.md.

## Persistence

```
$HERMES_DC_PATH/cheatsheet.txt
$HERMES_DC_PATH/episodes.jsonl
```

AppWorld `--all` workers reload from disk. In-process drivers (SkillsBench, ALFWorld) reuse a process singleton.

Env vars (set by the drivers):

- `HERMES_DC_ENABLED=1`
- `HERMES_DC_PATH` — store directory (default: `$HERMES_HOME/dcheatsheet`)
- `HERMES_DC_MODE` — `cu` (default), `rs`, or `curetr`
- `HERMES_DC_K` — retrieved episodes for `rs` / `curetr` (default 3)
- `HERMES_DC_LLM_MODEL` / `HERMES_DC_API_KEY` / `HERMES_DC_API_BASE` — curator LLM (same as the agent by default)
- `HERMES_DC_EMBED_MODEL` — retrieval encoder (default `all-MiniLM-L6-v2`)

See [`benchmark/baselines/dcheatsheet/README.md`](../../../benchmark/baselines/dcheatsheet/README.md) for train → eval commands.
