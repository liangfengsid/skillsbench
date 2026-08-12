<p align="center">
  <img src="assets/banner.png" alt="Hermes Agent" width="100%">
</p>

# Hermes Agent ☤

<p align="center">
  <a href="https://hermes-agent.nousresearch.com/docs/"><img src="https://img.shields.io/badge/Docs-hermes--agent.nousresearch.com-FFD700?style=for-the-badge" alt="Documentation"></a>
  <a href="https://discord.gg/NousResearch"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://github.com/NousResearch/hermes-agent/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License: MIT"></a>
  <a href="https://nousresearch.com"><img src="https://img.shields.io/badge/Built%20by-Nous%20Research-blueviolet?style=for-the-badge" alt="Built by Nous Research"></a>
</p>

**The self-improving AI agent built by [Nous Research](https://nousresearch.com).** It's the only agent with a built-in learning loop — it creates skills from experience, improves them during use, nudges itself to persist knowledge, searches its own past conversations, and builds a deepening model of who you are across sessions. Run it on a $5 VPS, a GPU cluster, or serverless infrastructure that costs nearly nothing when idle. It's not tied to your laptop — talk to it from Telegram while it works on a cloud VM.

Use any model you want — [Nous Portal](https://portal.nousresearch.com), [OpenRouter](https://openrouter.ai) (200+ models), [NVIDIA NIM](https://build.nvidia.com) (Nemotron), [Xiaomi MiMo](https://platform.xiaomimimo.com), [z.ai/GLM](https://z.ai), [Kimi/Moonshot](https://platform.moonshot.ai), [MiniMax](https://www.minimax.io), [Hugging Face](https://huggingface.co), OpenAI, or your own endpoint. Switch with `hermes model` — no code changes, no lock-in.

<table>
<tr><td><b>A real terminal interface</b></td><td>Full TUI with multiline editing, slash-command autocomplete, conversation history, interrupt-and-redirect, and streaming tool output.</td></tr>
<tr><td><b>Lives where you do</b></td><td>Telegram, Discord, Slack, WhatsApp, Signal, and CLI — all from a single gateway process. Voice memo transcription, cross-platform conversation continuity.</td></tr>
<tr><td><b>A closed learning loop</b></td><td>Agent-curated memory with periodic nudges. Autonomous skill creation after complex tasks. Skills self-improve during use. FTS5 session search with LLM summarization for cross-session recall. <a href="https://github.com/plastic-labs/honcho">Honcho</a> dialectic user modeling. Compatible with the <a href="https://agentskills.io">agentskills.io</a> open standard.</td></tr>
<tr><td><b>Scheduled automations</b></td><td>Built-in cron scheduler with delivery to any platform. Daily reports, nightly backups, weekly audits — all in natural language, running unattended.</td></tr>
<tr><td><b>Delegates and parallelizes</b></td><td>Spawn isolated subagents for parallel workstreams. Write Python scripts that call tools via RPC, collapsing multi-step pipelines into zero-context-cost turns.</td></tr>
<tr><td><b>Runs anywhere, not just your laptop</b></td><td>Six terminal backends — local, Docker, SSH, Daytona, Singularity, and Modal. Daytona and Modal offer serverless persistence — your agent's environment hibernates when idle and wakes on demand, costing nearly nothing between sessions. Run it on a $5 VPS or a GPU cluster.</td></tr>
<tr><td><b>Research-ready</b></td><td>Batch trajectory generation, Atropos RL environments, trajectory compression for training the next generation of tool-calling models.</td></tr>
</table>

---

## Quick Install

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

Works on Linux, macOS, WSL2, and Android via Termux. The installer handles the platform-specific setup for you.

> **Android / Termux:** The tested manual path is documented in the [Termux guide](https://hermes-agent.nousresearch.com/docs/getting-started/termux). On Termux, Hermes installs a curated `.[termux]` extra because the full `.[all]` extra currently pulls Android-incompatible voice dependencies.
>
> **Windows:** Native Windows is not supported. Please install [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) and run the command above.

After installation:

```bash
source ~/.bashrc    # reload shell (or: source ~/.zshrc)
hermes              # start chatting!
```

**SkillsBench** (`benchmark/skillsbench/`) is a nested Python subproject (BenchFlow). To add BenchFlow to the same venv as Hermes: `pip install -e ".[skillsbench]"` from the repo root. Task authoring and BenchFlow CLI: `benchmark/skillsbench/README.md` and `benchmark/skillsbench/AGENTS.md`.

### Benchmarks

Hermes batch drivers (SkillsBench, HLE, AppWorld), hot-pool evaluation, and run comparison live under **`benchmark/`**. See **[`benchmark/README.md`](benchmark/README.md)** for install prerequisites, example commands, and script reference. Conceptually, the hot skill pool is described under [Hot skills](#hot-skills-ephemeral-key-point-pool) below.

### Skills: step-level variant pools (optional)

Skills can expose **multiple procedural variants per step** instead of a single fixed instruction. When enabled, `skill_view` expands tagged regions in **`SKILL.md`** and merges in data from **`step_pools.json`** next to that file.

**1. Turn it on** in `~/.hermes/config.yaml`:

```yaml
skills:
  step_pools:
    enabled: true
    default_max_variants_per_step: 5   # optional cap when a step omits max_variants
    # Optional: how variants sort after manual_order (laplace | raw | wilson | ucb1)
    # rank_strategy: laplace
    # ucb1_c: 1.4142135623730951   # only for rank_strategy: ucb1
```

**2. Mark steps in `SKILL.md`** (baseline text is the content between the tags):

```markdown
<!-- hermes-step id="install" max_variants="4" -->
Run `npm ci` in the project root.
<!-- /hermes-step -->
```

**3. Store extra variants** (bodies + optional success/fail counts) in **`step_pools.json`** in the same skill directory. Hermes ranks variants (with optional manual order) and trims to `max_variants` or the default above.

**4. At runtime** the agent can maintain pools with the **`skill_step_variant`** tool (skills toolset): `record_attempt`, `add_variant`, `remove_variant`, `patch_variant`, `set_variant_order`, `clear_variant_order`, `list_pools`. Editing is limited to **local** skills under `~/.hermes/skills/`. Change the baseline prose in **`SKILL.md`** with `skill_manage`, not `patch_variant`.

**5. Rank strategy** (`skills.step_pools.rank_strategy`, default `laplace`): after any `manual_order` prefix, remaining variants sort by a numeric score (higher = try earlier). **`laplace`** — smoothed `(success+1)/(trials+2)` for small-sample stability. **`raw`** — empirical `success/max(1,trials)` (no prior; can reorder vs Laplace when trial counts differ). **`wilson`** — conservative 95% Wilson lower bound on the success rate. **`ucb1`** — mean rate plus an exploration bonus using total pool trials and optional `ucb1_c` (default `√2`).

Implementation: [`agent/skill_step_pools.py`](agent/skill_step_pools.py), [`tools/skill_step_variant_tool.py`](tools/skill_step_variant_tool.py).

### Hot skills (ephemeral key-point pool)

Skills already expose full procedures through `skill_view`, but loading every recently useful skill into context is expensive and cache-hostile. The **hot skill pool** keeps a small LRU of **decisive key points** (guardrails, pitfalls, “always/never” rules) and injects them **ephemerally** into the current turn’s user message at API-call time — not into the durable system prompt — so the model gets short reminders without rewriting cached prefixes.

**Idea in one sentence:** remember *what not to mess up* from skills you just used; open the full skill again only when you need the procedure.

```
skill_view / skill_manage / recent use
        │
        ▼
  extract key points  ──►  HotSkillPool (LRU + TTL + budget)
        │
        ▼
  <hot-skills>…</hot-skills>  prepended to this turn’s user message
        │
        ▼
  model may still call skill_view(name) for the full SKILL.md
```

**What gets extracted** (in order):

1. Guardrail-style headings used by Hermes skills **and** SkillsBench task
   skills under `tasks/*/environment/skills/`:
   - Hermes: `## Common Pitfalls` / `## Pitfalls` / `## Key Points` / …
   - SkillsBench: `## Best Practices` / `## Limitations` / `## Error Handling`
     / `## Important Requirements` / `CRITICAL: …` formula rules / …
2. Optional author override markers (rare; mostly for hand-tuned skills):
   ```markdown
   <!-- hermes-hot -->
   - NEVER hardcode `~/.hermes` — use `get_hermes_home()`.
   - ALWAYS run tests via `scripts/run_tests.sh`, not bare `pytest`.
   <!-- /hermes-hot -->
   ```
3. Fallback: imperative / NEVER–ALWAYS style bullets elsewhere in the skill.

**Lifecycle**

| Stage | Behavior |
|-------|----------|
| Populate | After a skill is viewed/used (and optionally by hydrating from recent tool history), key points enter the pool |
| Inject | Each user turn may prepend a capped `<hot-skills>` block with a system note that these are guardrails, not new user text |
| Evict | LRU + per-entry TTL (turns) + char/entry budgets keep the block small |
| Persist (optional) | Pool JSON can survive across conversations / sequential benchmark tasks |

**Configure** in `~/.hermes/config.yaml`:

```yaml
skills:
  hot_pool:
    enabled: true
    entry_schedule: global_pool   # or per_skill
    max_entries: 12               # injected key-point budget (global_pool)
    max_skills: 5
    max_pool_skills: 15           # how many skills the LRU may hold
    max_chars: 4000
    ttl_turns: 20
    persist_across_conversations: false
    # persist_path: ""            # default ~/.hermes/hot_skill_pool.json when persisting
```

**Not the same as** step-level variant pools (above): step pools rewrite *which procedure variant* `skill_view` shows; hot skills inject *short reminders* without opening the skill body.

Implementation: [`agent/hot_skills.py`](agent/hot_skills.py). Benchmark A/B usage (`--hot-pool`, `--hot-pool-persist`): [`benchmark/README.md`](benchmark/README.md#hot-skill-key-points-cross-task-pool).

---

## Getting Started

```bash
hermes              # Interactive CLI — start a conversation
hermes model        # Choose your LLM provider and model
hermes tools        # Configure which tools are enabled
hermes config set   # Set individual config values
hermes gateway      # Start the messaging gateway (Telegram, Discord, etc.)
hermes setup        # Run the full setup wizard (configures everything at once)
hermes claw migrate # Migrate from OpenClaw (if coming from OpenClaw)
hermes update       # Update to the latest version
hermes doctor       # Diagnose any issues
```



📖 **[Full documentation →](https://hermes-agent.nousresearch.com/docs/)**

## CLI vs Messaging Quick Reference

Hermes has two entry points: start the terminal UI with `hermes`, or run the gateway and talk to it from Telegram, Discord, Slack, WhatsApp, Signal, or Email. Once you're in a conversation, many slash commands are shared across both interfaces.

| Action | CLI | Messaging platforms |
|---------|-----|---------------------|
| Start chatting | `hermes` | Run `hermes gateway setup` + `hermes gateway start`, then send the bot a message |
| Start fresh conversation | `/new` or `/reset` | `/new` or `/reset` |
| Change model | `/model [provider:model]` | `/model [provider:model]` |
| Set a personality | `/personality [name]` | `/personality [name]` |
| Retry or undo the last turn | `/retry`, `/undo` | `/retry`, `/undo` |
| Compress context / check usage | `/compress`, `/usage`, `/insights [--days N]` | `/compress`, `/usage`, `/insights [days]` |
| Browse skills | `/skills` or `/<skill-name>` | `/<skill-name>` |
| Interrupt current work | `Ctrl+C` or send a new message | `/stop` or send a new message |
| Platform-specific status | `/platforms` | `/status`, `/sethome` |

For the full command lists, see the [CLI guide](https://hermes-agent.nousresearch.com/docs/user-guide/cli) and the [Messaging Gateway guide](https://hermes-agent.nousresearch.com/docs/user-guide/messaging).

---

## Documentation

All documentation lives at **[hermes-agent.nousresearch.com/docs](https://hermes-agent.nousresearch.com/docs/)**:

| Section | What's Covered |
|---------|---------------|
| [Quickstart](https://hermes-agent.nousresearch.com/docs/getting-started/quickstart) | Install → setup → first conversation in 2 minutes |
| [CLI Usage](https://hermes-agent.nousresearch.com/docs/user-guide/cli) | Commands, keybindings, personalities, sessions |
| [Configuration](https://hermes-agent.nousresearch.com/docs/user-guide/configuration) | Config file, providers, models, all options |
| [Messaging Gateway](https://hermes-agent.nousresearch.com/docs/user-guide/messaging) | Telegram, Discord, Slack, WhatsApp, Signal, Home Assistant |
| [Security](https://hermes-agent.nousresearch.com/docs/user-guide/security) | Command approval, DM pairing, container isolation |
| [Tools & Toolsets](https://hermes-agent.nousresearch.com/docs/user-guide/features/tools) | 40+ tools, toolset system, terminal backends |
| [Skills System](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills) | Procedural memory, Skills Hub, creating skills |
| [Memory](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory) | Persistent memory, user profiles, best practices |
| [MCP Integration](https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp) | Connect any MCP server for extended capabilities |
| [Cron Scheduling](https://hermes-agent.nousresearch.com/docs/user-guide/features/cron) | Scheduled tasks with platform delivery |
| [Context Files](https://hermes-agent.nousresearch.com/docs/user-guide/features/context-files) | Project context that shapes every conversation |
| [Architecture](https://hermes-agent.nousresearch.com/docs/developer-guide/architecture) | Project structure, agent loop, key classes |
| [Contributing](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing) | Development setup, PR process, code style |
| [CLI Reference](https://hermes-agent.nousresearch.com/docs/reference/cli-commands) | All commands and flags |
| [Environment Variables](https://hermes-agent.nousresearch.com/docs/reference/environment-variables) | Complete env var reference |

---

## Migrating from OpenClaw

If you're coming from OpenClaw, Hermes can automatically import your settings, memories, skills, and API keys.

**During first-time setup:** The setup wizard (`hermes setup`) automatically detects `~/.openclaw` and offers to migrate before configuration begins.

**Anytime after install:**

```bash
hermes claw migrate              # Interactive migration (full preset)
hermes claw migrate --dry-run    # Preview what would be migrated
hermes claw migrate --preset user-data   # Migrate without secrets
hermes claw migrate --overwrite  # Overwrite existing conflicts
```

What gets imported:
- **SOUL.md** — persona file
- **Memories** — MEMORY.md and USER.md entries
- **Skills** — user-created skills → `~/.hermes/skills/openclaw-imports/`
- **Command allowlist** — approval patterns
- **Messaging settings** — platform configs, allowed users, working directory
- **API keys** — allowlisted secrets (Telegram, OpenRouter, OpenAI, Anthropic, ElevenLabs)
- **TTS assets** — workspace audio files
- **Workspace instructions** — AGENTS.md (with `--workspace-target`)

See `hermes claw migrate --help` for all options, or use the `openclaw-migration` skill for an interactive agent-guided migration with dry-run previews.

---

## Contributing

We welcome contributions! See the [Contributing Guide](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing) for development setup, code style, and PR process.

Quick start for contributors — clone and go with `setup-hermes.sh`:

```bash
git clone https://github.com/NousResearch/hermes-agent.git
cd hermes-agent
./setup-hermes.sh     # installs uv, creates venv, installs .[all], symlinks ~/.local/bin/hermes
./hermes              # auto-detects the venv, no need to `source` first
```

Manual path (equivalent to the above):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv venv --python 3.11
source venv/bin/activate
uv pip install -e ".[all,dev]"
scripts/run_tests.sh
```

> **RL Training (optional):** The RL/Atropos integration (`environments/`) ships via the `atroposlib` and `tinker` dependencies pulled in by `.[all,dev]` — no submodule setup required.

---

## Community

- 💬 [Discord](https://discord.gg/NousResearch)
- 📚 [Skills Hub](https://agentskills.io)
- 🐛 [Issues](https://github.com/NousResearch/hermes-agent/issues)
- 🔌 [HermesClaw](https://github.com/AaronWong1999/hermesclaw) — Community WeChat bridge: Run Hermes Agent and OpenClaw on the same WeChat account.

---

## License

MIT — see [LICENSE](LICENSE).

Built by [Nous Research](https://nousresearch.com).
