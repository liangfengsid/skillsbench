"""Prompt templates adapted from CoEvoSkills appendix F and Alg. 1 roles.

Generator keeps persistent skill-meta context C = (I, S_meta).
Surrogate verifier is info-isolated: sees only instruction I + artifacts x (+ own V).
"""

from __future__ import annotations

SKILL_META_SKELETON = """\
# Skill package conventions (S_meta)

Each skill is a directory:

```
<skill-name>/
  SKILL.md          # YAML frontmatter (name, description) + workflow guidance
  scripts/          # small, independently testable utility functions
    utils.py
```

Rules:
- Prefer many small functions over one monolithic script.
- Skills must be self-contained (encode domain rules inside the skill; do not rely on external docs at use time).
- Name evolved skills with the ``evo-`` prefix (e.g. ``evo-citation-checker``).
- Skill directory names may contain hyphens; Python imports MUST use ``sys.path.insert`` on the ``scripts/`` directory, then ``from utils import ...``. Never ``from evo_foo.scripts...``.
- After fixing bugs at runtime, write fixes back into ``scripts/`` so a fresh agent can reuse the skill.
- Every SKILL.md MUST include a ``## Common Pitfalls`` or ``## Best Practices``
  (or ``## Key Points`` / ``## Limitations``) section with short bullets of
  mistakes to avoid — these are what Hermes injects into later turns via the
  hot skill pool (same conventions as SkillsBench ``environment/skills``).
"""


def generator_system_prompt(*, instruction: str, skills_dir: str, artifacts_dir: str) -> str:
    return f"""You are the CoEvoSkills Skill Generator (π_θ).

Your job is to CREATE/UPDATE reusable skill packages and then EXECUTE the task by
IMPORTING those skill utilities — never by editing final output files by hand.

Persistent skill conventions:
{SKILL_META_SKELETON}

Workspace:
- Write/update skills under: {skills_dir}
- Produce all task output artifacts under: {artifacts_dir}
- Input data lives under the task environment directory provided in the user message.

Phases (do in order):
1. Discover environment inputs (data files, docs). Read docs fully if present.
2. Create or update evo-* skills under {skills_dir} (SKILL.md + scripts/).
3. Self-check: every instruction requirement is covered by the skill.
4. Execute by importing skill scripts; write outputs only under {artifacts_dir}.
5. If prior feedback is provided, fix skills and regenerate outputs from scratch.

Task instruction I:
{instruction}
"""


def generator_turn_prompt(
    *,
    feedback: str,
    iteration: int,
    skills_listing: str,
) -> str:
    parts = [
        f"Generator turn {iteration}.",
        "Current skills on disk:",
        skills_listing or "(none yet)",
    ]
    if feedback.strip():
        parts.append("Feedback from prior verification (use this to improve skills):")
        parts.append(feedback.strip())
    else:
        parts.append("No prior verifier feedback — produce an initial skill + artifacts.")
    parts.append(
        "Update skills if needed, then regenerate all required artifacts. "
        "When done, briefly list which skill dirs and artifact files you wrote."
    )
    return "\n\n".join(parts)


def verifier_system_prompt(*, instruction: str, artifacts_dir: str, tests_dir: str) -> str:
    return f"""You are the CoEvoSkills Surrogate Verifier (π_θ^V).

CRITICAL INFORMATION BARRIER:
- You see ONLY the task instruction I and the agent's output artifacts x.
- You do NOT see ground-truth tests, hidden rubrics, or the generator's skill source.
- You may maintain and escalate your OWN test suite V under {tests_dir}.

Goals:
1. Inspect artifacts under {artifacts_dir}.
2. Write/update pytest tests in {tests_dir}/test_surrogate.py that check instruction compliance.
3. Run your tests (or reason carefully if you cannot execute) and report:
   - estimated reward R_hat in [0, 1]
   - whether all your checks passed
   - dense diagnostics (which checks failed and why) when R_hat < 1
4. When told that a previous surrogate pass disagreed with an opaque oracle fail,
   ESCALATE V: add stricter / more discriminative checks that catch the miss.

Task instruction I:
{instruction}
"""


def verifier_turn_prompt(
    *,
    iteration: int,
    escalate: bool,
    artifact_listing: str,
    prior_diagnostics: str = "",
) -> str:
    parts = [
        f"Surrogate verifier turn {iteration}.",
        "Artifacts present:",
        artifact_listing or "(no artifacts found)",
    ]
    if escalate:
        parts.append(
            "ESCALATION: Your previous suite estimated success, but the opaque ground-truth "
            "oracle failed. Strengthen V so false positives are less likely. Do not invent "
            "oracle test contents — only tighten checks implied by the instruction and artifacts."
        )
    if prior_diagnostics.strip():
        parts.append("Prior surrogate diagnostics:")
        parts.append(prior_diagnostics.strip())
    parts.append(
        f"Write/update tests under the tests directory, then summarize JSON on the last line:\n"
        f'{{"r_hat": <float 0-1>, "passed": <bool>, "diagnostics": "<markdown>"}}'
    )
    return "\n\n".join(parts)


def opaque_oracle_feedback(*, passed: bool, reward: float) -> str:
    """Ground-truth signal exposed to the generator — pass/fail only (paper §3.2)."""
    if passed:
        return (
            "OPAQUE ORACLE: PASS (reward=1.0). Task verified by ground-truth tests. "
            "Do not request test contents."
        )
    return (
        f"OPAQUE ORACLE: FAIL (reward={reward:.4f}). "
        "Ground-truth verification failed. Details of oracle tests are withheld. "
        "Improve skills and regenerate artifacts."
    )
