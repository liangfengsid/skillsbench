"""Skill / artifact filesystem helpers for CoEvoSkills workspaces."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class SkillPackage:
    name: str
    path: Path
    skill_md: str = ""
    script_files: Dict[str, str] = field(default_factory=dict)

    def meta_summary(self, max_chars: int = 4000) -> str:
        parts = [f"## Skill `{self.name}` ({self.path})", "### SKILL.md", self.skill_md[:max_chars]]
        for rel, body in sorted(self.script_files.items()):
            parts.append(f"### scripts/{rel}")
            parts.append(body[: max_chars // 2])
        return "\n\n".join(parts)


def ensure_workspace(root: Path, task_id: str) -> Dict[str, Path]:
    """Create per-task evolution dirs. Returns named paths."""
    base = root / task_id
    paths = {
        "base": base,
        "env": base / "environment",
        "skills": base / "skills",
        "artifacts": base / "artifacts",
        "surrogate_tests": base / "surrogate_tests",
        "history": base / "history",
        "logs": base / "logs",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def copy_task_environment(task_dir: Path, dest_env: Path) -> None:
    """Copy SkillsBench ``environment/`` (inputs only) into the workspace."""
    src = task_dir / "environment"
    if dest_env.exists():
        shutil.rmtree(dest_env)
    if src.is_dir():
        shutil.copytree(src, dest_env)
    else:
        dest_env.mkdir(parents=True, exist_ok=True)


def list_skills(skills_dir: Path) -> List[SkillPackage]:
    packs: List[SkillPackage] = []
    if not skills_dir.is_dir():
        return packs
    for child in sorted(skills_dir.iterdir()):
        if not child.is_dir():
            continue
        skill_md_path = child / "SKILL.md"
        if not skill_md_path.is_file():
            continue
        scripts: Dict[str, str] = {}
        scripts_dir = child / "scripts"
        if scripts_dir.is_dir():
            for f in sorted(scripts_dir.rglob("*")):
                if f.is_file() and f.suffix in {".py", ".sh", ".js", ".ts", ".r", ".R"}:
                    try:
                        scripts[str(f.relative_to(scripts_dir))] = f.read_text(
                            encoding="utf-8", errors="replace"
                        )
                    except OSError:
                        continue
        packs.append(
            SkillPackage(
                name=child.name,
                path=child,
                skill_md=skill_md_path.read_text(encoding="utf-8", errors="replace"),
                script_files=scripts,
            )
        )
    return packs


def skills_listing_text(skills_dir: Path) -> str:
    packs = list_skills(skills_dir)
    if not packs:
        return "(no skills yet)"
    lines = []
    for p in packs:
        n_scripts = len(p.script_files)
        lines.append(f"- {p.name}/  (SKILL.md + {n_scripts} script file(s))")
    return "\n".join(lines)


def skills_meta_context(skills_dir: Path, max_chars: int = 12000) -> str:
    """Persistent C skill content for the generator (paper: C = (I, S_meta))."""
    packs = list_skills(skills_dir)
    if not packs:
        return "(empty skill library)"
    chunks = [p.meta_summary() for p in packs]
    text = "\n\n---\n\n".join(chunks)
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n...[truncated skill meta]..."
    return text


def list_artifacts(artifacts_dir: Path, max_files: int = 80) -> str:
    if not artifacts_dir.is_dir():
        return "(missing artifacts dir)"
    files = [p for p in sorted(artifacts_dir.rglob("*")) if p.is_file()]
    if not files:
        return "(no artifact files)"
    lines = []
    for p in files[:max_files]:
        rel = p.relative_to(artifacts_dir)
        size = p.stat().st_size
        lines.append(f"- {rel} ({size} bytes)")
    if len(files) > max_files:
        lines.append(f"... and {len(files) - max_files} more")
    return "\n".join(lines)


def sync_artifacts_into_task_dir(artifacts_dir: Path, task_dir: Path) -> List[str]:
    """
    Copy evolved artifacts into the SkillsBench task tree so host pytest can see them.

    SkillsBench tests typically look under the task directory / common output paths.
    We mirror flat files into task_dir root and preserve relative structure.
    """
    copied: List[str] = []
    if not artifacts_dir.is_dir():
        return copied
    for src in artifacts_dir.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(artifacts_dir)
        dest = task_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied.append(str(rel))
    return copied


def install_skills_into_task(skills_dir: Path, task_dir: Path) -> Path:
    """Install evolved skills under ``task_dir/environment/skills/`` (paper path)."""
    dest = task_dir / "environment" / "skills"
    dest.mkdir(parents=True, exist_ok=True)
    for pack in list_skills(skills_dir):
        target = dest / pack.name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(pack.path, target)
    return dest


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def append_history(history_dir: Path, event: Dict[str, Any]) -> None:
    history_dir.mkdir(parents=True, exist_ok=True)
    path = history_dir / "events.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, default=str) + "\n")


def dump_run_summary(base: Path, summary: Dict[str, Any]) -> Path:
    out = base / "run_summary.json"
    write_json(out, summary)
    return out
