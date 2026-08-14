"""CoEvoSkills Terminal-Bench Harbor protocol (no Docker / no LLM)."""

from __future__ import annotations

import json
from pathlib import Path

from benchmark.baselines.coevoskills import harbor_eval, prompts, skill_io
from benchmark.baselines.coevoskills.algorithm import run_coevo_skills
from benchmark.baselines.coevoskills.config import CoEvoConfig
from benchmark.baselines.coevoskills.generator import SkillGenerator
from benchmark.baselines.coevoskills.run_terminalbench_protocol import main as tb_protocol_main
from benchmark.baselines.coevoskills.verifier import SurrogateVerifier


_REPO = Path(__file__).resolve().parents[2]


def _write_skill(root: Path, name: str) -> None:
    pack = root / name
    pack.mkdir(parents=True)
    (pack / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: demo\n---\n# {name}\n",
        encoding="utf-8",
    )


def test_overlay_skills_copies_packages(tmp_path: Path):
    src = tmp_path / "lib"
    dest = tmp_path / "hermes_home" / "skills"
    dest.mkdir(parents=True)
    _write_skill(src, "evo-foo")
    _write_skill(dest, "keep-me")
    n = skill_io.overlay_skills(src, dest)
    assert n == 1
    assert (dest / "evo-foo" / "SKILL.md").is_file()
    assert (dest / "keep-me" / "SKILL.md").is_file()


def test_harbor_oracle_from_materialize():
    ev = harbor_eval.harbor_oracle_from_materialize(
        last_materialize={"evaluation": {"task_success": True, "reward": 1.0}}
    )
    assert ev["task_success"] is True
    assert ev["ran"] is True
    missing = harbor_eval.harbor_oracle_from_materialize(last_materialize=None)
    assert missing["task_success"] is False
    assert missing["ran"] is False


def test_frozen_eval_argv_tags_coevoskills(tmp_path: Path):
    argv = harbor_eval.frozen_eval_argv(
        dataset_path=tmp_path / "tasks",
        split_file=tmp_path / "split.json",
        split_part="test",
        model="Qwen/Qwen3.6-27B",
        harbor_env="docker",
        experiment_dir=tmp_path / "exp",
        library_dir=tmp_path / "lib",
        log_jsonl=tmp_path / "runs.jsonl",
        max_iterations=60,
        resume=True,
    )
    assert "--all" in argv
    assert "--skills-overlay" in argv
    assert "--run-method" in argv and "coevoskills" in argv
    assert "--run-phase" in argv and "frozen_eval" in argv
    assert "--resume" in argv
    assert "--isolate-hermes-home" in argv
    assert "--no-hot-pool" in argv
    assert "--pass-k" not in argv


def test_frozen_eval_argv_forwards_exclude_task_names(tmp_path: Path):
    argv = harbor_eval.frozen_eval_argv(
        dataset_path=tmp_path / "tasks",
        split_file=tmp_path / "split.json",
        split_part="train",
        model="Qwen/Qwen3.6-27B",
        harbor_env="docker",
        experiment_dir=tmp_path / "exp",
        library_dir=tmp_path / "lib",
        log_jsonl=tmp_path / "runs.jsonl",
        max_iterations=60,
        exclude_task_names=["math-eval-grader", "jax-speedrun-gpu"],
    )
    assert argv.count("--exclude-task-name") == 2
    assert argv[argv.index("--exclude-task-name") + 1] == "math-eval-grader"
    assert "jax-speedrun-gpu" in argv


def test_protocol_list_tasks_honors_exclude(tmp_path: Path, capsys):
    root = tmp_path / "terminal-bench" / "tasks"
    for name in ("keep-me", "math-eval-grader", "jax-speedrun-gpu"):
        d = root / name
        d.mkdir(parents=True)
        (d / "instruction.md").write_text("x\n", encoding="utf-8")
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps(
            {
                "train": ["keep-me", "math-eval-grader", "jax-speedrun-gpu"],
                "test": [],
                "val": [],
            }
        ),
        encoding="utf-8",
    )
    rc = tb_protocol_main(
        [
            "--hermes-root",
            str(_REPO),
            "--dataset-path",
            str(root),
            "--split-file",
            str(split),
            "--split-part",
            "train",
            "--experiment-dir",
            str(tmp_path / "exp"),
            "--list-tasks",
            "--exclude-task-name",
            "math-eval-grader",
            "--",
            "--exclude-task-name",
            "jax-speedrun-gpu",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "keep-me" in out
    assert "math-eval-grader" not in out.splitlines()
    assert "jax-speedrun-gpu" not in out.splitlines()
    assert "excluding 2 task(s)" in out


def test_tb_generator_prompt_mentions_sandbox():
    text = prompts.generator_system_prompt_terminalbench(
        instruction="do the thing",
        skills_dir="/tmp/skills",
        artifacts_dir="/tmp/artifacts",
    )
    assert "/app" in text
    assert "Harbor" in text
    assert "do the thing" in text


def test_algorithm_uses_harbor_oracle_when_surrogate_passes(tmp_path: Path, monkeypatch):
    dataset = tmp_path / "terminal-bench"
    task = dataset / "tasks" / "demo"
    (task / "environment").mkdir(parents=True)
    (task / "instruction.md").write_text("solve it\n", encoding="utf-8")
    work = tmp_path / "work"

    monkeypatch.setattr(
        SkillGenerator,
        "propose",
        lambda self, **kw: {
            "final_response": "wrote skills",
            "skills_listing": "(none)",
            "artifacts_listing": "(none)",
        },
    )
    monkeypatch.setattr(
        SurrogateVerifier,
        "evaluate",
        lambda self, **kw: {"r_hat": 1.0, "passed": True, "diagnostics": ""},
    )
    mats = []

    def materialize(**kw):
        mats.append(kw["iteration"])
        return {"evaluation": {"task_success": True, "reward": 1.0, "ran": True}}

    summary = run_coevo_skills(
        task_id="demo",
        cfg=CoEvoConfig(max_oracle_iters=1, max_surrogate_iters=2, model="x"),
        hermes_root=_REPO,
        skillsbench_root=dataset,
        work_root=work,
        progress=False,
        prompt_mode="harbor",
        materialize_fn=materialize,
        oracle_fn=harbor_eval.harbor_oracle_from_materialize,
    )
    assert summary["success"] is True
    assert mats == [1]
    assert summary["oracle_iters_used"] == 1


def test_protocol_frozen_eval_dry_run(tmp_path: Path, capsys):
    tasks = tmp_path / "terminal-bench" / "tasks" / "cad-model"
    tasks.mkdir(parents=True)
    (tasks / "instruction.md").write_text("do it\n", encoding="utf-8")
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps({"train": [], "test": ["cad-model"], "val": []}),
        encoding="utf-8",
    )
    exp = tmp_path / "exp"
    lib = exp / "coevoskills" / "_frozen_library"
    lib.mkdir(parents=True)
    _write_skill(lib, "evo-demo")
    rc = tb_protocol_main(
        [
            "--hermes-root",
            str(_REPO),
            "--dataset-path",
            str(tmp_path / "terminal-bench" / "tasks"),
            "--split-file",
            str(split),
            "--split-part",
            "test",
            "--experiment-dir",
            str(exp),
            "--frozen-eval",
            "--dry-run",
            "--log-jsonl",
            str(exp / "frozen.jsonl"),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "skills-overlay" in out
    assert "coevoskills" in out
    assert "frozen_eval" in out
    assert "--pass-k" not in out
    assert "--print-summary" not in out


def test_protocol_frozen_eval_dry_run_print_summary(tmp_path: Path, capsys):
    tasks = tmp_path / "terminal-bench" / "tasks" / "cad-model"
    tasks.mkdir(parents=True)
    (tasks / "instruction.md").write_text("do it\n", encoding="utf-8")
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps({"train": [], "test": ["cad-model"], "val": []}),
        encoding="utf-8",
    )
    exp = tmp_path / "exp"
    lib = exp / "coevoskills" / "_frozen_library"
    lib.mkdir(parents=True)
    _write_skill(lib, "evo-demo")
    rc = tb_protocol_main(
        [
            "--hermes-root",
            str(_REPO),
            "--dataset-path",
            str(tmp_path / "terminal-bench" / "tasks"),
            "--split-file",
            str(split),
            "--split-part",
            "test",
            "--experiment-dir",
            str(exp),
            "--frozen-eval",
            "--dry-run",
            "--print-summary",
            "--log-jsonl",
            str(exp / "frozen.jsonl"),
        ]
    )
    assert rc == 0
    assert "--print-summary" in capsys.readouterr().out


def test_print_task_summary_format(capsys):
    from benchmark.baselines.coevoskills.run_terminalbench_protocol import (
        _print_task_summary,
    )

    _print_task_summary(
        [
            {
                "skillsbench_task_id": "cad-model",
                "evaluation": {"task_success": True, "reward": 1.0},
            },
            {
                "skillsbench_task_id": "wdm-design",
                "evaluation": {"task_success": False, "reward": 0.0},
            },
        ],
        label="evolve",
    )
    out = capsys.readouterr().out
    assert "[evolve] tasks=2 passed=1 rate=0.500" in out
    assert "cad-model: success=True reward=1.0" in out
    assert "wdm-design: success=False reward=0.0" in out
