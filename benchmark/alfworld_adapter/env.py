"""Discover ALFWorld games and open a single TextWorld env (eval, not THOR)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Official AlfredTWEnv skips these path substrings (unsupported in TextWorld).
_SKIP_SUBSTRINGS = ("movable", "Sliced")

TASK_TYPES = {
    1: "pick_and_place_simple",
    2: "look_at_obj_in_light",
    3: "pick_clean_then_place_in_recep",
    4: "pick_heat_then_place_in_recep",
    5: "pick_cool_then_place_in_recep",
    6: "pick_two_obj_and_place",
}

# CLI names → json_2.1.1 folder names.
SPLIT_ALIASES = {
    "train": "train",
    "valid_seen": "valid_seen",
    "valid_unseen": "valid_unseen",
    "valid_train": "valid_train",
    "eval_in_distribution": "valid_seen",
    "eval_out_of_distribution": "valid_unseen",
    "seen": "valid_seen",
    "unseen": "valid_unseen",
}

_CANDIDATE_DATA_DIRS = (
    Path("/data/liangfeng/alfwordData"),  # existing checkout (typo in folder name)
    Path("/data/liangfeng/alfworldData"),
    Path.home() / ".cache" / "alfworld",
)


def _repo_alfworld_src() -> Path:
    return Path(__file__).resolve().parents[1] / "alfworld"


def ensure_alfworld_on_path() -> Path:
    """Make vendored ``benchmark/alfworld`` importable as ``alfworld``."""
    src = _repo_alfworld_src()
    src_s = str(src)
    if src.is_dir() and src_s not in sys.path:
        sys.path.insert(0, src_s)
    return src


def resolve_alfworld_data(explicit: Optional[str | Path] = None) -> Path:
    """Resolve ``ALFWORLD_DATA``: CLI flag, env, then known local paths."""
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"ALFWORLD_DATA is not a directory: {path}")
        return path
    env = os.environ.get("ALFWORLD_DATA", "").strip()
    if env:
        path = Path(env).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"ALFWORLD_DATA is not a directory: {path}")
        return path
    for candidate in _CANDIDATE_DATA_DIRS:
        if candidate.is_dir() and (candidate / "json_2.1.1").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find ALFWorld data. Download PDDL/game files and set "
        "ALFWORLD_DATA or pass --alfworld-data. Looked at: "
        + ", ".join(str(p) for p in _CANDIDATE_DATA_DIRS)
    )


def apply_alfworld_data_env(data_root: Path) -> None:
    """Export ``ALFWORLD_DATA`` before importing ``alfworld.info``."""
    resolved = data_root.expanduser().resolve()
    os.environ["ALFWORLD_DATA"] = str(resolved)


def split_dir(data_root: Path, split: str) -> Path:
    folder = SPLIT_ALIASES.get(split, split)
    path = data_root / "json_2.1.1" / folder
    if not path.is_dir():
        raise FileNotFoundError(
            f"ALFWorld split {split!r} not found at {path}. "
            f"Known splits: {', '.join(sorted(set(SPLIT_ALIASES.values())))}"
        )
    return path


def _task_type_from_folder(name: str) -> str:
    return name.split("-", 1)[0]


def _should_skip(path: Path) -> bool:
    text = str(path)
    return any(token in text for token in _SKIP_SUBSTRINGS)


def _game_is_solvable(game_file: Path) -> bool:
    try:
        data = json.loads(game_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if "solvable" not in data:
        return False
    return bool(data["solvable"])


def _task_id_for(split_root: Path, game_file: Path) -> str:
    rel = game_file.parent.relative_to(split_root)
    return rel.as_posix()


def discover_games(
    data_root: Path,
    split: str,
    *,
    task_types: Optional[Sequence[int]] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Return solvable TextWorld games for *split*, sorted by task id."""
    root = split_dir(data_root, split)
    allowed: Optional[set[str]] = None
    if task_types:
        allowed = {TASK_TYPES[int(i)] for i in task_types if int(i) in TASK_TYPES}

    games: List[Dict[str, Any]] = []
    for game_file in sorted(root.rglob("game.tw-pddl")):
        if _should_skip(game_file):
            continue
        if not _game_is_solvable(game_file):
            continue
        task_id = _task_id_for(root, game_file)
        folder = game_file.parent.parent.name if game_file.parent.parent != root else game_file.parent.name
        task_type = _task_type_from_folder(folder)
        if allowed is not None and task_type not in allowed:
            continue
        games.append(
            {
                "task_id": task_id,
                "task_type": task_type,
                "game_file": str(game_file),
                "problem_dir": str(game_file.parent),
                "split": SPLIT_ALIASES.get(split, split),
            }
        )
        if limit is not None and len(games) >= int(limit):
            break
    return games


def lookup_game(games: Sequence[Dict[str, Any]], task_id: str) -> Dict[str, Any]:
    wanted = task_id.strip().rstrip("/")
    for game in games:
        if game["task_id"] == wanted:
            return game
        if game["task_id"].endswith("/" + wanted) or game["task_id"].startswith(wanted + "/"):
            return game
        if Path(game["problem_dir"]).name == wanted:
            return game
    raise KeyError(f"Unknown ALFWorld task id: {task_id}")


def _unpack_reset(out: Any) -> Tuple[Any, Dict[str, Any]]:
    if isinstance(out, tuple) and len(out) == 2:
        obs, info = out
        return obs, info if isinstance(info, dict) else {}
    return out, {}


def _unpack_step(out: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
    if not isinstance(out, tuple):
        raise TypeError(f"Unexpected env.step return: {type(out)!r}")
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
        return obs, float(reward or 0), done, info if isinstance(info, dict) else {}
    if len(out) == 4:
        obs, reward, done, info = out
        return obs, float(reward or 0), bool(done), info if isinstance(info, dict) else {}
    raise TypeError(f"Unexpected env.step arity: {len(out)}")


def _as_text(obs: Any) -> str:
    if isinstance(obs, (list, tuple)) and obs:
        return str(obs[0])
    return str(obs or "")


def _as_command_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], (list, tuple)):
            return [str(x) for x in value[0]]
        return [str(x) for x in value]
    return [str(value)]


def _info_flag(info: Dict[str, Any], key: str) -> Any:
    val = info.get(key)
    if isinstance(val, (list, tuple)) and val and not isinstance(val[0], str):
        return val[0]
    return val


class AlfWorldTextEnv:
    """Single-game TextWorld wrapper used by the Hermes driver."""

    def __init__(self, env: Any, game: Dict[str, Any]):
        self._env = env
        self.game = game
        self.obs = ""
        self.admissible: List[str] = []
        self.won = False
        self.score = 0.0

    def reset(self) -> str:
        obs, info = _unpack_reset(self._env.reset())
        self._update(obs, info, score=0.0, done=False)
        return self.obs

    def step(self, command: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        obs, reward, done, info = _unpack_step(self._env.step(command))
        self._update(obs, info, score=reward, done=done)
        return self.obs, self.score, done, info

    def close(self) -> None:
        close = getattr(self._env, "close", None)
        if callable(close):
            close()

    def _update(self, obs: Any, info: Dict[str, Any], *, score: float, done: bool) -> None:
        self.obs = _as_text(obs)
        self.admissible = _as_command_list(info.get("admissible_commands"))
        won = _info_flag(info, "won")
        self.won = bool(won) if won is not None else False
        scored = _info_flag(info, "score")
        if scored is not None:
            try:
                self.score = float(scored)
            except (TypeError, ValueError):
                self.score = float(score)
        else:
            self.score = 1.0 if self.won else float(score)
        if done and self.won:
            self.score = max(self.score, 1.0)


def make_text_env(game: Dict[str, Any], *, max_episode_steps: int = 50) -> AlfWorldTextEnv:
    """Register and construct a TextWorld gym env for one ``game.tw-pddl``."""
    ensure_alfworld_on_path()
    import textworld
    import textworld.gym
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

    request_infos = textworld.EnvInfos(
        won=True,
        admissible_commands=True,
        score=True,
        extras=["gamefile"],
    )
    env_id = textworld.gym.register_game(
        game["game_file"],
        request_infos,
        max_episode_steps=int(max_episode_steps),
        wrappers=[AlfredDemangler(), AlfredInfos],
    )
    gym_env = textworld.gym.make(env_id)
    return AlfWorldTextEnv(gym_env, game)


def format_turn_prompt(
    observation: str,
    admissible: Iterable[str],
    *,
    first_turn: bool,
) -> str:
    commands = "\n".join(f"- {cmd}" for cmd in admissible) or "- look"
    if first_turn:
        return (
            "You are playing ALFWorld, a text household simulator.\n"
            "Each turn you get an observation and a list of admissible commands.\n"
            "Reply with EXACTLY one admissible command and nothing else "
            "(no quotes, no numbering, no explanation).\n"
            "Typical verbs: go to, open, close, take, move, put, inventory, "
            "examine, use, heat, cool, clean, slice, look.\n\n"
            f"{observation.strip()}\n\n"
            f"Admissible commands:\n{commands}"
        )
    return (
        f"{observation.strip()}\n\n"
        f"Admissible commands:\n{commands}\n\n"
        "Reply with EXACTLY one admissible command."
    )
