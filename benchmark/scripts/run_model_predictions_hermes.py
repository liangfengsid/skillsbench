"""
HLE (Humanity's Last Exam) — generate model predictions via in-process Hermes ``AIAgent``.

Lives under ``benchmark/scripts/`` with the rest of the Hermes-agent benchmark drivers.
Upstream OpenAI-based scripts remain in ``benchmark/hle/hle_eval/``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from datasets import load_dataset
from tqdm.asyncio import tqdm_asyncio

from hle_hermes_inprocess import hermes_chat

_SCRIPT = Path(__file__).resolve()


def _default_output_dir() -> Path:
    return _SCRIPT.parent.parent / "hle"


SYSTEM_PROMPT = (
    "Your response should be in the following format:\n"
    "Explanation: {your explanation for your answer choice}\n"
    "Answer: {your chosen answer}\n"
    "Confidence: {your confidence score between 0% and 100% for your answer}"
)


def format_prompt(question):
    question_text = question["question"]
    if question.get("image"):
        question_text = f"{question_text}\n\n[image_url]: {question['image']}"
    return question_text


def load_questions(args):
    if args.dataset_file:
        path = args.dataset_file
        if path.endswith(".jsonl"):
            questions = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    questions.append(json.loads(line))
            return questions
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("--dataset_file must contain a JSON array of questions")
        return data

    dataset = load_dataset(
        args.dataset,
        split="test",
        download_mode="reuse_cache_if_exists",
    ).to_dict()
    return [dict(zip(dataset.keys(), values)) for values in zip(*dataset.values())]


async def attempt_question(question, args):
    prompt = format_prompt(question)
    try:
        content = await asyncio.to_thread(
            hermes_chat,
            model=args.model,
            user_message=prompt,
            system_message=SYSTEM_PROMPT,
            max_iterations=1,
        )
        tokens = None
    except Exception as e:
        print("Error:", e)
        return None

    if content is None:
        return None

    return question["id"], content, tokens


async def attempt_all(questions, args):
    async def bound_func(question):
        async with semaphore:
            content = await attempt_question(question, args)
            return content

    semaphore = asyncio.Semaphore(args.num_workers)
    async with semaphore:
        tasks = [bound_func(q) for q in questions]
        results = await tqdm_asyncio.gather(*tasks)
    return results


def main(args):
    assert args.num_workers > 1, "num_workers must be 2 or greater"
    if not args.dataset_file and not args.dataset:
        raise SystemExit("Provide --dataset (HF id, e.g. cais/hle) or --dataset_file (.json/.jsonl).")

    questions = load_questions(args)

    if args.max_samples:
        questions = questions[: args.max_samples]

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_slug = (args.model or "default").replace("/", "_").strip() or "default"
    output_filepath = output_dir / f"hle_{model_slug}_hermes.json"

    if output_filepath.exists():
        with open(output_filepath, "r", encoding="utf-8") as f:
            predictions = json.load(f)
        questions = [q for q in questions if q["id"] not in predictions]
    else:
        predictions = {}

    results = asyncio.run(attempt_all(questions, args))

    for result in results:
        if result is None:
            continue
        unique_id, response, usage = result
        predictions[unique_id] = {"model": args.model, "response": response, "usage": usage}

    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=4)

    print(f"Wrote predictions to {output_filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="HLE HF Dataset (e.g. cais/hle)")
    parser.add_argument(
        "--dataset_file",
        type=str,
        default=None,
        help="Optional local dataset file (.json or .jsonl). If set, --dataset is ignored.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Hermes model id (provider/model). If empty, Hermes uses its configured default.",
    )
    parser.add_argument(
        "--max_completion_tokens",
        type=int,
        default=None,
        help="Limit completion tokens. Recommended to avoid model collapse.",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Unused (parity with upstream).")
    parser.add_argument("--num_workers", type=int, default=10, help="Async semaphore size.")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit evaluation to first N samples")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(_default_output_dir()),
        help="Directory for hle_<model>_hermes.json (default: benchmark/hle/).",
    )
    _args = parser.parse_args()
    main(_args)
