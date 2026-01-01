#!/usr/bin/env python3
"""
Evaluate multiple OpenAI models against a CSV of questions + acceptable answers, using the Responses API.

Input CSV format (header required):
question,answer1,answer2,...,answerN

Run:
  export OPENAI_API_KEY="..."
  python3 src/ask-easy-questions.py data/questions.csv gpt-4o gpt-4o-mini gpt-5

Outputs:
  results.csv  -> model,question,answer,correct
  summary.csv  -> model,total_correct,skipped,total_questions,accuracy

Notes:
  - Any per-question API/parse error is logged to stderr and counts as "skipped".
  - Skipped questions do NOT affect totals, results.csv, or accuracy.
  - The script adaptively retries requests if a model rejects certain parameters
    (e.g., temperature), and caches that per-model capability for future calls.
  - Error logs do NOT print question text.
"""

import csv
import os
import sys
import time
import re
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

# ----------------------------
# Constants (edit as desired)
# ----------------------------
SYSTEM_INSTRUCTION = "Return only the exact answer with no explanation."
MAX_OUTPUT_TOKENS = 1024
TEMPERATURE = 0.0
SLEEP_BETWEEN_CALLS_SECONDS = 0.25

RESULTS_PATH = "results/results.csv"
SUMMARY_PATH = "results/summary.csv"

# Per-model learned capabilities (only stores what we discover)
# Example: {"gpt-5": {"temperature": False}}
MODEL_CAPS: Dict[str, Dict[str, bool]] = {}


def log_error(msg: str) -> None:
    """Log errors only (no question text)."""
    print(msg, file=sys.stderr)


def normalize(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s)  # remove punctuation
    return " ".join(s.split())     # normalize whitespace


def load_questions(csv_path: str) -> List[Tuple[str, List[str]]]:
    rows: List[Tuple[str, List[str]]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)

        if not header or len(header) < 2:
            raise ValueError("CSV must have header: question,answer1,... or category,question,answer1,...")

        # Detect whether the first column is category
        header_lc = [h.strip().lower() for h in header]

        if header_lc[0] == "question":
            question_idx = 0
            answers_start_idx = 1
        elif header_lc[0] == "category" and len(header_lc) >= 3 and header_lc[1] == "question":
            question_idx = 1
            answers_start_idx = 2
        else:
            raise ValueError(
                "CSV must have header: question,answer1,... "
                "or category,question,answer1,..."
            )

        for line_no, row in enumerate(reader, start=2):
            if not row:
                continue

            # Safely extract question
            q = row[question_idx].strip() if len(row) > question_idx else ""
            if not q:
                continue

            # Extract answers (ignore category if present)
            answers = [
                a.strip()
                for a in row[answers_start_idx:]
                if a and a.strip()
            ]

            if not answers:
                raise ValueError(f"Row {line_no}: missing acceptable answers.")

            rows.append((q, answers))

    return rows


def _is_unsupported_param_error(e: Exception) -> Optional[str]:
    """
    Return the unsupported parameter name if this looks like an OpenAI 400
    complaining about an unsupported parameter, otherwise None.
    """
    # If SDK provides structured param
    param = getattr(e, "param", None)
    if param:
        return str(param)

    msg = str(e)

    # Common message format: Unsupported parameter: 'temperature'
    m = re.search(r"Unsupported parameter:\s*'([^']+)'", msg)
    if m:
        return m.group(1)

    # Sometimes the error string includes: 'param': 'temperature'
    m = re.search(r"'param':\s*'([^']+)'", msg)
    if m:
        return m.group(1)

    return None


def create_response_adaptive(
    client: OpenAI,
    model: str,
    question: str,
    instructions: str,
    max_output_tokens: int,
    temperature: float,
) -> Any:
    """
    Call responses.create with best-effort params.
    If model rejects a param as unsupported, retry without it and remember per model.
    """
    caps = MODEL_CAPS.setdefault(model, {})

    def build_kwargs() -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": model,
            "instructions": instructions,
            "input": question,
        }
        # Only include params we currently believe are supported.
        if caps.get("max_output_tokens", True):
            kwargs["max_output_tokens"] = max_output_tokens
        if caps.get("temperature", True):
            kwargs["temperature"] = temperature
        return kwargs

    kwargs = build_kwargs()

    # Retry a few times, removing unsupported params one by one
    for attempt in range(3):
        try:
            return client.responses.create(**kwargs)
        except Exception as e:
            unsupported = _is_unsupported_param_error(e)
            if not unsupported:
                raise  # Not a capability issue; caller will log/skip

            # Learn and retry
            caps[unsupported] = False
            if unsupported in kwargs:
                del kwargs[unsupported]

            # If we've removed everything removable and still failing, bubble up
            if attempt == 2:
                raise

    # Shouldn't reach here
    raise RuntimeError("Adaptive request retry loop exhausted unexpectedly.")


def main() -> int:
    if len(sys.argv) < 3:
        print(
            "Usage: python3 src/ask-easy-questions.py <questions.csv> <model1> [model2 ...]",
            file=sys.stderr,
        )
        return 2

    csv_path = sys.argv[1]
    models = sys.argv[2:]

    if not os.environ.get("OPENAI_API_KEY"):
        print("Missing OPENAI_API_KEY environment variable.", file=sys.stderr)
        return 2

    # Ensure output directories exist
    os.makedirs(os.path.dirname(RESULTS_PATH) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(SUMMARY_PATH) or ".", exist_ok=True)

    client = OpenAI()

    try:
        items = load_questions(csv_path)
    except Exception as e:
        log_error(f"[FATAL] Failed to load questions CSV: {type(e).__name__}: {e}")
        return 2

    print(f"Loaded {len(items)} questions")

    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as rf, open(
        SUMMARY_PATH, "w", newline="", encoding="utf-8"
    ) as sf:
        results_writer = csv.writer(rf)
        summary_writer = csv.writer(sf)

        results_writer.writerow(["model", "question", "answer", "correct"])
        summary_writer.writerow(["model", "total_correct", "skipped", "total_questions", "accuracy"])

        for model in models:
            total = 0        # excludes skipped
            correct = 0
            skipped = 0

            for idx, (question, accepted_answers) in enumerate(items, start=1):
                try:
                    resp = create_response_adaptive(
                        client=client,
                        model=model,
                        question=question,
                        instructions=SYSTEM_INSTRUCTION,
                        max_output_tokens=MAX_OUTPUT_TOKENS,
                        temperature=TEMPERATURE,
                    )
                    
                    answer_text = (getattr(resp, "output_text", "") or "").strip()
                    if not answer_text:
                        raise RuntimeError("Empty response output_text")

                    accepted_norm = {normalize(a) for a in accepted_answers}
                    is_correct = normalize(answer_text) in accepted_norm

                    total += 1
                    if is_correct:
                        correct += 1

                    results_writer.writerow(
                        [model, question, answer_text, "correct" if is_correct else "incorrect"]
                    )

                except Exception as e:
                    skipped += 1
                    # No question text printed
                    log_error(f"[ERROR] model={model} q#{idx} skipped: {type(e).__name__}: {e}")
                    continue

                if SLEEP_BETWEEN_CALLS_SECONDS > 0:
                    time.sleep(SLEEP_BETWEEN_CALLS_SECONDS)

            accuracy = (correct / total) if total else 0.0
            summary_writer.writerow([model, correct, skipped, total, f"{accuracy:.6f}"])

    print(f"Wrote {RESULTS_PATH} and {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
