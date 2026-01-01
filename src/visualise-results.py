#!/usr/bin/env python3
"""
Visualize one or more OpenAI eval results.csv files, with category support from questions.csv.

Expected results.csv format:
model,question,answer,correct

Expected questions.csv format:
category,question,answer1,...

Charts produced:
  - accuracy-by-model.png
  - accuracy-by-category.png
  - question-difficulty.png
  - run-to-run-stability.png (only if multiple runs)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt

RESULTS_COLS = ["model", "question", "answer", "correct"]


def eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def sanitize_correct(v: str) -> int:
    v = (v or "").strip().lower()
    if v in ("correct", "true", "yes", "1"):
        return 1
    if v in ("incorrect", "false", "no", "0"):
        return 0
    raise ValueError(f"Unexpected correct value: {v!r}")


def load_questions_map(path: str) -> Tuple[Dict[str, int], Dict[str, str]]:
    df = pd.read_csv(path)
    if "question" not in df.columns or "category" not in df.columns:
        raise ValueError("questions.csv must contain 'category' and 'question' columns")

    df["question"] = df["question"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip().replace("", "UNCATEGORIZED")

    qnum = {}
    qcat = {}

    for i, row in enumerate(df.itertuples(index=False), start=1):
        if row.question not in qnum:
            qnum[row.question] = i
            qcat[row.question] = row.category

    return qnum, qcat


def load_results(paths: List[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        missing = [c for c in RESULTS_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{p} missing columns {missing}")

        df = df[RESULTS_COLS].copy()
        df["run"] = Path(p).stem
        df["correct_bin"] = df["correct"].apply(sanitize_correct)
        df["model"] = df["model"].astype(str).str.strip()
        df["question"] = df["question"].astype(str).str.strip()
        frames.append(df)

    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["run", "model", "question"], keep="last")
    )


def attach_question_meta(df, qnum, qcat):
    df = df.copy()
    df["qnum"] = df["question"].map(qnum)
    df["category"] = df["question"].map(qcat).fillna("UNKNOWN")
    return df


def savefig(outdir: Path, name: str):
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / name
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    print(f"Wrote {path}")


# ---------------- Charts ----------------

def chart_accuracy_by_model(df, outdir, prefix):
    acc = df.groupby("model")["correct_bin"].mean().sort_values(ascending=False)

    plt.figure(figsize=(8, 4))
    acc.plot(kind="bar")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title("Accuracy by model")
    savefig(outdir, f"{prefix}-accuracy-by-model.png")


def chart_accuracy_by_category(df, outdir, prefix):
    acc = df.groupby("category")["correct_bin"].mean().sort_values(ascending=False)

    plt.figure(figsize=(8, 4))
    acc.plot(kind="bar")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title("Accuracy by category (all models combined)")
    savefig(outdir, f"{prefix}-accuracy-by-category.png")


def chart_question_accuracy(df, outdir, prefix, top_n):
    grp = (
        df.groupby(["question", "qnum", "category"])["correct_bin"]
        .mean()
        .reset_index()
    )

    # Order by question number, not by accuracy
    ordered = grp.sort_values("qnum").head(top_n)

    labels = [
        f"Q{int(r.qnum):02d} {r.category}" if pd.notna(r.qnum) else "Q??"
        for r in ordered.itertuples()
    ]

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(ordered)), ordered["correct_bin"])
    plt.xticks(range(len(ordered)), labels, rotation=60, ha="right")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title("Question accuracy (lower = harder)")
    savefig(outdir, f"{prefix}-question-accuracy.png")


def chart_run_to_run_stability(df, outdir, prefix):
    if df["run"].nunique() < 2:
        return

    pivot = (
        df.groupby(["run", "model"])["correct_bin"]
        .mean()
        .reset_index()
        .pivot(index="run", columns="model", values="correct_bin")
    )

    plt.figure(figsize=(9, 4))
    for model in pivot.columns:
        plt.plot(pivot.index, pivot[model], marker="o", label=model)

    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title("Run-to-run accuracy stability")
    plt.xticks(rotation=45, ha="right")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    savefig(outdir, f"{prefix}-run-to-run-stability.png")


# ---------------- Main ----------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_files", nargs="+")
    ap.add_argument("--questions", default="data/questions.csv")
    ap.add_argument("--outdir", default="results/plots")
    ap.add_argument("--prefix", default="eval")
    ap.add_argument("--top-hardest", type=int, default=30)
    args = ap.parse_args()

    try:
        qnum, qcat = load_questions_map(args.questions)
        df = load_results(args.results_files)
        df = attach_question_meta(df, qnum, qcat)
    except Exception as e:
        eprint(f"[FATAL] {e}")
        return 2

    outdir = Path(args.outdir)

    chart_accuracy_by_model(df, outdir, args.prefix)
    chart_accuracy_by_category(df, outdir, args.prefix)
    chart_question_accuracy(df, outdir, args.prefix, args.top_hardest)
    chart_run_to_run_stability(df, outdir, args.prefix)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
