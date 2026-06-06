from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import load_corpus, load_eval_cases
from .reporting import save_markdown_report
from .runner import DEFAULT_OUTPUT_DIR, run_eval, save_eval_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LLM-EvalBench RAG / Agent evaluation.")
    parser.add_argument("--cases", type=Path, default=None, help="Path to eval cases JSON.")
    parser.add_argument("--corpus", type=Path, default=None, help="Path to corpus JSON.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for JSON and Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print aggregate metrics as JSON.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cases = load_eval_cases(args.cases) if args.cases else None
    corpus = load_corpus(args.corpus) if args.corpus else None
    run = run_eval(cases=cases, corpus=corpus)
    json_path = save_eval_run(run, args.output_dir)
    md_path = save_markdown_report(run, args.output_dir)

    if args.json:
        print(json.dumps([item.model_dump() for item in run.aggregate_metrics], indent=2, ensure_ascii=False))
    else:
        print(f"LLM-EvalBench run complete: {run.run_id}")
        print(f"JSON report: {json_path}")
        print(f"Markdown report: {md_path}")


if __name__ == "__main__":
    main()
