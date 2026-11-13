#!/usr/bin/env python3
"""Run a fresh response-quality set using the project's current evaluator, without changing source datasets."""
from __future__ import annotations
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path.cwd()
FRESH_CASES = ROOT / "evaluation" / "response_quality_fresh_cases.json"
EVALUATOR = ROOT / "src" / "evaluate_response_quality_generalization.py"
GENERALIZATION_INPUT = "evaluation/response_quality_generalization_cases.json"
GENERALIZATION_REPORT = "output/response_quality_generalization_probe.json"


def main() -> int:
    if not FRESH_CASES.is_file():
        print(f"Missing fresh dataset: {FRESH_CASES}", file=sys.stderr)
        return 2
    if not EVALUATOR.is_file():
        print(f"Missing evaluator: {EVALUATOR}", file=sys.stderr)
        return 2

    try:
        dataset = json.loads(FRESH_CASES.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Could not parse fresh dataset: {exc}", file=sys.stderr)
        return 2

    cases = dataset.get("cases") if isinstance(dataset, dict) else None
    if not isinstance(cases, list) or len(cases) < 30:
        print("Fresh dataset must contain a 'cases' list of at least 30 cases.", file=sys.stderr)
        return 2
    ids = [c.get("case_id") for c in cases if isinstance(c, dict)]
    if len(ids) != len(cases) or any(not isinstance(x, str) for x in ids) or len(set(ids)) != len(ids):
        print("Every fresh case must be an object with a unique string case_id.", file=sys.stderr)
        return 2
    required = {"case_id", "ticket_id", "draft", "expected"}
    for index, case in enumerate(cases):
        if not required.issubset(case):
            print(f"Case {index} missing required fields: {sorted(required - set(case))}", file=sys.stderr)
            return 2
        expected = case.get("expected", {})
        scores = expected.get("scores", {}) if isinstance(expected, dict) else {}
        if set(("relevance", "factual_accuracy", "actionability", "policy_compliance")) - set(scores):
            print(f"Case {case.get('case_id')} has an incomplete score rubric.", file=sys.stderr)
            return 2

    source = EVALUATOR.read_text(encoding="utf-8")
    if GENERALIZATION_INPUT not in source or GENERALIZATION_REPORT not in source:
        print(
            "Evaluator paths differ from the expected paths. No files were changed. "
            "Inspect src/evaluate_response_quality_generalization.py before adapting this runner.",
            file=sys.stderr,
        )
        return 2

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    fresh_report = output_dir / f"response_quality_fresh_evaluation_{timestamp}.json"
    fresh_log = output_dir / f"response_quality_fresh_evaluation_{timestamp}.txt"
    temp_evaluator = ROOT / "src" / f"_evaluate_response_quality_fresh_{timestamp}.py"

    # Redirect only the evaluator's two path literals in a temporary source copy.
    # The current regression dataset and its report are never overwritten.
    redirected = source.replace(GENERALIZATION_INPUT, "evaluation/response_quality_fresh_cases.json")
    redirected = redirected.replace(GENERALIZATION_REPORT, str(fresh_report.relative_to(ROOT)))
    temp_evaluator.write_text(redirected, encoding="utf-8")
    try:
        completed = subprocess.run(
            [sys.executable, str(temp_evaluator.relative_to(ROOT))],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
    finally:
        temp_evaluator.unlink(missing_ok=True)

    log_text = completed.stdout
    if completed.stderr:
        log_text += "\n--- STDERR ---\n" + completed.stderr
    fresh_log.write_text(log_text, encoding="utf-8")
    print(completed.stdout, end="" if completed.stdout.endswith("\n") or not completed.stdout else "\n")
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="" if completed.stderr.endswith("\n") else "\n")

    if not fresh_report.exists():
        print("Evaluator did not create the fresh report. See " + str(fresh_log), file=sys.stderr)
        return completed.returncode or 1

    try:
        report = json.loads(fresh_report.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Fresh report is not valid JSON: {exc}. See {fresh_log}", file=sys.stderr)
        return 1

    print(f"Fresh dataset cases: {len(cases)}")
    print(f"Fresh report: {fresh_report.relative_to(ROOT)}")
    print(f"Console log: {fresh_log.relative_to(ROOT)}")
    print("Original regression dataset/report were not modified.")
    print("Important: this is the first run on these cases; do not tune against them until you have reviewed the mismatch report.")
    return completed.returncode

if __name__ == "__main__":
    raise SystemExit(main())
