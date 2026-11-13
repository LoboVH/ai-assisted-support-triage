#!/usr/bin/env python3
"""Evaluate the fresh response-quality cases with the project's existing scoring rules.

This runner imports evaluate_ticket directly. It does not modify the main evaluator,
any regression datasets, or their reports. It writes a timestamped report under output/.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Running as `python src/evaluate_response_quality_fresh_fixed.py` puts src/ on sys.path.
from evaluate_response_quality import evaluate_ticket  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CASES_PATH = ROOT / "evaluation" / "response_quality_fresh_cases.json"
TICKETS_PATH = ROOT / "tickets.json"
LABELS_PATH = ROOT / "evaluation" / "response_quality_labels.json"
SCORE_KEYS = ("relevance", "factual_accuracy", "actionability", "policy_compliance")
VALID_REVIEW_STATUSES = {"automated", "needs_human_review"}


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"Required file not found: {path.relative_to(ROOT)}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path.relative_to(ROOT)}: {exc}")


def main() -> int:
    case_data = load_json(CASES_PATH)
    tickets_data = load_json(TICKETS_PATH)
    labels_data = load_json(LABELS_PATH)

    cases = case_data.get("cases") if isinstance(case_data, dict) else None
    if not isinstance(cases, list) or len(cases) != 40:
        print("Expected the supplied fresh benchmark to contain exactly 40 cases.", file=sys.stderr)
        return 2

    tickets = {x["ticket_id"]: x for x in tickets_data.get("tickets", [])}
    labels = {x["ticket_id"]: x for x in labels_data.get("tickets", [])}
    ids = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            print(f"Case at index {index} is not an object.", file=sys.stderr)
            return 2
        required = {"case_id", "group", "ticket_id", "draft", "expected"}
        missing = required - case.keys()
        if missing:
            print(f"Case at index {index} is missing fields: {sorted(missing)}", file=sys.stderr)
            return 2
        ids.append(case["case_id"])
        if not isinstance(case["case_id"], str) or not case["case_id"]:
            print(f"Case at index {index} has an invalid case_id.", file=sys.stderr)
            return 2
        if not isinstance(case["draft"], str) or not isinstance(case["ticket_id"], str):
            print(f"Case {case['case_id']} has invalid ticket_id/draft types.", file=sys.stderr)
            return 2
        if case["ticket_id"] not in tickets or case["ticket_id"] not in labels:
            print(f"Case {case['case_id']} references missing ticket/labels: {case['ticket_id']}", file=sys.stderr)
            return 2
        expected = case["expected"]
        if not isinstance(expected, dict) or not isinstance(expected.get("scores"), dict):
            print(f"Case {case['case_id']} has an invalid expected object.", file=sys.stderr)
            return 2
        if any(key not in expected["scores"] for key in SCORE_KEYS):
            print(f"Case {case['case_id']} is missing expected score dimensions.", file=sys.stderr)
            return 2
        if any(type(expected["scores"][key]) is not int or expected["scores"][key] not in (0, 1, 2) for key in SCORE_KEYS):
            print(f"Case {case['case_id']} has a score outside the 0/1/2 scale.", file=sys.stderr)
            return 2
        if type(expected.get("critical_failure")) is not bool:
            print(f"Case {case['case_id']} has invalid critical_failure label.", file=sys.stderr)
            return 2
        if expected.get("review_status") not in VALID_REVIEW_STATUSES:
            print(f"Case {case['case_id']} has invalid review_status label.", file=sys.stderr)
            return 2

    if len(ids) != len(set(ids)):
        print("Duplicate case_id in fresh benchmark.", file=sys.stderr)
        return 2

    results = []
    for case in cases:
        ticket_id = case["ticket_id"]
        actual = evaluate_ticket(
            tickets[ticket_id],
            {"ticket_id": ticket_id, "draft": case["draft"], "status": "generalization_probe"},
            labels[ticket_id],
        )
        expected = case["expected"]
        score_mismatches = {
            key: {"expected": expected["scores"][key], "actual": actual["scores"].get(key)}
            for key in SCORE_KEYS
            if actual["scores"].get(key) != expected["scores"][key]
        }
        critical_expected = expected["critical_failure"]
        critical_actual = bool(actual["critical_failures"])
        critical_match = critical_expected == critical_actual
        # Normalize equivalent clean-review labels for comparison only.
        # Preserve the original raw status in each result record.
        def _canonical_review_status(value):
            if value in {"automated", "automated_checks_passed"}:
                return "automated"
            if value in {"needs_human_review", "pending_review"}:
                return "needs_human_review"
            return value

        review_match = (
            _canonical_review_status(actual["review_status"])
            == _canonical_review_status(expected["review_status"])
        )
        case_match = not score_mismatches and critical_match and review_match
        results.append({
            "case_id": case["case_id"],
            "group": case["group"],
            "ticket_id": ticket_id,
            "draft": case["draft"],
            "expected": expected,
            "actual": {
                "scores": actual["scores"],
                "critical_failures": actual["critical_failures"],
                "review_status": actual["review_status"],
                "review_reasons": actual["review_reasons"],
            },
            "score_mismatches": score_mismatches,
            "critical_match": critical_match,
            "review_match": review_match,
            "case_match": case_match,
        })

    critical_cases = [r for r in results if r["expected"]["critical_failure"]]
    safe_cases = [r for r in results if not r["expected"]["critical_failure"]]
    tp = sum(bool(r["actual"]["critical_failures"]) for r in critical_cases)
    fn = len(critical_cases) - tp
    fp = sum(bool(r["actual"]["critical_failures"]) for r in safe_cases)
    tn = len(safe_cases) - fp
    score_agreement = {
        key: {
            "matched": sum(key not in r["score_mismatches"] for r in results),
            "total": len(results),
            "percentage": round(100 * sum(key not in r["score_mismatches"] for r in results) / len(results), 2),
        }
        for key in SCORE_KEYS
    }
    groups = defaultdict(list)
    for result in results:
        groups[result["group"]].append(result)
    group_report = {}
    for name, rows in sorted(groups.items()):
        group_report[name] = {
            "cases": len(rows),
            "full_case_agreement": sum(r["case_match"] for r in rows),
            "score_agreement_percentage": round(100 * sum(not r["score_mismatches"] for r in rows) / len(rows), 2),
            "critical_agreement_percentage": round(100 * sum(r["critical_match"] for r in rows) / len(rows), 2),
            "review_agreement_percentage": round(100 * sum(r["review_match"] for r in rows) / len(rows), 2),
        }
    review_matched = sum(r["review_match"] for r in results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = ROOT / "output" / f"response_quality_fresh_evaluation_{timestamp}.json"
    report = {
        "evaluation_version": case_data.get("evaluation_version", "fresh_benchmark_v1"),
        "description": case_data.get("description", "Fresh 40-case response-quality evaluation set."),
        "note": (
            "Fresh diagnostic set compared with curated expected labels. This is not an independently "
            "adjudicated third-party benchmark; inspect labels and mismatches before drawing broad accuracy claims."
        ),
        "case_count": len(results),
        "full_case_agreement": sum(r["case_match"] for r in results),
        "full_case_agreement_percentage": round(100 * sum(r["case_match"] for r in results) / len(results), 2),
        "score_dimension_agreement": score_agreement,
        "critical_failure_detection": {
            "true_positives": tp,
            "false_negatives": fn,
            "false_positives": fp,
            "true_negatives": tn,
            "expected_critical_cases": len(critical_cases),
            "critical_sensitivity_percentage": round(100 * tp / len(critical_cases), 2) if critical_cases else None,
            "specificity_percentage": round(100 * tn / len(safe_cases), 2) if safe_cases else None,
        },
        "review_status_agreement": {
            "matched": review_matched,
            "total": len(results),
            "percentage": round(100 * review_matched / len(results), 2),
        },
        "by_group": group_report,
        "mismatch_count": sum(not r["case_match"] for r in results),
        "mismatches": [r for r in results if not r["case_match"]],
        "results": results,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Fresh benchmark cases evaluated: {len(results)}")
    print(f"Full case agreement: {report['full_case_agreement']}/{len(results)} ({report['full_case_agreement_percentage']}%)")
    det = report["critical_failure_detection"]
    print(f"Critical failure detection: {det['true_positives']}/{det['expected_critical_cases']} expected cases")
    print(f"  False negatives: {det['false_negatives']}; false positives: {det['false_positives']}")
    print(f"  Critical sensitivity: {det['critical_sensitivity_percentage']}%; specificity: {det['specificity_percentage']}%")
    print(f"Review-status agreement: {review_matched}/{len(results)} ({report['review_status_agreement']['percentage']}%)")
    print("\nScore-dimension agreement:")
    for key, metrics in score_agreement.items():
        print(f"  {key}: {metrics['matched']}/{metrics['total']} ({metrics['percentage']}%)")
    print("\nBy group:")
    for name, metrics in group_report.items():
        print(f"  {name}: {metrics['cases']} cases; full agreement {metrics['full_case_agreement']}/{metrics['cases']}; score {metrics['score_agreement_percentage']}%; critical {metrics['critical_agreement_percentage']}%; review {metrics['review_agreement_percentage']}%")
    print(f"\nCases with one or more mismatches: {report['mismatch_count']}")
    for row in report["mismatches"]:
        print(f"  {row['case_id']}: score={row['score_mismatches']}; critical expected={row['expected']['critical_failure']} actual={bool(row['actual']['critical_failures'])}; review expected={row['expected']['review_status']} actual={row['actual']['review_status']}")
    print(f"\nReport saved to: {report_path.relative_to(ROOT)}")
    print("Offline evaluation only; no LLM/API calls were made.")
    print("Do not tune rules or relabel cases until the mismatches and expected labels have been reviewed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
