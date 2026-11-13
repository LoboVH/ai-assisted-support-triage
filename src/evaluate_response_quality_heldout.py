#!/usr/bin/env python3
"""Run the labelled held-out response-quality benchmark without LLM/API calls."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.evaluate_response_quality import evaluate_ticket

CASES_PATH = ROOT / "evaluation" / "response_quality_heldout_cases.json"
TICKETS_PATH = ROOT / "tickets.json"
LABELS_PATH = ROOT / "evaluation" / "response_quality_labels.json"
OUTPUT_PATH = ROOT / "output" / "response_quality_heldout_evaluation.json"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    cases_data = load_json(CASES_PATH)
    tickets_data = load_json(TICKETS_PATH)
    labels_data = load_json(LABELS_PATH)
    tickets = {item["ticket_id"]: item for item in tickets_data["tickets"]}
    labels = {item["ticket_id"]: item for item in labels_data["tickets"]}
    cases = cases_data["cases"]
    if len(cases) != cases_data.get("case_count"):
        raise ValueError("Case count does not match the dataset declaration.")
    ids = [case["case_id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate held-out case IDs found.")

    results = []
    dimension_matches = Counter()
    dimension_totals = Counter()
    group_stats = defaultdict(lambda: {"cases": 0, "expected_critical": 0,
                                       "critical_true_positive": 0,
                                       "critical_false_negative": 0,
                                       "critical_false_positive": 0,
                                       "critical_true_negative": 0,
                                       "review_matches": 0,
                                       "score_matches": 0})
    critical_counts = Counter()
    review_match_count = 0
    full_case_matches = 0

    for case in cases:
        ticket_id = case["ticket_id"]
        if ticket_id not in tickets or ticket_id not in labels:
            raise ValueError(f"Unknown ticket in case {case['case_id']}: {ticket_id}")
        result = evaluate_ticket(
            tickets[ticket_id],
            {"ticket_id": ticket_id, "draft": case["draft"], "status": "heldout_case"},
            labels[ticket_id],
        )
        expected = case["expected"]
        actual_critical = bool(result["critical_failures"])
        expected_critical = bool(expected["critical_failure"])
        actual_review = result["review_status"]
        expected_review = expected["review_status"]
        score_mismatches = {}
        for dimension, expected_score in expected["scores"].items():
            dimension_totals[dimension] += 1
            actual_score = result["scores"][dimension]
            if actual_score == expected_score:
                dimension_matches[dimension] += 1
            else:
                score_mismatches[dimension] = {
                    "expected": expected_score,
                    "actual": actual_score,
                }

        if actual_critical == expected_critical:
            critical_counts["matches"] += 1
        if expected_critical and actual_critical:
            critical_counts["true_positive"] += 1
        elif expected_critical and not actual_critical:
            critical_counts["false_negative"] += 1
        elif not expected_critical and actual_critical:
            critical_counts["false_positive"] += 1
        else:
            critical_counts["true_negative"] += 1

        review_match = actual_review == expected_review
        if review_match:
            review_match_count += 1
        group = group_stats[case["group"]]
        group["cases"] += 1
        group["expected_critical"] += int(expected_critical)
        group["review_matches"] += int(review_match)
        group["score_matches"] += int(not score_mismatches)
        if expected_critical and actual_critical:
            group["critical_true_positive"] += 1
        elif expected_critical and not actual_critical:
            group["critical_false_negative"] += 1
        elif not expected_critical and actual_critical:
            group["critical_false_positive"] += 1
        else:
            group["critical_true_negative"] += 1

        exact_match = (
            not score_mismatches
            and actual_critical == expected_critical
            and review_match
        )
        full_case_matches += int(exact_match)
        results.append({
            "case_id": case["case_id"],
            "group": case["group"],
            "ticket_id": ticket_id,
            "expected": expected,
            "actual": {
                "scores": result["scores"],
                "critical_failure": actual_critical,
                "critical_failures": result["critical_failures"],
                "review_status": actual_review,
                "review_reasons": result["review_reasons"],
            },
            "score_mismatches": score_mismatches,
            "critical_failure_match": actual_critical == expected_critical,
            "review_status_match": review_match,
            "exact_case_match": exact_match,
        })

    critical_positive = critical_counts["true_positive"]
    expected_positive = sum(1 for case in cases if case["expected"]["critical_failure"])
    sensitivity = round(critical_positive / expected_positive * 100, 2) if expected_positive else None
    expected_negative = len(cases) - expected_positive
    specificity = round(critical_counts["true_negative"] / expected_negative * 100, 2) if expected_negative else None
    report = {
        "evaluation_version": cases_data["evaluation_version"],
        "evaluation_type": "diagnostic_heldout_benchmark",
        "llm_api_calls": 0,
        "case_count": len(cases),
        "full_case_matches": full_case_matches,
        "full_case_agreement_percent": round(full_case_matches / len(cases) * 100, 2) if cases else 0,
        "critical_failure_detection": {
            "expected_critical_cases": expected_positive,
            "true_positive": critical_counts["true_positive"],
            "false_negative": critical_counts["false_negative"],
            "false_positive": critical_counts["false_positive"],
            "true_negative": critical_counts["true_negative"],
            "sensitivity_percent": sensitivity,
            "specificity_percent": specificity,
        },
        "review_status_agreement": {
            "matches": review_match_count,
            "total": len(cases),
            "percent": round(review_match_count / len(cases) * 100, 2) if cases else 0,
        },
        "score_dimension_agreement": {
            dimension: {
                "matches": dimension_matches[dimension],
                "total": dimension_totals[dimension],
                "percent": round(dimension_matches[dimension] / dimension_totals[dimension] * 100, 2) if dimension_totals[dimension] else 0,
            }
            for dimension in ("relevance", "factual_accuracy", "actionability", "policy_compliance")
        },
        "group_summary": {
            group_name: {
                **stats,
                "critical_recall_percent": round(stats["critical_true_positive"] / stats["expected_critical"] * 100, 2) if stats["expected_critical"] else None,
                "review_agreement_percent": round(stats["review_matches"] / stats["cases"] * 100, 2) if stats["cases"] else 0,
                "score_agreement_percent": round(stats["score_matches"] / stats["cases"] * 100, 2) if stats["cases"] else 0,
            }
            for group_name, stats in sorted(group_stats.items())
        },
        "cases": results,
        "interpretation_note": "Curated-label agreement is diagnostic, not proof of real-world accuracy. Review false negatives especially carefully. Do not tune on this set and continue calling it held-out; version it as a regression set if used for tuning.",
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Held-out cases evaluated: {len(cases)}")
    print(f"Full case agreement: {full_case_matches}/{len(cases)} ({report['full_case_agreement_percent']}%)")
    cf = report["critical_failure_detection"]
    print(f"Critical failure detection: {cf['true_positive']}/{cf['expected_critical_cases']} expected cases")
    print(f"  False negatives: {cf['false_negative']}; false positives: {cf['false_positive']}")
    print(f"Critical sensitivity: {cf['sensitivity_percent']}%; specificity: {cf['specificity_percent']}%")
    print(f"Review-status agreement: {review_match_count}/{len(cases)} ({report['review_status_agreement']['percent']}%)")
    print("\nScore-dimension agreement:")
    for dimension, stats in report["score_dimension_agreement"].items():
        print(f"  {dimension}: {stats['matches']}/{stats['total']} ({stats['percent']}%)")
    print("\nBy group:")
    for name, stats in report["group_summary"].items():
        print(f"  {name}: {stats['cases']} cases; score agreement {stats['score_agreement_percent']}%; review agreement {stats['review_agreement_percent']}%")
    mismatches = [item for item in results if not item["exact_case_match"]]
    print(f"\nCases with one or more mismatches: {len(mismatches)}")
    for item in mismatches:
        print(f"  {item['case_id']}: score mismatches={item['score_mismatches']}; critical expected={item['expected']['critical_failure']} actual={item['actual']['critical_failure']}; review expected={item['expected']['review_status']} actual={item['actual']['review_status']}")
    print(f"\nReport saved to: {OUTPUT_PATH.relative_to(ROOT)}")
    print("Diagnostic only: mismatches are findings, not a test-run failure. No LLM/API calls were made.")


if __name__ == "__main__":
    main()
