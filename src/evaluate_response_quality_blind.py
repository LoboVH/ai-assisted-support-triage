#!/usr/bin/env python3
"""Run the frozen response-quality evaluator on the unseen policy-rubric dataset.

This script imports evaluate_ticket directly. It does not modify the evaluator,
source benchmark labels, tickets, or existing reports. Run once before tuning;
after observing results, treat this dataset as a regression set.
"""
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from evaluate_response_quality import evaluate_ticket

ROOT = Path(__file__).resolve().parent.parent
CASES_PATH = ROOT / "evaluation" / "response_quality_blind_cases.json"
TICKETS_PATH = ROOT / "tickets.json"
LABELS_PATH = ROOT / "evaluation" / "response_quality_labels.json"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_review_status(value):
    if value in {"automated", "automated_checks_passed"}:
        return "automated"
    if value in {"needs_human_review", "pending_review"}:
        return "needs_human_review"
    return value


def main():
    case_data = load(CASES_PATH)
    tickets_data = load(TICKETS_PATH)
    labels_data = load(LABELS_PATH)
    cases = case_data["cases"]
    tickets = {row["ticket_id"]: row for row in tickets_data["tickets"]}
    labels = {row["ticket_id"]: row for row in labels_data["tickets"]}

    ids = [case.get("case_id") for case in cases]
    if len(cases) != 40 or len(ids) != len(set(ids)):
        raise ValueError(f"Expected 40 unique cases; got {len(cases)} cases and {len(set(ids))} IDs.")
    expected_ticket_counts = {"T-1001": 10, "T-1002": 10, "T-1003": 10, "T-1004": 10}
    actual_ticket_counts = {}
    for case in cases:
        tid = case["ticket_id"]
        actual_ticket_counts[tid] = actual_ticket_counts.get(tid, 0) + 1
        if tid not in tickets or tid not in labels:
            raise ValueError(f"Missing ticket or response-quality labels for {tid}")
        if set(case["expected"]["scores"]) != {"relevance", "factual_accuracy", "actionability", "policy_compliance"}:
            raise ValueError(f"Invalid dimension labels for {case['case_id']}")
        if case["expected"]["review_status"] not in {"automated", "needs_human_review"}:
            raise ValueError(f"Invalid expected review status for {case['case_id']}")
    if actual_ticket_counts != expected_ticket_counts:
        raise ValueError(f"Unexpected ticket distribution: {actual_ticket_counts}")

    results = []
    score_keys = ("relevance", "factual_accuracy", "actionability", "policy_compliance")
    for case in cases:
        tid = case["ticket_id"]
        actual = evaluate_ticket(
            tickets[tid],
            {"ticket_id": tid, "draft": case["draft"], "status": "blind_benchmark"},
            labels[tid],
        )
        expected = case["expected"]
        score_mismatches = {
            key: {"expected": expected["scores"][key], "actual": actual["scores"].get(key)}
            for key in score_keys if actual["scores"].get(key) != expected["scores"][key]
        }
        critical_actual = bool(actual["critical_failures"])
        critical_match = bool(expected["critical_failure"]) == critical_actual
        review_match = canonical_review_status(actual["review_status"]) == canonical_review_status(expected["review_status"])
        case_match = not score_mismatches and critical_match and review_match
        results.append({
            "case_id": case["case_id"], "group": case["group"], "ticket_id": tid,
            "draft": case["draft"], "label_rationale": case["label_rationale"],
            "expected": expected,
            "actual": {"scores": actual["scores"], "critical_failures": actual["critical_failures"],
                       "review_status": actual["review_status"], "review_reasons": actual["review_reasons"]},
            "score_mismatches": score_mismatches, "critical_match": critical_match,
            "review_match": review_match, "case_match": case_match,
        })

    critical_rows = [r for r in results if r["expected"]["critical_failure"]]
    safe_rows = [r for r in results if not r["expected"]["critical_failure"]]
    tp = sum(bool(r["actual"]["critical_failures"]) for r in critical_rows)
    fn = len(critical_rows) - tp
    fp = sum(bool(r["actual"]["critical_failures"]) for r in safe_rows)
    tn = len(safe_rows) - fp
    dimension_report = {
        key: {"matched": sum(key not in row["score_mismatches"] for row in results),
              "total": len(results),
              "percentage": round(100 * sum(key not in row["score_mismatches"] for row in results) / len(results), 2)}
        for key in score_keys
    }
    groups = defaultdict(list)
    by_ticket = defaultdict(list)
    for row in results:
        groups[row["group"]].append(row)
        by_ticket[row["ticket_id"]].append(row)
    group_report = {
        name: {"cases": len(rows), "full_case_agreement": sum(r["case_match"] for r in rows),
               "score_agreement_percentage": round(100 * sum(not r["score_mismatches"] for r in rows) / len(rows), 2),
               "critical_agreement_percentage": round(100 * sum(r["critical_match"] for r in rows) / len(rows), 2),
               "review_agreement_percentage": round(100 * sum(r["review_match"] for r in rows) / len(rows), 2)}
        for name, rows in sorted(groups.items())
    }
    ticket_report = {
        tid: {"cases": len(rows), "full_case_agreement": sum(r["case_match"] for r in rows),
              "critical_expected": sum(bool(r["expected"]["critical_failure"]) for r in rows),
              "critical_detected": sum(bool(r["actual"]["critical_failures"]) for r in rows)}
        for tid, rows in sorted(by_ticket.items())
    }
    report = {
        "evaluation_version": case_data["evaluation_version"],
        "description": case_data["description"],
        "rubric_version": case_data["rubric_version"],
        "note": "Newly authored policy-rubric benchmark; not externally adjudicated. Do not tune on this set and continue calling it blind.",
        "case_count": len(results),
        "full_case_agreement": sum(r["case_match"] for r in results),
        "full_case_agreement_percentage": round(100 * sum(r["case_match"] for r in results) / len(results), 2),
        "score_dimension_agreement": dimension_report,
        "critical_failure_detection": {"true_positives": tp, "false_negatives": fn,
            "false_positives": fp, "true_negatives": tn, "expected_critical_cases": len(critical_rows),
            "expected_safe_cases": len(safe_rows),
            "critical_sensitivity_percentage": round(100 * tp / len(critical_rows), 2) if critical_rows else None,
            "specificity_percentage": round(100 * tn / len(safe_rows), 2) if safe_rows else None},
        "review_status_agreement": {"matched": sum(r["review_match"] for r in results), "total": len(results),
            "percentage": round(100 * sum(r["review_match"] for r in results) / len(results), 2)},
        "by_ticket": ticket_report, "by_group": group_report,
        "mismatch_count": sum(not r["case_match"] for r in results),
        "mismatches": [r for r in results if not r["case_match"]],
        "results": results,
    }
    output_dir = ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"response_quality_blind_evaluation_{stamp}.json"
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Unseen benchmark cases evaluated: {len(results)}")
    print(f"Full case agreement: {report['full_case_agreement']}/40 ({report['full_case_agreement_percentage']}%)")
    det = report["critical_failure_detection"]
    print(f"Critical failure detection: {det['true_positives']}/{det['expected_critical_cases']} expected critical cases")
    print(f"  False negatives: {fn}; false positives: {fp}")
    print(f"  Critical sensitivity: {det['critical_sensitivity_percentage']}%; specificity: {det['specificity_percentage']}%")
    rev = report["review_status_agreement"]
    print(f"Review-status agreement: {rev['matched']}/{rev['total']} ({rev['percentage']}%)")
    print("\nScore-dimension agreement:")
    for key, metrics in dimension_report.items():
        print(f"  {key}: {metrics['matched']}/{metrics['total']} ({metrics['percentage']}%)")
    print("\nBy ticket:")
    for tid, metrics in ticket_report.items():
        print(f"  {tid}: {metrics['full_case_agreement']}/{metrics['cases']} full agreement; critical {metrics['critical_detected']}/{metrics['critical_expected']} detected")
    print(f"\nCases with one or more mismatches: {report['mismatch_count']}")
    for row in report["mismatches"]:
        print(f"  {row['case_id']}: score={row['score_mismatches']}; critical expected={row['expected']['critical_failure']} actual={bool(row['actual']['critical_failures'])}; review expected={row['expected']['review_status']} actual={row['actual']['review_status']}")
    print(f"\nReport saved to: {output_path.relative_to(ROOT)}")
    print("Offline evaluation only; no LLM/API calls were made.")
    print("After the first run, preserve this benchmark as a regression set; do not tune against it and continue calling it blind.")

if __name__ == "__main__":
    main()
