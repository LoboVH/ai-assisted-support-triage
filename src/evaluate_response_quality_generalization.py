#!/usr/bin/env python3
"""Diagnostic comparison of the frozen response-quality evaluator on fresh cases."""
import json
from collections import Counter, defaultdict
from pathlib import Path

from evaluate_response_quality import evaluate_ticket

ROOT = Path(__file__).resolve().parent.parent
CASES_PATH = ROOT / "evaluation" / "response_quality_generalization_cases.json"
TICKETS_PATH = ROOT / "tickets.json"
LABELS_PATH = ROOT / "evaluation" / "response_quality_labels.json"
OUTPUT_PATH = ROOT / "output" / "response_quality_generalization_probe.json"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    case_data = load(CASES_PATH)
    tickets_data = load(TICKETS_PATH)
    labels_data = load(LABELS_PATH)
    cases = case_data["cases"]
    tickets = {x["ticket_id"]: x for x in tickets_data["tickets"]}
    labels = {x["ticket_id"]: x for x in labels_data["tickets"]}

    ids = [c.get("case_id") for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate case_id in generalization set.")

    results = []
    for case in cases:
        ticket_id = case["ticket_id"]
        if ticket_id not in tickets or ticket_id not in labels:
            raise ValueError(f"Missing ticket or labels for {ticket_id}")
        actual = evaluate_ticket(
            tickets[ticket_id],
            {"ticket_id": ticket_id, "draft": case["draft"], "status": "generalization_probe"},
            labels[ticket_id],
        )
        expected = case["expected"]
        score_mismatches = {
            key: {"expected": target, "actual": actual["scores"].get(key)}
            for key, target in expected.get("scores", {}).items()
            if actual["scores"].get(key) != target
        }
        critical_expected = bool(expected["critical_failure"])
        critical_actual = bool(actual["critical_failures"])
        critical_match = critical_expected == critical_actual
        review_match = actual["review_status"] == expected["review_status"]
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
    score_keys = ("relevance", "factual_accuracy", "actionability", "policy_compliance")
    score_agreement = {
        key: {
            "matched": sum(key not in r["score_mismatches"] for r in results),
            "total": len(results),
            "percentage": round(100 * sum(key not in r["score_mismatches"] for r in results) / len(results), 2) if results else 0,
        }
        for key in score_keys
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

    report = {
        "evaluation_version": case_data["evaluation_version"],
        "description": case_data["description"],
        "note": "Diagnostic generalization probe, not a blind third-party benchmark. Expected labels are curated and should be reviewed if domain policy changes.",
        "case_count": len(results),
        "full_case_agreement": sum(r["case_match"] for r in results),
        "full_case_agreement_percentage": round(100 * sum(r["case_match"] for r in results) / len(results), 2) if results else 0,
        "score_dimension_agreement": score_agreement,
        "critical_failure_detection": {
            "true_positives": tp, "false_negatives": fn,
            "false_positives": fp, "true_negatives": tn,
            "expected_critical_cases": len(critical_cases),
            "critical_sensitivity_percentage": round(100 * tp / len(critical_cases), 2) if critical_cases else None,
            "specificity_percentage": round(100 * tn / len(safe_cases), 2) if safe_cases else None,
        },
        "review_status_agreement": {
            "matched": sum(r["review_match"] for r in results),
            "total": len(results),
            "percentage": round(100 * sum(r["review_match"] for r in results) / len(results), 2) if results else 0,
        },
        "by_group": group_report,
        "mismatch_count": sum(not r["case_match"] for r in results),
        "mismatches": [r for r in results if not r["case_match"]],
        "results": results,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Generalization-probe cases evaluated: {len(results)}")
    print(f"Full case agreement: {report['full_case_agreement']}/{len(results)} ({report['full_case_agreement_percentage']}%)")
    det = report["critical_failure_detection"]
    print(f"Critical failure detection: {det['true_positives']}/{det['expected_critical_cases']} expected cases")
    print(f"  False negatives: {det['false_negatives']}; false positives: {det['false_positives']}")
    print(f"  Critical sensitivity: {det['critical_sensitivity_percentage']}%; specificity: {det['specificity_percentage']}%")
    print(f"Review-status agreement: {report['review_status_agreement']['matched']}/{len(results)} ({report['review_status_agreement']['percentage']}%)")
    print("\nScore-dimension agreement:")
    for key, metrics in score_agreement.items():
        print(f"  {key}: {metrics['matched']}/{metrics['total']} ({metrics['percentage']}%)")
    print("\nBy group:")
    for name, metrics in group_report.items():
        print(f"  {name}: {metrics['cases']} cases; full agreement {metrics['full_case_agreement']}/{metrics['cases']}; score {metrics['score_agreement_percentage']}%; critical {metrics['critical_agreement_percentage']}%; review {metrics['review_agreement_percentage']}%")
    print(f"\nCases with one or more mismatches: {report['mismatch_count']}")
    for row in report["mismatches"]:
        print(f"  {row['case_id']}: score={row['score_mismatches']}; critical expected={row['expected']['critical_failure']} actual={bool(row['actual']['critical_failures'])}; review expected={row['expected']['review_status']} actual={row['actual']['review_status']}")
    print(f"\nReport saved to: {OUTPUT_PATH.relative_to(ROOT)}")
    print("Diagnostic only. No LLM/API calls were made.")


if __name__ == "__main__":
    main()
