#!/usr/bin/env python3
"""Evaluate critical-failure detection on challenge-v1 without modifying labels."""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from evaluate_response_quality import evaluate_ticket

ROOT = Path(__file__).resolve().parent.parent
CASES_PATH = ROOT / "evaluation" / "response_quality_challenge_v1.json"
TICKETS_PATH = ROOT / "tickets.json"
LABELS_PATH = ROOT / "evaluation" / "response_quality_labels.json"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    dataset = load(CASES_PATH)
    tickets_data = load(TICKETS_PATH)
    labels_data = load(LABELS_PATH)

    cases = dataset["cases"]
    tickets = {row["ticket_id"]: row for row in tickets_data["tickets"]}
    labels = {row["ticket_id"]: row for row in labels_data["tickets"]}

    ids = [case.get("case_id") for case in cases]
    if len(cases) != 32 or len(ids) != len(set(ids)):
        raise ValueError(f"Expected 32 cases with unique IDs; got {len(cases)}.")

    expected_counts = {
        "T-1001": 8, "T-1002": 8, "T-1003": 8, "T-1004": 8
    }
    ticket_counts = Counter(case["ticket_id"] for case in cases)
    if dict(ticket_counts) != expected_counts:
        raise ValueError(f"Unexpected ticket distribution: {dict(ticket_counts)}")

    for tid in expected_counts:
        if tid not in tickets or tid not in labels:
            raise ValueError(f"Missing ticket or response-quality label for {tid}")

        rows = [case for case in cases if case["ticket_id"] == tid]
        outcomes = Counter(case["expected_critical_failure"] for case in rows)
        if outcomes[True] != 4 or outcomes[False] != 4:
            raise ValueError(f"{tid} must contain 4 critical and 4 safe cases: {outcomes}")

    results = []
    for case in cases:
        tid = case["ticket_id"]
        actual = evaluate_ticket(
            tickets[tid],
            {
                "ticket_id": tid,
                "draft": case["draft"],
                "status": "challenge_v1",
            },
            labels[tid],
        )
        expected_critical = case["expected_critical_failure"]
        actual_critical = bool(actual["critical_failures"])

        results.append({
            "case_id": case["case_id"],
            "ticket_id": tid,
            "draft": case["draft"],
            "label_rationale": case["label_rationale"],
            "expected_critical_failure": expected_critical,
            "actual_critical_failure": actual_critical,
            "matched": expected_critical == actual_critical,
            "critical_failures": actual["critical_failures"],
        })

    tp = sum(
        r["expected_critical_failure"] and r["actual_critical_failure"]
        for r in results
    )
    fn = sum(
        r["expected_critical_failure"] and not r["actual_critical_failure"]
        for r in results
    )
    fp = sum(
        not r["expected_critical_failure"] and r["actual_critical_failure"]
        for r in results
    )
    tn = sum(
        not r["expected_critical_failure"] and not r["actual_critical_failure"]
        for r in results
    )

    critical_total = tp + fn
    safe_total = tn + fp
    sensitivity = round(100 * tp / critical_total, 2) if critical_total else None
    specificity = round(100 * tn / safe_total, 2) if safe_total else None
    matched = sum(r["matched"] for r in results)

    by_ticket = {}
    for tid in sorted(expected_counts):
        rows = [r for r in results if r["ticket_id"] == tid]
        by_ticket[tid] = {
            "cases": len(rows),
            "matched": sum(r["matched"] for r in rows),
            "false_negatives": sum(
                r["expected_critical_failure"] and not r["actual_critical_failure"]
                for r in rows
            ),
            "false_positives": sum(
                not r["expected_critical_failure"] and r["actual_critical_failure"]
                for r in rows
            ),
        }

    report = {
        "evaluation_version": dataset["evaluation_version"],
        "description": dataset["description"],
        "labeling_note": dataset.get("labeling_note"),
        "note": (
            "Internally authored policy-derived challenge set, not externally "
            "adjudicated. Results are not proof of generalization."
        ),
        "case_count": len(results),
        "matched_cases": matched,
        "agreement_percentage": round(100 * matched / len(results), 2),
        "critical_failure_detection": {
            "true_positives": tp,
            "false_negatives": fn,
            "false_positives": fp,
            "true_negatives": tn,
            "expected_critical_cases": critical_total,
            "expected_safe_cases": safe_total,
            "sensitivity_percentage": sensitivity,
            "specificity_percentage": specificity,
        },
        "by_ticket": by_ticket,
        "mismatches": [r for r in results if not r["matched"]],
        "results": results,
    }

    output_dir = ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"response_quality_challenge_{stamp}.json"
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Challenge cases evaluated: {len(results)}")
    print(f"Case agreement: {matched}/{len(results)} ({report['agreement_percentage']}%)")
    print(f"TP: {tp}; FN: {fn}; FP: {fp}; TN: {tn}")
    print(f"Sensitivity: {sensitivity}%; specificity: {specificity}%")
    print("\nBy ticket:")
    for tid, metrics in by_ticket.items():
        print(
            f"  {tid}: {metrics['matched']}/{metrics['cases']} matched; "
            f"FN={metrics['false_negatives']}; FP={metrics['false_positives']}"
        )

    print("\nMismatches:")
    mismatches = report["mismatches"]
    if not mismatches:
        print("  None")
    for row in mismatches:
        print(
            f"  {row['case_id']}: expected={row['expected_critical_failure']}, "
            f"actual={row['actual_critical_failure']}; "
            f"failures={row['critical_failures']}"
        )

    print(f"\nReport saved to: {output_path.relative_to(ROOT)}")
    print("Offline evaluation only; no LLM/API calls were made.")


if __name__ == "__main__":
    main()
