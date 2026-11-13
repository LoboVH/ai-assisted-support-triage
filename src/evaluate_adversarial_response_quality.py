"""Run the deterministic adversarial response-quality benchmark.

This evaluates the existing heuristic evaluator against explicitly labelled
cases. It does not call an LLM and does not claim to establish real-world
accuracy; it measures agreement with the expected labels in the benchmark.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Support both `python src/evaluate_adversarial_response_quality.py` and
# imports from the project root.
HERE = Path(__file__).resolve()
ROOT = HERE.parent.parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from evaluate_response_quality import evaluate_ticket  # noqa: E402

TICKETS_PATH = ROOT / "tickets.json"
LABELS_PATH = ROOT / "evaluation" / "response_quality_labels.json"
CASES_PATH = ROOT / "evaluation" / "response_quality_cases.json"
OUTPUT_PATH = ROOT / "output" / "response_quality_adversarial_evaluation.json"

SCORE_KEYS = ("relevance", "factual_accuracy", "actionability", "policy_compliance")


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path.relative_to(ROOT)}")
    return json.loads(path.read_text(encoding="utf-8"))


def benchmark_group(case_id: str) -> str:
    """Assign each curated case to the principal behaviour being tested."""
    explicit = {
        "withdrawal_supported_paraphrase": "safe_paraphrases",
        "locked_account_safe_paraphrase_keeps_review": "safe_paraphrases",
        "trade_policy_supported_paraphrase": "safe_paraphrases",
        "security_supported_paraphrase": "safe_paraphrases",
        "withdrawal_missing_payout_caveat": "missing_required_caveat",
        "withdrawal_guaranteed_tomorrow": "unsupported_promises",
        "locked_account_unlock_promise": "unsupported_promises",
        "trade_reversal_promise": "unsupported_promises",
        "locked_account_invented_reason": "invented_explanations_or_assurances",
        "security_unsupported_assurance": "invented_explanations_or_assurances",
        "trade_policy_omits_exceptions": "missing_policy_exceptions",
        "security_contact_instruction_weakened": "weakened_security_instructions",
    }
    return explicit.get(case_id, "uncategorised")


def evaluate_case(case: dict[str, Any], tickets_by_id: dict, labels_by_id: dict) -> dict[str, Any]:
    case_id = case.get("case_id")
    ticket_id = case.get("ticket_id")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError(f"Case has missing/invalid case_id: {case!r}")
    if ticket_id not in tickets_by_id:
        raise ValueError(f"Case {case_id} references unknown ticket: {ticket_id}")
    if ticket_id not in labels_by_id:
        raise ValueError(f"No base labels exist for ticket {ticket_id}")
    if not isinstance(case.get("draft"), str) or not case["draft"].strip():
        raise ValueError(f"Case {case_id} has an empty or invalid draft")

    result = evaluate_ticket(
        tickets_by_id[ticket_id],
        {
            "ticket_id": ticket_id,
            "draft": case["draft"],
            "status": "adversarial_evaluation_case",
        },
        labels_by_id[ticket_id],
    )
    expected = case.get("expected")
    if not isinstance(expected, dict):
        raise ValueError(f"Case {case_id} is missing its expected object")

    mismatches = []
    checked_fields = []
    expected_scores = expected.get("scores", expected)
    for key in SCORE_KEYS:
        if key in expected_scores:
            checked_fields.append(f"scores.{key}")
            actual_value = result.get("scores", {}).get(key)
            if actual_value != expected_scores[key]:
                mismatches.append({
                    "field": f"scores.{key}",
                    "expected": expected_scores[key],
                    "actual": actual_value,
                })

    if "critical_failure" in expected:
        checked_fields.append("critical_failure")
        actual_critical_failure = bool(result.get("critical_failures"))
        if actual_critical_failure != expected["critical_failure"]:
            mismatches.append({
                "field": "critical_failure",
                "expected": expected["critical_failure"],
                "actual": actual_critical_failure,
            })

    if "review_status" in expected:
        checked_fields.append("review_status")
        if result.get("review_status") != expected["review_status"]:
            mismatches.append({
                "field": "review_status",
                "expected": expected["review_status"],
                "actual": result.get("review_status"),
            })

    return {
        "case_id": case_id,
        "benchmark_group": benchmark_group(case_id),
        "ticket_id": ticket_id,
        "expected_match": not mismatches,
        "checked_fields": checked_fields,
        "mismatches": mismatches,
        "expected": expected,
        "actual": {
            "scores": result.get("scores", {}),
            "critical_failure": bool(result.get("critical_failures")),
            "critical_failures": result.get("critical_failures", []),
            "review_status": result.get("review_status"),
            "review_reasons": result.get("review_reasons", []),
            "total_score": result.get("total_score"),
            "maximum_score": result.get("maximum_score"),
        },
    }


def main() -> int:
    tickets_data = load_json(TICKETS_PATH)
    labels_data = load_json(LABELS_PATH)
    cases_data = load_json(CASES_PATH)

    tickets_by_id = {item["ticket_id"]: item for item in tickets_data["tickets"]}
    labels_by_id = {item["ticket_id"]: item for item in labels_data["tickets"]}
    cases = cases_data.get("cases", [])
    if not cases:
        raise ValueError("No adversarial evaluation cases found")

    case_ids = [case.get("case_id") for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Adversarial case IDs must be unique")

    results = [evaluate_case(case, tickets_by_id, labels_by_id) for case in cases]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        groups[result["benchmark_group"]].append(result)

    group_summaries = {}
    for group_name, group_results in sorted(groups.items()):
        matched = sum(item["expected_match"] for item in group_results)
        group_summaries[group_name] = {
            "case_count": len(group_results),
            "expected_matches": matched,
            "mismatches": len(group_results) - matched,
            "agreement_percentage": round(100 * matched / len(group_results), 2),
        }

    expected_matches = sum(item["expected_match"] for item in results)
    critical_expected = [
        item for item in results if item.get("expected", {}).get("critical_failure") is True
    ]
    critical_detected = sum(item["actual"]["critical_failure"] for item in critical_expected)
    critical_missed = [
        item["case_id"] for item in critical_expected
        if not item["actual"]["critical_failure"]
    ]
    critical_false_positives = [
        item["case_id"] for item in results
        if item.get("expected", {}).get("critical_failure") is False
        and item["actual"]["critical_failure"]
    ]
    mismatches = [item for item in results if not item["expected_match"]]

    report = {
        "evaluation_version": cases_data.get("evaluation_version", "unknown"),
        "report_type": "adversarial_response_quality",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_source": "evaluation/response_quality_cases.json",
        "scoring_note": (
            "This report measures agreement with curated expected labels. "
            "It is not a measure of real-world accuracy and does not replace "
            "claim-level grounding checks or human review."
        ),
        "case_count": len(results),
        "expected_matches": expected_matches,
        "mismatch_count": len(mismatches),
        "agreement_percentage": round(100 * expected_matches / len(results), 2),
        "critical_failure_cases": {
            "expected_count": len(critical_expected),
            "detected_count": critical_detected,
            "missed_case_ids": critical_missed,
            "false_positive_case_ids": critical_false_positives,
        },
        "group_summaries": group_summaries,
        "results": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Adversarial cases evaluated: {report['case_count']}")
    print(
        f"Expected-label agreement: {expected_matches}/{len(results)} "
        f"({report['agreement_percentage']}%)"
    )
    print(f"Case mismatches: {len(mismatches)}")
    print(
        "Critical failures detected: "
        f"{critical_detected}/{len(critical_expected)} expected critical cases"
    )
    if critical_missed:
        print("MISSED CRITICAL CASES:", ", ".join(critical_missed))
    if critical_false_positives:
        print("CRITICAL FALSE POSITIVES:", ", ".join(critical_false_positives))

    print("\nAgreement by benchmark group:")
    for group_name, summary in group_summaries.items():
        print(
            f"  {group_name}: {summary['expected_matches']}/{summary['case_count']} "
            f"({summary['agreement_percentage']}%)"
        )

    if mismatches:
        print("\nMismatches:")
        for item in mismatches:
            print(f"  {item['case_id']}: {json.dumps(item['mismatches'], ensure_ascii=False)}")

    print(f"\nReport saved to: {OUTPUT_PATH.relative_to(ROOT)}")
    print("No LLM/API calls were made.")
    return 0 if not mismatches else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
