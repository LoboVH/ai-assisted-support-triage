
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EVALUATION_FILE = PROJECT_ROOT / "output" / "relevance_evaluation.json"
OUTPUT_FILE = PROJECT_ROOT / "output" / "human_relevance_labels.json"

# Human judgments from our review.
# Key: (ticket_id, claim, evidence_source)
LABELS = {
    # T-1001: acknowledgment
    ("T-1001", "Thank you for contacting support about your pending withdrawal", "ticket"): ("relevant", "neutral"),
    ("T-1001", "Thank you for contacting support about your pending withdrawal", "payments-001"): ("relevant", "neutral"),
    ("T-1001", "Thank you for contacting support about your pending withdrawal", "security-001"): ("irrelevant", "neutral"),
    ("T-1001", "Thank you for contacting support about your pending withdrawal", "payments-002"): ("irrelevant", "neutral"),

    # T-1001: withdrawal review policy
    ("T-1001", "Verified customers usually have withdrawals reviewed within 24 hours, although delays may occur when additional checks are required or account activity is under review", "ticket"): ("relevant", "neutral"),
    ("T-1001", "Verified customers usually have withdrawals reviewed within 24 hours, although delays may occur when additional checks are required or account activity is under review", "payments-001"): ("relevant", "entailment"),
    ("T-1001", "Verified customers usually have withdrawals reviewed within 24 hours, although delays may occur when additional checks are required or account activity is under review", "security-001"): ("relevant", "neutral"),
    ("T-1001", "Verified customers usually have withdrawals reviewed within 24 hours, although delays may occur when additional checks are required or account activity is under review", "payments-002"): ("relevant", "neutral"),

    # T-1001: payout timing rule
    ("T-1001", "We cannot guarantee payout timing unless the payment provider has confirmed processing", "ticket"): ("irrelevant", "neutral"),
    ("T-1001", "We cannot guarantee payout timing unless the payment provider has confirmed processing", "payments-001"): ("relevant", "neutral"),
    ("T-1001", "We cannot guarantee payout timing unless the payment provider has confirmed processing", "security-001"): ("irrelevant", "neutral"),
    ("T-1001", "We cannot guarantee payout timing unless the payment provider has confirmed processing", "payments-002"): ("relevant", "entailment"),

    # T-1002: acknowledgment
    ("T-1002", "Thank you for contacting support about your locked account", "ticket"): ("relevant", "neutral"),
    ("T-1002", "Thank you for contacting support about your locked account", "security-001"): ("relevant", "neutral"),
    ("T-1002", "Thank you for contacting support about your locked account", "security-002"): ("relevant", "neutral"),
    ("T-1002", "Thank you for contacting support about your locked account", "payments-001"): ("irrelevant", "neutral"),

    # T-1002: restriction policy
    ("T-1002", "Accounts may be temporarily restricted when suspicious login, device, password, or payment activity is detected", "ticket"): ("relevant", "neutral"),
    ("T-1002", "Accounts may be temporarily restricted when suspicious login, device, password, or payment activity is detected", "security-001"): ("relevant", "entailment"),
    ("T-1002", "Accounts may be temporarily restricted when suspicious login, device, password, or payment activity is detected", "security-002"): ("relevant", "neutral"),
    ("T-1002", "Accounts may be temporarily restricted when suspicious login, device, password, or payment activity is detected", "payments-001"): ("irrelevant", "neutral"),

    # T-1002: specific reason unknown
    ("T-1002", "The available information does not establish the specific reason for your account restriction", "ticket"): ("relevant", "neutral"),
    ("T-1002", "The available information does not establish the specific reason for your account restriction", "security-001"): ("relevant", "neutral"),
    ("T-1002", "The available information does not establish the specific reason for your account restriction", "security-002"): ("irrelevant", "neutral"),
    ("T-1002", "The available information does not establish the specific reason for your account restriction", "payments-001"): ("irrelevant", "neutral"),

    # T-1003: acknowledgment
    ("T-1003", "Thank you for contacting support about your trade", "ticket"): ("relevant", "neutral"),
    ("T-1003", "Thank you for contacting support about your trade", "trading_policies-001"): ("irrelevant", "neutral"),

    # T-1003: trade finality
    ("T-1003", "Executed trades are final unless there is a confirmed platform error or an approved operational adjustment", "ticket"): ("irrelevant", "neutral"),
    ("T-1003", "Executed trades are final unless there is a confirmed platform error or an approved operational adjustment", "trading_policies-001"): ("relevant", "entailment"),

    # T-1004: acknowledgment
    ("T-1004", "Thank you for contacting support about possible unauthorised account access", "ticket"): ("relevant", "neutral"),
    ("T-1004", "Thank you for contacting support about possible unauthorised account access", "security-002"): ("relevant", "neutral"),
    ("T-1004", "Thank you for contacting support about possible unauthorised account access", "security-001"): ("relevant", "neutral"),

    # T-1004: security advice
    ("T-1004", "Please reset your password, enable two-factor authentication if available, and contact support immediately", "ticket"): ("relevant", "neutral"),
    ("T-1004", "Please reset your password, enable two-factor authentication if available, and contact support immediately", "security-002"): ("relevant", "entailment"),
    ("T-1004", "Please reset your password, enable two-factor authentication if available, and contact support immediately", "security-001"): ("irrelevant", "neutral"),
}


def main():
    with EVALUATION_FILE.open(encoding="utf-8") as file:
        evaluation = json.load(file)

    output = []
    missing = []

    for item in evaluation:
        key = (
            item["ticket_id"],
            item["claim"],
            item["evidence_source"],
        )

        if key not in LABELS:
            missing.append(key)
            continue

        relevance, relationship = LABELS[key]

        output.append({
            **item,
            "expected_relevance": relevance,
            "expected_relation": relationship,
        })

    if missing:
        print(f"Missing labels for {len(missing)} pairs:")
        for key in missing:
            print(key)
        raise SystemExit("No output written. Fix missing labels first.")

    if len(output) != len(evaluation):
        raise SystemExit(
            f"Expected {len(evaluation)} labels, produced {len(output)}."
        )

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print(f"Saved {len(output)} reviewed pairs to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
