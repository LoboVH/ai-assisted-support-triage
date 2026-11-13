
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABELS_FILE = ROOT / "output" / "human_relevance_labels.json"

THRESHOLD = 0.35

with LABELS_FILE.open(encoding="utf-8") as file:
    rows = json.load(file)

rows = [
    row for row in rows
    if not row["claim"].startswith("Thank you for contacting support")
]

errors = []

for row in rows:
    actual = row["expected_relevance"] == "relevant"
    predicted = row["similarity"] >= THRESHOLD

    if actual != predicted:
        errors.append((row, actual, predicted))

print(f"Threshold: {THRESHOLD:.2f}")
print(f"Misclassified pairs: {len(errors)}")
print()

for row, actual, predicted in errors:
    if predicted and not actual:
        error_type = "FALSE POSITIVE"
    else:
        error_type = "FALSE NEGATIVE"

    print(f"[{error_type}]")
    print(f"Ticket: {row['ticket_id']}")
    print(f"Evidence: {row['evidence_source']}")
    print(f"Similarity: {row['similarity']:.3f}")
    print(f"Human label: {row['expected_relevance']}")
    print(f"Claim: {row['claim']}")
    print()
