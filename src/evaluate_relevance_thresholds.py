
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABELS_FILE = PROJECT_ROOT / "output" / "human_relevance_labels.json"

THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


def main():
    with LABELS_FILE.open(encoding="utf-8") as file:
        rows = json.load(file)

    # Only evaluate substantive policy/advice claims for now.
    # Acknowledgments are conversational and should not determine
    # the relevance threshold for factual grounding.
    substantive_rows = [
        row for row in rows
        if not row["claim"].startswith("Thank you for contacting support")
    ]

    if not substantive_rows:
        raise ValueError(
            "No substantive claim pairs available for relevance evaluation."
        )

    print(f"All reviewed pairs: {len(rows)}")
    print(f"Substantive claim pairs: {len(substantive_rows)}")
    print()

    for threshold in THRESHOLDS:
        tp = fp = tn = fn = 0

        for row in substantive_rows:
            actual = row["expected_relevance"] == "relevant"
            predicted = row["similarity"] >= threshold

            if predicted and actual:
                tp += 1
            elif predicted and not actual:
                fp += 1
            elif not predicted and not actual:
                tn += 1
            else:
                fn += 1

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        accuracy = (tp + tn) / len(substantive_rows)

        print(
            f"threshold={threshold:.2f} | "
            f"TP={tp:2} FP={fp:2} TN={tn:2} FN={fn:2} | "
            f"precision={precision:.3f} "
            f"recall={recall:.3f} "
            f"accuracy={accuracy:.3f}"
        )


if __name__ == "__main__":
    main()
