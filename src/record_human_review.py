
import json
import argparse
from pathlib import Path

from src.human_review import record_review

ROOT = Path(__file__).resolve().parent.parent
DRAFTS_PATH = ROOT / "output" / "response_drafts.json"


def load_drafts(path: Path = DRAFTS_PATH) -> list[dict]:
    """Load generated response drafts."""
    with path.open("r", encoding="utf-8") as file:
        drafts = json.load(file)

    if not isinstance(drafts, list):
        raise ValueError("Response drafts must be a JSON list.")

    return drafts


def main(drafts_path: Path = DRAFTS_PATH) -> None:
    drafts = load_drafts(drafts_path)

    if not drafts:
        print("No response drafts found.")
        return

    print("\nAvailable tickets:")
    for draft in drafts:
        print(f"  {draft['ticket_id']} - {draft.get('status', 'unknown')}")

    ticket_id = input("\nTicket ID to review: ").strip().upper()

    matches = [
        draft for draft in drafts
        if draft.get("ticket_id") == ticket_id
    ]

    if not matches:
        print(f"No draft found for ticket {ticket_id!r}.")
        return

    draft_record = matches[0]
    draft_text = draft_record.get("draft", "")

    if not isinstance(draft_text, str) or not draft_text.strip():
        print("The selected ticket has no valid draft text.")
        return

    print("\n--- Draft for review ---")
    print(draft_text)
    print("------------------------")
    print(f"Grounding status: {draft_record.get('status', 'unknown')}")

    print("\nDecisions:")
    print("  1. approved")
    print("  2. changes_requested")
    print("  3. rejected")

    decision_options = {
        "1": "approved",
        "2": "changes_requested",
        "3": "rejected",
    }

    choice = input("Your decision (1/2/3): ").strip()

    if choice not in decision_options:
        print("Invalid choice. No review was recorded.")
        return

    decision = decision_options[choice]

    score_text = input(
        "Reviewer score (0-8, or Enter to skip): "
    ).strip()

    reviewer_score = None

    if score_text:
        try:
            reviewer_score = int(score_text)
        except ValueError:
            print("Score must be an integer. No review was recorded.")
            return

    notes = input("Review notes (or Enter to skip): ").strip() or None
    reviewer = input("Reviewer ID (or Enter to skip): ").strip() or None

    try:
        review = record_review(
            ticket_id=ticket_id,
            draft=draft_text,
            decision=decision,
            reviewer_score=reviewer_score,
            notes=notes,
            reviewer=reviewer,
        )
    except ValueError as error:
        print(f"Review not recorded: {error}")
        return

    print("\nReview recorded successfully.")
    print(f"Ticket: {review['ticket_id']}")
    print(f"Decision: {review['decision']}")
    print(f"Draft hash: {review['draft_hash']}")
    print(f"Reviewed at: {review['reviewed_at']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Record a human review of response drafts."
    )
    parser.add_argument(
        "--drafts",
        type=Path,
        default=DRAFTS_PATH,
        help="Path to the response drafts JSON file.",
    )
    args = parser.parse_args()

    try:
        main(args.drafts)
    except KeyboardInterrupt:
        print("\nReview cancelled. No review was recorded.")
