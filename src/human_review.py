
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
REVIEWS_PATH = ROOT / "evaluation" / "human_reviews.json"

VALID_DECISIONS = {
    "approved",
    "changes_requested",
    "rejected",
}


def draft_hash(draft: str) -> str:
    """Return a stable hash identifying the exact draft text."""
    return hashlib.sha256(draft.encode("utf-8")).hexdigest()


def load_reviews(path: Path = REVIEWS_PATH) -> dict:
    """Load the persistent review ledger."""
    if not path.exists():
        return {
            "review_schema_version": "1.0",
            "reviews": [],
        }

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("Review ledger must be a JSON object.")

    if not isinstance(data.get("reviews"), list):
        raise ValueError("Review ledger must contain a reviews list.")

    if not all(isinstance(review, dict) for review in data["reviews"]):
        raise ValueError("Every review ledger entry must be an object.")

    return data


def record_review(
    ticket_id: str,
    draft: str,
    decision: str,
    reviewer_score: int | None = None,
    notes: str | None = None,
    reviewer: str | None = None,
    path: Path = REVIEWS_PATH,
) -> dict:
    """Record a human decision for the exact draft that was reviewed."""
    if not ticket_id.strip():
        raise ValueError("ticket_id must not be empty.")

    if decision not in VALID_DECISIONS:
        raise ValueError(
            f"Invalid decision {decision!r}. "
            f"Choose one of: {', '.join(sorted(VALID_DECISIONS))}"
        )

    if reviewer_score is not None and (
        isinstance(reviewer_score, bool)
        or not isinstance(reviewer_score, int)
        or not 0 <= reviewer_score <= 8
    ):
        raise ValueError("reviewer_score must be an integer from 0 to 8.")

    if decision == "approved" and reviewer_score is not None:
        if reviewer_score < 6:
            raise ValueError(
                "An approved draft must have a reviewer score of at least 6."
            )

    if not draft.strip():
        raise ValueError("draft must not be empty.")

    ledger = load_reviews(path)

    review = {
        "ticket_id": ticket_id,
        "draft_hash": draft_hash(draft),
        "decision": decision,
        "reviewer_score": reviewer_score,
        "notes": notes,
        "reviewer": reviewer,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
    }

    ledger["reviews"].append(review)


    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(ledger, file, indent=2, ensure_ascii=False)
        file.write("\n")

    return review


def get_current_review(ticket_id: str, draft: str, path: Path = REVIEWS_PATH) -> dict | None:
    """Return the latest review matching this exact draft, if present."""
    current_hash = draft_hash(draft)
    ledger = load_reviews(path)

    matching_reviews = [
        review
        for review in ledger["reviews"]
        if review.get("ticket_id") == ticket_id
        and review.get("draft_hash") == current_hash
    ]

    if not matching_reviews:
        return None

    return matching_reviews[-1]
