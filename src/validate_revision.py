
import json
from pathlib import Path
from runtime_paths import project_root

from chunking import split_into_chunks
from embeddings import EmbeddingModel
from grounding import (
    is_conversational_language,
    split_claims,
    EmbeddingRelevanceAssessor,
    validate_grounding,
)
from nli import NLIModel
from retrieval import semantic_retrieve, select_chunks_with_policy

ROOT = project_root()
TICKETS_PATH = ROOT / "tickets.json"
KNOWLEDGE_BASE_DIR = ROOT / "knowledge_base"
REVISIONS_PATH = ROOT / "output" / "revised_response_drafts.json"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, data) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    tickets_data = load_json(TICKETS_PATH)
    revisions = load_json(REVISIONS_PATH)

    if not isinstance(revisions, list) or not revisions:
        raise ValueError("No revision records found.")

    tickets_by_id = {
        ticket["ticket_id"]: ticket
        for ticket in tickets_data["tickets"]
    }

    # Build chunks from the existing knowledge base without modifying it.
    all_chunks = []

    for path in sorted(KNOWLEDGE_BASE_DIR.rglob("*")):
        if not path.is_file():
            continue

        document_text = path.read_text(encoding="utf-8")
        all_chunks.extend(
            split_into_chunks(
                document_text,
                path.stem,
                str(path),
            )
        )

    if not all_chunks:
        raise ValueError("No knowledge chunks were created.")

    print("Loading embedding and NLI models...")
    embedding_model = EmbeddingModel()
    relevance_assessor = EmbeddingRelevanceAssessor(
        embedding_model,
        threshold=0.40,
    )
    nli_model = NLIModel()

    chunk_texts = [chunk.text for chunk in all_chunks]
    chunk_embeddings = embedding_model.encode(chunk_texts)

    for revision in revisions:
        ticket_id = revision["ticket_id"]

        if ticket_id not in tickets_by_id:
            raise ValueError(f"No ticket found for {ticket_id}.")

        ticket = tickets_by_id[ticket_id]
        ticket_text = (
            ticket.get("subject", "") + " " + ticket.get("message", "")
        ).strip()

        if not revision.get("draft", "").strip():
            revision["status"] = "validation_failed"
            revision["grounding_results"] = []
            revision["grounding_passed"] = False
            revision["validation_error"] = "Draft text is empty."
            continue

        print(f"\nValidating revision for {ticket_id}...")

        # Retrieve evidence using both the original ticket and each
        # substantive claim in the revised response.
        retrieval_queries = [ticket_text]
        retrieval_queries.extend(
            claim
            for claim in split_claims(revision["draft"])
            if not is_conversational_language(claim)
        )

        retrieved_by_id = {}

        for query in retrieval_queries:
            scored_chunks = semantic_retrieve(
                query,
                all_chunks,
                chunk_embeddings,
                embedding_model,
            )

            selected_for_query = select_chunks_with_policy(
                query,
                scored_chunks,
                top_k=3,
                min_score=0.30,
            )

            for score, chunk in selected_for_query:
                existing = retrieved_by_id.get(chunk.chunk_id)

                if existing is None or score > existing[0]:
                    retrieved_by_id[chunk.chunk_id] = (score, chunk)

        selected_chunks = sorted(
            retrieved_by_id.values(),
            key=lambda item: item[0],
            reverse=True,
        )

        grounding_results = validate_grounding(
            revision["draft"],
            ticket_text,
            selected_chunks,
            nli_model,
            relevance_assessor=relevance_assessor,
            embedding_model=embedding_model,
        )

        grounding_passed = (
            bool(grounding_results)
            and all(
                result.get("grounded") is True
                for result in grounding_results
            )
        )

        revision["evidence_chunk_ids"] = [
            chunk.chunk_id for _, chunk in selected_chunks
        ]
        revision["grounding_results"] = grounding_results
        revision["grounding_passed"] = grounding_passed
        revision["status"] = (
            "grounded_pending_human_review"
            if grounding_passed
            else "grounding_failed_needs_review"
        )
        revision.pop("validation_error", None)

        print(f"Grounding passed: {grounding_passed}")
        print(f"Status: {revision['status']}")
        print(f"Claims checked: {len(grounding_results)}")
        print(f"Evidence chunks selected: {len(selected_chunks)}")

        for result in grounding_results:
            print(
                f"  [{'PASS' if result.get('grounded') else 'FAIL'}] "
                f"{result['claim']}"
            )

    save_json(REVISIONS_PATH, revisions)
    print(f"\nUpdated revision file: {REVISIONS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nValidation cancelled.")
