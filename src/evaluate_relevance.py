
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from embeddings import EmbeddingModel
from grounding import inspect_claim_relevance
from models import KnowledgeChunk


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"


def load_json(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def main():
    drafts = load_json(OUTPUT_DIR / "response_drafts.json")
    corpus = load_json(OUTPUT_DIR / "knowledge_corpus.json")
    tickets = load_json(ROOT / "tickets.json")["tickets"]

    chunks_by_id = {
        item["chunk_id"]: KnowledgeChunk(**item)
        for item in corpus
    }

    tickets_by_id = {
        item["ticket_id"]: item
        for item in tickets
    }

    embedding_model = EmbeddingModel()
    report = []

    for draft_item in drafts:
        ticket_id = draft_item["ticket_id"]
        ticket = tickets_by_id[ticket_id]
        query = f'{ticket["subject"]} {ticket["message"]}'

        selected_chunks = [
            (0.0, chunks_by_id[chunk_id])
            for chunk_id in draft_item["evidence_chunk_ids"]
        ]

        scores = inspect_claim_relevance(
            draft_item["draft"],
            query,
            selected_chunks,
            embedding_model,
        )

        for result in scores:
            report.append({
                "ticket_id": ticket_id,
                "claim": result["claim"],
                "evidence_source": result["evidence_source"],
                "source_type": result["source_type"],
                "similarity": round(
                    result["claim_similarity"], 4
                ),
                # To be labelled manually during evaluation.
                "expected_relevance": None,
                "expected_relation": None,
            })

    report_path = OUTPUT_DIR / "relevance_evaluation.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print(f"Saved {len(report)} claim-evidence pairs to {report_path}")


if __name__ == "__main__":
    main()
