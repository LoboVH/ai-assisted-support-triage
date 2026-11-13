import re
import hashlib

from models import KnowledgeChunk

def calculate_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def split_into_chunks(
    text: str,
    document_id: str,
    source_file: str,
    ):
    sections = re.split(r"(?m)^#{1,6}\s+", text)

    chunks = []
    chunk_number = 0

    for section in sections:
        section = section.strip()

        if not section:
            continue

        lines = section.splitlines()

        title = lines[0].strip()
        content = "\n".join(lines[1:]).strip()

        if not content:
            continue

        chunk_number += 1

        chunk_id = f"{document_id}-{chunk_number:03d}"

        chunks.append(
            KnowledgeChunk(
                document_id=document_id,
                source_file=source_file,
                chunk_id=chunk_id,
                section_title=title,
                text=content,
                character_count=len(content),
                content_hash=calculate_hash(content),
            )
        )

    return chunks