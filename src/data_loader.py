from __future__ import annotations

import re
from pathlib import Path

from .chunking import SectionChunker
from .models import Document

DATA_DIR = Path(__file__).resolve().parent / "data"

_SECTION_HEADER = re.compile(r"^##\s+(.+)$", re.MULTILINE)
_URL_RE = re.compile(r"^- URL:\s*(.+)$", re.MULTILINE)
_ORIGINAL_PRICE_RE = re.compile(r"^- Giá gốc:\s*(.*)$", re.MULTILINE)
_CURRENT_PRICE_RE = re.compile(r"^- Giá hiện tại:\s*(.*)$", re.MULTILINE)


def _infer_product_type(title: str, content: str) -> str:
    text = f"{title} {content[:500]}".lower()
    if "golf" in text or "tee time" in text or "tee-time" in text:
        return "golf_voucher"
    if any(k in text for k in ("combo", "2n1đ", "2n2đ", "3n2đ", "phòng", "khách sạn", "hotel")):
        return "hotel_combo"
    if any(k in text for k in ("aquafield", "spa", "jjimjilbang", "xông hơi")):
        return "spa"
    if any(k in text for k in ("safari", "vinwonders", "công viên", "vé vào cửa", "bảo tàng")):
        return "attraction"
    if "grand world" in text:
        return "entertainment"
    return "tour"


def _infer_destination(title: str, content: str) -> str:
    text = f"{title} {content[:800]}"
    destinations = [
        "Phú Quốc",
        "Nha Trang",
        "Cần Thơ",
        "Hà Nội",
        "Hội An",
        "Đà Nẵng",
        "Hạ Long",
        "Hải Phòng",
    ]
    for name in destinations:
        if name.lower() in text.lower():
            return name
    return "Việt Nam"


def parse_markdown_file(path: Path) -> dict:
    """Parse a Vinpearl booking markdown file into doc-level fields."""
    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()
    title = lines[0].lstrip("# ").strip() if lines and lines[0].startswith("#") else path.stem

    url_match = _URL_RE.search(content)
    original_match = _ORIGINAL_PRICE_RE.search(content)
    current_match = _CURRENT_PRICE_RE.search(content)

    return {
        "doc_id": path.stem,
        "title": title,
        "source_url": url_match.group(1).strip() if url_match else "",
        "original_price": original_match.group(1).strip() if original_match else "",
        "current_price": current_match.group(1).strip() if current_match else "",
        "product_type": _infer_product_type(title, content),
        "destination": _infer_destination(title, content),
        "content": content,
        "source_file": str(path),
    }


def _section_name(chunk_text: str, fallback: str = "Tổng quan") -> str:
    match = _SECTION_HEADER.search(chunk_text)
    if match:
        return match.group(1).strip()
    return fallback


def chunk_to_schema_records(parsed: dict, chunker: SectionChunker | None = None) -> list[dict]:
    """
    Chunk a parsed document into records matching src/data/schema.js:

    { "text": "...", "metadata": { title, url, original_price, current_price, section, chunk_index } }
    """
    chunker = chunker or SectionChunker()
    chunks = chunker.chunk(parsed["content"])
    records: list[dict] = []

    for index, chunk_text in enumerate(chunks):
        records.append(
            {
                "text": chunk_text,
                "metadata": {
                    "title": parsed["title"],
                    "url": parsed["source_url"],
                    "original_price": parsed["original_price"],
                    "current_price": parsed["current_price"],
                    "section": _section_name(chunk_text),
                    "chunk_index": index,
                    "doc_id": parsed["doc_id"],
                    "product_type": parsed["product_type"],
                    "destination": parsed["destination"],
                },
            }
        )
    return records


def records_to_documents(records: list[dict]) -> list[Document]:
    documents: list[Document] = []
    for record in records:
        metadata = dict(record["metadata"])
        doc_id = metadata.get("doc_id") or metadata.get("title", "doc")
        chunk_index = metadata.get("chunk_index", 0)
        documents.append(
            Document(
                id=f"{doc_id}::{chunk_index}",
                content=record["text"],
                metadata=metadata,
            )
        )
    return documents


def load_vinpearl_documents(data_dir: Path | None = None) -> list[dict]:
    """Load and parse all markdown files under src/data (excluding schema files)."""
    root = data_dir or DATA_DIR
    parsed_docs: list[dict] = []
    for path in sorted(root.glob("*.md")):
        parsed_docs.append(parse_markdown_file(path))
    return parsed_docs


def build_documents_for_store(
    data_dir: Path | None = None,
    chunker: SectionChunker | None = None,
) -> list[Document]:
    """Parse markdown files, chunk by section, return Document objects for EmbeddingStore."""
    all_records: list[dict] = []
    for parsed in load_vinpearl_documents(data_dir):
        all_records.extend(chunk_to_schema_records(parsed, chunker=chunker))
    return records_to_documents(all_records)
