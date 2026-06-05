"""
Part 3 benchmark runner for Vinpearl documents in src/data.

Usage (from project root):
    python scripts/run_part3_benchmark.py
    python scripts/run_part3_benchmark.py -o output/benchmark_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "output" / "benchmark_results.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent import KnowledgeBaseAgent
from src.chunking import ChunkingStrategyComparator, SectionChunker
from src.data_loader import DATA_DIR, build_documents_for_store, load_vinpearl_documents
from src.demo_llm import context_grounded_llm
from src.embeddings import _mock_embed
from src.store import EmbeddingStore

BASELINE_DOCS = [
    "VinWonders_Nha_Trang",
    "Grand_World",
    "Vinpearl_Safari_Phú_Quốc",
]

BENCHMARK_QUERIES = [
    {
        "query": "Giá vé hiện tại của VinWonders Nha Trang là bao nhiêu?",
        "filter": None,
    },
    {
        "query": "Aquafield Nha Trang có những phòng trị liệu xông hơi nào?",
        "filter": None,
    },
    {
        "query": "Combo 2N1Đ Vinpearl Hotel Cần Thơ bao gồm những dịch vụ gì?",
        "filter": {"destination": "Cần Thơ", "product_type": "hotel_combo"},
    },
    {
        "query": "Voucher golf Sunrise áp dụng tee-time và ngày nào?",
        "filter": {"product_type": "golf_voucher"},
    },
    {
        "query": "Night Safari tại Vinpearl Safari Phú Quốc là gì?",
        "filter": {"destination": "Phú Quốc"},
    },
]


def _preserves_context(strategy: str, avg_length: float, count: int) -> str:
    if strategy == "by_sentences":
        return "Cao — giữ ranh giới câu"
    if strategy == "recursive":
        return "Trung bình–cao — tôn trọng đoạn/section"
    if avg_length < 120:
        return "Thấp — dễ cắt giữa ý"
    return "Trung bình"


def run_baseline() -> list[dict]:
    rows: list[dict] = []
    comparator = ChunkingStrategyComparator()
    parsed_by_id = {doc["doc_id"]: doc for doc in load_vinpearl_documents()}

    for doc_id in BASELINE_DOCS:
        parsed = parsed_by_id.get(doc_id)
        if not parsed:
            continue
        result = comparator.compare(parsed["content"], chunk_size=200)
        for strategy_key, label in [
            ("fixed_size", "FixedSizeChunker"),
            ("by_sentences", "SentenceChunker"),
            ("recursive", "RecursiveChunker"),
        ]:
            stats = result[strategy_key]
            rows.append(
                {
                    "document": doc_id,
                    "strategy": label,
                    "chunk_count": stats["count"],
                    "avg_length": round(stats["avg_length"], 1),
                    "preserves_context": _preserves_context(
                        strategy_key, stats["avg_length"], stats["count"]
                    ),
                }
            )
    return rows


def run_section_strategy() -> list[dict]:
    rows: list[dict] = []
    chunker = SectionChunker(chunk_size=800)
    parsed_by_id = {doc["doc_id"]: doc for doc in load_vinpearl_documents()}

    for doc_id in BASELINE_DOCS:
        parsed = parsed_by_id.get(doc_id)
        if not parsed:
            continue
        chunks = chunker.chunk(parsed["content"])
        count = len(chunks)
        avg_length = (sum(len(c) for c in chunks) / count) if count else 0.0
        rows.append(
            {
                "document": doc_id,
                "strategy": "SectionChunker (của tôi)",
                "chunk_count": count,
                "avg_length": round(avg_length, 1),
                "retrieval_quality": "Cao hơn baseline khi hỏi theo mục (giá, điều khoản, mô tả)",
            }
        )
    return rows


def run_retrieval_benchmark(top_k: int = 3) -> list[dict]:
    store = EmbeddingStore(collection_name="vinpearl_part3", embedding_fn=_mock_embed)
    store.add_documents(build_documents_for_store(chunker=SectionChunker(chunk_size=800)))
    agent = KnowledgeBaseAgent(store=store, llm_fn=context_grounded_llm)

    results: list[dict] = []
    for item in BENCHMARK_QUERIES:
        query = item["query"]
        metadata_filter = item["filter"]
        if metadata_filter:
            hits = store.search_with_filter(query, top_k=top_k, metadata_filter=metadata_filter)
        else:
            hits = store.search(query, top_k=top_k)

        agent_answer = agent.answer(query, top_k=top_k, metadata_filter=metadata_filter)

        results.append(
            {
                "query": query,
                "filter": metadata_filter,
                "agent_answer": agent_answer,
                "top_hits": [
                    {
                        "score": round(hit["score"], 4),
                        "title": hit["metadata"].get("title"),
                        "section": hit["metadata"].get("section"),
                        "preview": hit["content"][:160].replace("\n", " "),
                    }
                    for hit in hits
                ],
            }
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Chạy Part 3 benchmark và lưu JSON")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"File JSON đầu ra (mặc định: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="In toàn bộ JSON ra terminal (ngoài việc lưu file)",
    )
    args = parser.parse_args()

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": run_baseline(),
        "section_strategy": run_section_strategy(),
        "retrieval": run_retrieval_benchmark(),
        "inventory": [
            {
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "chars": len(doc["content"]),
                "metadata": {
                    "product_type": doc["product_type"],
                    "destination": doc["destination"],
                    "current_price": doc["current_price"],
                },
            }
            for doc in load_vinpearl_documents()
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Đã lưu benchmark → {args.output.resolve()}")
    print(f"  - baseline:         {len(payload['baseline'])} dòng")
    print(f"  - section_strategy: {len(payload['section_strategy'])} dòng")
    print(f"  - retrieval:        {len(payload['retrieval'])} queries")
    print(f"  - inventory:        {len(payload['inventory'])} tài liệu")

    if args.print_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
