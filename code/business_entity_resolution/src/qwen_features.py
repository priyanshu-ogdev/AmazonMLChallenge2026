"""
Stage 2a-ii: Qwen3-Embedding-0.6B auxiliary dense pair features.

Qwen is deliberately inference-only in v1 (subordinate to Stage 2a-i BGE-M3).
This module encodes each entity once, applies the same task instruction to S1
and S2/S3 text, and computes cosine similarity for the exact candidate set that
will be scored downstream. No candidate is created or removed here.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.normalize import normalize_entity

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_INSTRUCTION = (
    "Represent a business entity for matching records that refer to the same "
    "real-world business."
)


def format_entity_text(text: str, instruction: str = DEFAULT_INSTRUCTION) -> str:
    """Apply identical instruction treatment to every entity representation."""
    return f"Instruct: {instruction}\nText: {text}"


class QwenEntityEncoder:
    """Batched Qwen encoder with deterministic, L2-normalized outputs."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        instruction: str = DEFAULT_INSTRUCTION,
        max_seq_length: int = 256,
        batch_size: int = 32,
        device: Optional[str] = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.instruction = instruction
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = max_seq_length
        self.model_name = model_name

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode texts once and return float32 unit vectors."""
        prompted = [format_entity_text(str(text), self.instruction) for text in texts]
        if not prompted:
            dimension = self.model.get_sentence_embedding_dimension()
            return np.empty((0, dimension), dtype=np.float32)
        embeddings = self.model.encode(
            prompted,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return np.asarray(embeddings, dtype=np.float32)


def load_records(paths: Iterable[Path]) -> Dict[str, str]:
    """Load and normalize entity text from raw or Layer 0 normalized challenge TSV files."""
    records: Dict[str, str] = {}
    for path in paths:
        path = Path(path)
        frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        if "entity_id" not in frame.columns:
            raise ValueError(f"{path} is missing required column: entity_id")

        if "encoder_text" in frame.columns:
            for row in frame.itertuples(index=False):
                records[row.entity_id] = getattr(row, "encoder_text")
            continue

        name_col = next((c for c in ("business_name", "raw_name", "norm_name") if c in frame.columns), None)
        addr_col = next((c for c in ("business_address", "raw_address", "norm_address") if c in frame.columns), None)
        if not name_col or not addr_col:
            raise ValueError(f"{path} is missing name/address columns: {list(frame.columns)}")

        for row in frame.itertuples(index=False):
            name_val = getattr(row, name_col, "")
            addr_val = getattr(row, addr_col, "")
            records[row.entity_id] = normalize_entity(name_val, addr_val)
    return records


def load_candidates(path: Path) -> List[Tuple[str, str]]:
    """Expand candidate_pairs.tsv into unique (S1, candidate) rows."""
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    required = {"source1_entity_id", "candidate_entity_ids"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    pairs: List[Tuple[str, str]] = []
    seen = set()
    for row in frame.itertuples(index=False):
        candidate_ids = [
            candidate.strip()
            for candidate in row.candidate_entity_ids.split(",")
            if candidate.strip()
        ]
        for candidate_id in candidate_ids:
            pair = (row.source1_entity_id, candidate_id)
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
    return pairs


def build_qwen_features(
    source1_paths: Iterable[Path],
    candidate_paths: Iterable[Path],
    candidate_file: Path,
    output_file: Path,
    model_name: str = DEFAULT_MODEL,
    instruction: str = DEFAULT_INSTRUCTION,
    max_seq_length: int = 256,
    batch_size: int = 32,
    device: Optional[str] = None,
) -> pd.DataFrame:
    """
    Encode all entities referenced by the candidate set and write pair features.

    The output contains one row per candidate pair and is safe to left-join
    into the Stage 3 feature table. Missing IDs are rejected instead of
    producing a misleading default score.
    """
    records = load_records(source1_paths)
    candidate_pairs = load_candidates(candidate_file)
    candidate_records = load_records(candidate_paths)
    records.update(candidate_records)

    missing_ids = sorted(
        {entity_id for pair in candidate_pairs for entity_id in pair}
        - set(records)
    )
    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} candidate IDs were not found in input records; "
            f"examples: {missing_ids[:5]}"
        )

    entity_ids = sorted({entity_id for pair in candidate_pairs for entity_id in pair})
    empty_text_ids = {
        entity_id for entity_id in entity_ids if not records[entity_id].strip()
    }
    encoder = QwenEntityEncoder(
        model_name=model_name,
        instruction=instruction,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        device=device,
    )
    embeddings = encoder.encode([records[entity_id] for entity_id in entity_ids])
    embedding_by_id = dict(zip(entity_ids, embeddings))

    rows = []
    for source1_id, candidate_id in candidate_pairs:
        similarity = float(np.dot(embedding_by_id[source1_id], embedding_by_id[candidate_id]))
        rows.append(
            {
                "source1_entity_id": source1_id,
                "candidate_entity_id": candidate_id,
                "qwen_cosine": similarity,
                "qwen_cosine_missing": int(
                    source1_id in empty_text_ids or candidate_id in empty_text_ids
                ),
            }
        )

    result = pd.DataFrame(
        rows,
        columns=["source1_entity_id", "candidate_entity_id", "qwen_cosine", "qwen_cosine_missing"],
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_file, sep="\t", index=False)
    metadata = {
        "model_name": model_name,
        "instruction": instruction,
        "max_seq_length": max_seq_length,
        "batch_size": batch_size,
        "normalized_embeddings": True,
        "candidate_pairs": len(result),
        "unique_entities": len(entity_ids),
    }
    with open(output_file.with_suffix(".json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    logger.info("Wrote %d Qwen pair features to %s", len(result), output_file)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Qwen Stage 2a-ii auxiliary pair features")
    parser.add_argument("--source1", type=Path, required=True)
    parser.add_argument("--source2", type=Path, required=True)
    parser.add_argument("--source3", type=Path, required=True)
    parser.add_argument("--candidate-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    build_qwen_features(
        source1_paths=[args.source1],
        candidate_paths=[args.source2, args.source3],
        candidate_file=args.candidate_file,
        output_file=args.output_file,
        model_name=args.model_name,
        instruction=args.instruction,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
