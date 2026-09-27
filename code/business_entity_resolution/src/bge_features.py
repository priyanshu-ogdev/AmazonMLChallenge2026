"""
Stage 2a: BGE-M3 Bi-Encoder pair features.

Encodes each unique entity referenced in candidate_pairs.tsv once using
the fine-tuned (or base) BGE-M3 model, and computes cosine similarity for
every candidate pair. Emits bge_features.tsv for left-joining into Stage 3 GBM.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.normalize import normalize_entity

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-m3"


class BGEEntityEncoder:
    """Batched BGE-M3 encoder with deterministic, L2-normalized dense outputs."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_seq_length: int = 128,  # Increased from 80: French legal names + addresses need ~100-120 tokens
        batch_size: int = 256,
        device: Optional[str] = None,
    ) -> None:
        try:
            import torch
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name, device=device)
            if device and "cuda" in device:
                self.model.half()  # 2x speedup and VRAM reduction on Ampere GPUs
            self.model.max_seq_length = max_seq_length
            self.model_name = model_name
            self.batch_size = batch_size
        except ImportError:
            raise ImportError(
                "sentence_transformers is required for BGEEntityEncoder. "
                "Install with: pip install sentence-transformers"
            )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode texts and return float32 unit vectors."""
        clean_texts = [str(text) if text else "[NO_ADDRESS]" for text in texts]
        if not clean_texts:
            dimension = getattr(self.model, "get_sentence_embedding_dimension", lambda: 1024)()
            return np.empty((0, dimension), dtype=np.float32)
            
        import torch
        # If multiple GPUs are available, use SentenceTransformer's native multi-processing!
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print(f"[GPU] Detected {torch.cuda.device_count()} GPUs! Distributing BGE-M3 workload across all devices...", flush=True)
            pool = self.model.start_multi_process_pool()
            embeddings = self.model.encode_multi_process(
                clean_texts,
                pool=pool,
                batch_size=self.batch_size,
                normalize_embeddings=True,
            )
            self.model.stop_multi_process_pool(pool)
        else:
            embeddings = self.model.encode(
                clean_texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
        return np.asarray(embeddings, dtype=np.float32)


def load_records(paths: Iterable[Path], needed_ids: Optional[Set[str]] = None) -> Dict[str, str]:
    """Load and normalize entity text from raw or Layer 0 normalized challenge TSV files."""
    records: Dict[str, str] = {}
    for path in paths:
        path = Path(path)
        frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
        if "entity_id" not in frame.columns:
            raise ValueError(f"{path} is missing required column: entity_id")

        if "encoder_text" in frame.columns:
            for row in frame.itertuples(index=False):
                eid = row.entity_id
                if needed_ids is not None and eid not in needed_ids:
                    continue
                records[eid] = getattr(row, "encoder_text")
            continue

        name_col = next((c for c in ("business_name", "raw_name", "norm_name") if c in frame.columns), None)
        addr_col = next((c for c in ("business_address", "raw_address", "norm_address") if c in frame.columns), None)
        if not name_col or not addr_col:
            raise ValueError(f"{path} is missing name/address columns: {list(frame.columns)}")

        for row in frame.itertuples(index=False):
            eid = row.entity_id
            if needed_ids is not None and eid not in needed_ids:
                continue
            name_val = getattr(row, name_col, "")
            addr_val = getattr(row, addr_col, "")
            records[eid] = normalize_entity(name_val, addr_val)
    return records


def load_candidates(path: Path, max_candidates_per_entity: Optional[int] = 15) -> List[Tuple[str, str]]:
    """Expand candidate_pairs.tsv into unique (S1, candidate) rows with optional per-entity candidate cap."""
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
    required = {"source1_entity_id", "candidate_entity_ids"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    pairs: List[Tuple[str, str]] = []
    cap = max_candidates_per_entity or 999999
    s1_vals = frame["source1_entity_id"].values
    cand_vals = frame["candidate_entity_ids"].values

    for i in range(len(s1_vals)):
        s1_id = s1_vals[i]
        cands = [c.strip() for c in cand_vals[i].split(",") if c.strip()][:cap]
        cand_seen = set()
        for cid in cands:
            if cid not in cand_seen:
                cand_seen.add(cid)
                pairs.append((s1_id, cid))
    return pairs


def build_bge_features(
    source1_paths: Iterable[Path],
    candidate_paths: Iterable[Path],
    candidate_file: Path,
    output_file: Path,
    model_name: str = DEFAULT_MODEL,
    max_seq_length: int = 64,
    batch_size: int = 256,
    device: Optional[str] = None,
    max_candidates_per_entity: int = 15,
) -> Path:
    """
    Encode candidate entities and stream BGE-M3 cosine features directly to TSV.
    Optimized for high-throughput batching, FP16 tensor core acceleration, and low RAM.
    """
    import csv as _csv
    import gc
    candidate_pairs = load_candidates(candidate_file, max_candidates_per_entity=max_candidates_per_entity)
    needed_ids = {pair[0] for pair in candidate_pairs} | {pair[1] for pair in candidate_pairs}

    records = load_records(source1_paths, needed_ids=needed_ids)
    candidate_records = load_records(candidate_paths, needed_ids=needed_ids)
    records.update(candidate_records)
    del candidate_records
    gc.collect()

    missing_ids = sorted(needed_ids - set(records))
    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} candidate IDs were not found in input records; "
            f"examples: {missing_ids[:5]}"
        )

    entity_ids = sorted(needed_ids)
    empty_text_ids = {
        entity_id for entity_id in entity_ids if not records[entity_id].strip()
    }
    logger.info("[BGE-M3] Encoding %d unique entities (batch_size=%d, max_seq_length=%d)...",
                len(entity_ids), batch_size, max_seq_length)
    encoder = BGEEntityEncoder(
        model_name=model_name,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        device=device,
    )
    embeddings = encoder.encode([records[entity_id] for entity_id in entity_ids])
    embeddings_fp16 = np.asarray(embeddings, dtype=np.float16)
    del embeddings
    del records
    gc.collect()

    id_to_idx = {entity_id: i for i, entity_id in enumerate(entity_ids)}
    n_pairs = len(candidate_pairs)
    s1_indices = np.empty(n_pairs, dtype=np.int32)
    cand_indices = np.empty(n_pairs, dtype=np.int32)
    for idx, (sid, cid) in enumerate(candidate_pairs):
        s1_indices[idx] = id_to_idx[sid]
        cand_indices[idx] = id_to_idx[cid]
    del candidate_pairs
    del id_to_idx
    gc.collect()

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    total_pairs = len(s1_indices)
    chunk_size = 200_000

    logger.info("[BGE-M3] Streaming %d candidate pair cosine similarities to %s...",
                total_pairs, output_file)
    with open(output_file, "w", encoding="utf-8", newline="") as out_f:
        writer = _csv.writer(out_f, delimiter="\t", quoting=_csv.QUOTE_NONE, escapechar="\\")
        writer.writerow(["source1_entity_id", "candidate_entity_id", "bge_cosine", "bge_cosine_missing"])

        for i in range(0, total_pairs, chunk_size):
            end = min(i + chunk_size, total_pairs)
            s1_sub = s1_indices[i:end]
            cand_sub = cand_indices[i:end]
            # Vectorized dot product on normalized FP16 vectors
            dots = np.sum(embeddings_fp16[s1_sub] * embeddings_fp16[cand_sub], axis=1, dtype=np.float32)
            dots = np.clip(dots, -1.0, 1.0)
            
            chunk_rows = []
            for j, sim in enumerate(dots):
                pair_idx = i + j
                s1_e = entity_ids[s1_indices[pair_idx]]
                cand_e = entity_ids[cand_indices[pair_idx]]
                is_miss = 1 if (s1_e in empty_text_ids or cand_e in empty_text_ids) else 0
                chunk_rows.append((s1_e, cand_e, f"{sim:.4f}", is_miss))
            writer.writerows(chunk_rows)

    del embeddings_fp16
    metadata = {
        "model_name": model_name,
        "max_seq_length": max_seq_length,
        "batch_size": batch_size,
        "normalized_embeddings": True,
        "candidate_pairs": total_pairs,
        "unique_entities": len(entity_ids),
        "max_candidates_per_entity": max_candidates_per_entity,
    }
    with open(output_file.with_suffix(".json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Stage 2a BGE-M3 cosine features for candidate pairs"
    )
    parser.add_argument("--source1", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate-sources", type=Path, nargs="+", default=None,
                        help="Candidate source TSV paths (e.g. source2.tsv source3.tsv)")
    parser.add_argument("--source2", type=Path, default=None, help="Candidate source 2 TSV (alias)")
    parser.add_argument("--source3", type=Path, default=None, help="Candidate source 3 TSV (alias)")
    parser.add_argument("--candidate-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None, help="Output TSV path")
    parser.add_argument("--output-file", type=Path, default=None, help="Output TSV path (alias)")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max-seq-length", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    output_path = args.output or args.output_file
    if not output_path:
        parser.error("Either --output or --output-file is required")

    candidate_sources = list(args.candidate_sources or [])
    if args.source2:
        candidate_sources.append(args.source2)
    if args.source3:
        candidate_sources.append(args.source3)
    if not candidate_sources:
        parser.error("Either --candidate-sources or --source2/--source3 is required")

    build_bge_features(
        source1_paths=args.source1,
        candidate_paths=candidate_sources,
        candidate_file=args.candidate_file,
        output_file=output_path,
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
