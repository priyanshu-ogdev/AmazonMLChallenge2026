"""
Stage 2b: Qwen3-0.6B causal generative-matcher pair features (stretch goal).

Emits qwen_matcher_prob for every candidate pair, in the exact same
(source1_entity_id, candidate_entity_id) TSV contract as bge_features.py and
qwen_features.py -- this file exists specifically so scoring.py's existing
merge_feature_file() can pick it up with zero changes to that function.
Stage 2b is optional by design (docs/07_stage2b_qwen3_generative_matcher_spec.md):
if the held-out-country gate fails, this file is simply never produced or
passed to scoring.py, and the GBM trains without the column -- no special-case
code is needed anywhere downstream for the "feature absent" case, because the
merge and monotonic-constraint machinery both already handle a missing
feature file / missing column gracefully.

Requires a LoRA-fine-tuned checkpoint at inference time (see
docs/07_stage2b_qwen3_generative_matcher_spec.md for the training spec:
verdict-token slicing, CachedMNRL-analogue causal LM training, anti-forgetting
stack). This module implements inference/feature-extraction only.
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

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
YES_TOKEN = "Yes"
NO_TOKEN = "No"

PROMPT_TEMPLATE = (
    "Task: Do the following two records refer to the same business entity? "
    "Answer Yes or No.\n"
    "Record 1: [COL] name [VAL] {name1} [COL] address [VAL] {address1} "
    "[COL] country [VAL] {country1}\n"
    "Record 2: [COL] name [VAL] {name2} [COL] address [VAL] {address2} "
    "[COL] country [VAL] {country2}\n"
    "Match:"
)


class QwenMatcherScorer:
    """
    Loads the LoRA-fine-tuned Qwen3-0.6B checkpoint and computes a sliced
    2-way softmax P(Match) at the verdict token position -- see
    docs/07_stage2b_qwen3_generative_matcher_spec.md Section 5.2.
    """

    def __init__(
        self,
        adapter_path: str,
        base_model_name: str = DEFAULT_MODEL,
        max_seq_length: int = 224,
        batch_size: int = 48,
        device: Optional[str] = None,
    ) -> None:
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers and peft are required for QwenMatcherScorer. "
                "Install with: pip install transformers peft torch"
            ) from exc

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        base = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16)
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.model.eval()
        if device:
            self.model.to(device)
        self.device = device
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

        # Resolve the fixed Yes/No label-word token ids once, not per batch.
        yes_ids = self.tokenizer.encode(YES_TOKEN, add_special_tokens=False)
        no_ids = self.tokenizer.encode(NO_TOKEN, add_special_tokens=False)
        if len(yes_ids) != 1 or len(no_ids) != 1:
            raise ValueError(
                f"Expected single-token Yes/No for {base_model_name}'s tokenizer; "
                f"got {yes_ids} / {no_ids}. Adjust YES_TOKEN/NO_TOKEN or the "
                "prompt template if this tokenizer splits them differently."
            )
        self.yes_id, self.no_id = yes_ids[0], no_ids[0]

    def score_pairs(self, prompts: Sequence[str]) -> np.ndarray:
        """Return P(Match) in [0, 1] for each prompt, via sliced 2-way softmax."""
        torch = self.torch
        probs: List[float] = []
        with torch.no_grad():
            for start in range(0, len(prompts), self.batch_size):
                batch = prompts[start : start + self.batch_size]
                encoded = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_length,
                )
                if self.device:
                    encoded = {k: v.to(self.device) for k, v in encoded.items()}
                # Verdict-token slicing (spec Section 4.2): only the final
                # position's hidden state is ever projected through the LM
                # head, so the logits tensor stays [batch, 1, vocab] rather
                # than [batch, seq_len, vocab].
                outputs = self.model(**encoded)
                final_logits = outputs.logits[:, -1, :]  # [batch, vocab]
                yes_no_logits = final_logits[:, [self.no_id, self.yes_id]]
                batch_probs = torch.softmax(yes_no_logits, dim=-1)[:, 1]  # P(Yes)
                probs.extend(batch_probs.float().cpu().tolist())
        return np.array(probs, dtype=np.float32)


def load_records(paths: Iterable[Path]) -> Dict[str, Dict[str, str]]:
    """Same contract as pair_features.py's load_records -- name/address/country per entity_id."""
    records: Dict[str, Dict[str, str]] = {}
    for path in paths:
        frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        for row in frame.to_dict("records"):
            eid = row["entity_id"]
            records[eid] = {
                "name": row.get("business_name", row.get("norm_name", "")),
                "address": row.get("business_address", row.get("norm_address", "")),
                "country": row.get("country", ""),
            }
    return records


def load_candidates(candidate_file: Path) -> List[Tuple[str, str]]:
    frame = pd.read_csv(candidate_file, sep="\t", dtype=str, keep_default_na=False)
    pairs: List[Tuple[str, str]] = []
    for row in frame.itertuples():
        source1_id = row.source1_entity_id
        raw = getattr(row, "candidate_entity_ids", "")
        for candidate_id in (c for c in raw.split(",") if c):
            pairs.append((source1_id, candidate_id))
    return pairs


def build_qwen_matcher_features(
    source1_paths: Sequence[Path],
    candidate_paths: Sequence[Path],
    candidate_file: Path,
    output_file: Path,
    adapter_path: str,
    base_model_name: str = DEFAULT_MODEL,
    max_seq_length: int = 224,
    batch_size: int = 48,
    device: Optional[str] = None,
) -> pd.DataFrame:
    """
    Emits qwen_matcher_features.tsv with columns:
      source1_entity_id, candidate_entity_id, qwen_matcher_prob, qwen_matcher_prob_missing
    -- the same contract as bge_features.py / qwen_features.py, so
    scoring.py's merge_feature_file() requires no changes to consume this.
    """
    records = load_records(source1_paths)
    records.update(load_records(candidate_paths))
    candidate_pairs = load_candidates(candidate_file)

    scorer = QwenMatcherScorer(
        adapter_path=adapter_path,
        base_model_name=base_model_name,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        device=device,
    )

    prompts: List[str] = []
    missing_flags: List[int] = []
    for source1_id, candidate_id in candidate_pairs:
        left = records.get(source1_id, {"name": "", "address": "", "country": ""})
        right = records.get(candidate_id, {"name": "", "address": "", "country": ""})
        empty = not (left["name"] or left["address"]) or not (right["name"] or right["address"])
        missing_flags.append(int(empty))
        prompts.append(
            PROMPT_TEMPLATE.format(
                name1=left["name"], address1=left["address"], country1=left["country"],
                name2=right["name"], address2=right["address"], country2=right["country"],
            )
        )

    probs = scorer.score_pairs(prompts)

    result = pd.DataFrame(
        {
            "source1_entity_id": [p[0] for p in candidate_pairs],
            "candidate_entity_id": [p[1] for p in candidate_pairs],
            "qwen_matcher_prob": probs,
            "qwen_matcher_prob_missing": missing_flags,
        }
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_file, sep="\t", index=False)
    metadata = {
        "adapter_path": adapter_path,
        "base_model_name": base_model_name,
        "max_seq_length": max_seq_length,
        "batch_size": batch_size,
        "candidate_pairs": len(result),
    }
    with open(output_file.with_suffix(".json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    logger.info("Wrote %d Qwen matcher features to %s", len(result), output_file)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Stage 2b Qwen3-0.6B generative-matcher features (stretch goal)"
    )
    parser.add_argument("--source1", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate-sources", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter-path", type=str, required=True,
                         help="Path to the LoRA adapter checkpoint that passed the held-out-country gate")
    parser.add_argument("--base-model-name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max-seq-length", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    build_qwen_matcher_features(
        source1_paths=args.source1,
        candidate_paths=args.candidate_sources,
        candidate_file=args.candidate_file,
        output_file=args.output,
        adapter_path=args.adapter_path,
        base_model_name=args.base_model_name,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
