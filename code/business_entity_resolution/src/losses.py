"""
Custom loss functions for bi-encoder LoRA fine-tuning.

Primary loss: CachedMultipleNegativesRankingLoss (CachedMNRL)
  — contrastive ranking loss with gradient caching to decouple effective
    batch size from physical VRAM. Processes activations in mini_batch_size
    chunks, accumulating gradients before the optimizer step.

Secondary loss: Self-distillation anchor loss
  — cosine distance between fine-tuned and frozen base model embeddings,
    also chunked to match CachedMNRL’s mini_batch_size window.
    Penalizes drift from the pretrained multilingual embedding space.
    This is the second layer of the anti-forgetting stack (after LoRA itself).

Design decisions (from docs/06_stage2a_bge_m3_training_spec.md):
  - CachedMNRL over raw triplet loss: MNRL optimizes ranking directly and
    avoids triplet collapse. GradCache further decouples effective batch size
    from VRAM by processing activations in mini_batch_size windows.
  - Self-distillation weight=0.10: mid-point of 0.05-0.15 range.
    A compliant, no-external-data substitute for replay training.
  - Compute overhead: CachedMNRL executes 2 forward passes per chunk (cache +
    backprop). Self-distillation adds 1 forward pass with grad through the
    trainable model and 1 pass without grad through the frozen model (~100%
    overhead per distilled column). To prevent hard negatives from multiplying
    this cost, distill_anchor_positive_only limits distillation to anchor and
    positive text columns.
  - Peak activation memory is bounded to mini_batch_size window across all passes.

IMPORTANT: CachedMNRL optimizes ranking, not absolute cosine values. The GBM
  consumes raw cosine similarity as a feature, so calibration (isotonic or
  Platt) happens downstream at Stage 3, not here.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
try:
    from sentence_transformers.sentence_transformer.losses import CachedMultipleNegativesRankingLoss
except ImportError:
    from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from typing import Optional, Dict, List, Iterable
import logging

logger = logging.getLogger(__name__)


class DistillationCachedMNRL(nn.Module):
    """
    CachedMultipleNegativesRankingLoss + Self-Distillation Anchor Loss.

    Combines contrastive learning with a regularization term that penalizes
    drift from the frozen pretrained model's embedding space. This protects
    multilingual capabilities critical for French generalization.

    Loss = MNRL_loss + distill_weight × distillation_loss

    Where:
        distillation_loss = mean(1 - cosine_similarity(current_emb, frozen_emb))
        over input texts in the batch.

    Architecture:
        - self.model: the training model (BGE-M3 + LoRA adapter)
        - self.frozen_model: frozen copy of base BGE-M3 (no LoRA, no grad)
        - self.cached_mnrl: the contrastive loss with gradient caching

    Memory overhead: frozen model adds ~1.14 GB (568M × 2 bytes bf16).
    Compute overhead: CachedMNRL does 2 forward passes; distillation adds 2 passes
    (1 trainable with grad + 1 frozen no_grad), totaling ~100% additional compute
    per distilled text column.

    Args:
        model: The SentenceTransformer model being fine-tuned.
        frozen_model: A frozen copy of the base model (before LoRA).
        mini_batch_size: GradCache chunk size for CachedMNRL.
        distill_weight: Weight for the distillation loss term.
        scale: Temperature scaling for the contrastive loss.
        distill_anchor_positive_only: Restrict distillation to anchor + positive
            columns (default: True), preventing hard negatives from linearly
            scaling distillation forward pass cost.
    """

    def __init__(
        self,
        model: SentenceTransformer,
        frozen_model: SentenceTransformer,
        mini_batch_size: int = 16,
        distill_weight: float = 0.10,
        scale: float = 20.0,
        distill_anchor_positive_only: bool = True,
    ):
        super().__init__()
        self.model = model
        self.distill_weight = distill_weight
        self.mini_batch_size = mini_batch_size   # keep for chunked distillation
        self.distill_anchor_positive_only = distill_anchor_positive_only

        # Main contrastive loss with gradient caching
        self.cached_mnrl = CachedMultipleNegativesRankingLoss(
            model=model,
            mini_batch_size=mini_batch_size,
            scale=scale,
        )

        # Frozen model for distillation — completely detached from training
        self.frozen_model = frozen_model
        self.frozen_model.eval()
        for param in self.frozen_model.parameters():
            param.requires_grad = False

        self._device_synced = False

        logger.info(
            f"DistillationCachedMNRL initialized: "
            f"distill_weight={distill_weight}, mini_batch_size={mini_batch_size}, "
            f"scale={scale}"
        )

    def _ensure_device_sync(self, sentence_features: List[Dict[str, torch.Tensor]]):
        """Lazily move frozen model to the same device as input tensors."""
        if not self._device_synced:
            device = sentence_features[0]["input_ids"].device
            frozen_device = next(self.frozen_model.parameters()).device
            if frozen_device != device:
                logger.info(f"Moving frozen model from {frozen_device} to {device}")
                self.frozen_model = self.frozen_model.to(device)
            self._device_synced = True

    def forward(
        self,
        sentence_features: Iterable[Dict[str, torch.Tensor]],
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute combined CachedMNRL + self-distillation loss.

        Flow:
        1. Ensure frozen model is on the correct device
        2. Compute self-distillation loss in mini_batch_size chunks
           (matches CachedMNRL’s own chunking to keep peak VRAM bounded)
        3. Compute CachedMNRL loss (uses gradient caching internally)
        4. Return weighted sum

        Distillation is chunked BEFORE CachedMNRL so that its gradient
        accumulation does not interact with CachedMNRL’s own internal
        caching state. Both are bounded to mini_batch_size activations
        in memory at any one time.
        """
        sentence_features = list(sentence_features)
        self._ensure_device_sync(sentence_features)

        # --- Self-distillation loss (chunked to mini_batch_size) ---
        distill_loss = self._compute_distillation_chunked(sentence_features)

        # --- Main contrastive loss ---
        mnrl_loss = self.cached_mnrl(sentence_features, labels)

        # --- Combined loss ---
        total_loss = mnrl_loss + self.distill_weight * distill_loss

        return total_loss

    def _compute_distillation_chunked(
        self, features: List[Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """
        Compute cosine distance between current and frozen embeddings,
        chunked to mini_batch_size to mirror CachedMNRL’s own memory budget.

        BUG FIX: The original _compute_distillation() ran a single forward
        pass over the full physical batch (e.g. 48 samples) outside
        CachedMNRL’s chunk loop, allocating full-batch activation tensors and
        completely defeating the purpose of GradCache. This version slices
        each sentence-feature dict’s tensors to mini_batch_size rows and
        accumulates the mean cosine distance across chunks — peak activation
        memory is now bounded to one chunk, matching CachedMNRL’s footprint.

        For each text column in the batch:
        1. Slice tensors to mini_batch_size chunks
        2. Get frozen model embedding (no grad — just reference)
        3. Get current model embedding (with grad — backprop updates LoRA)
        4. Accumulate mean cosine distance (1 - cos_sim) across chunks
        """
        device = features[0]["input_ids"].device
        total_distance = torch.tensor(0.0, device=device)
        total_samples = 0

        distill_features = features[:2] if self.distill_anchor_positive_only and len(features) > 2 else features
        for sf in distill_features:
            # Determine the batch dimension of this sentence-feature dict
            batch_size = next(iter(sf.values())).shape[0]

            for start in range(0, batch_size, self.mini_batch_size):
                end = min(start + self.mini_batch_size, batch_size)

                # Slice tensors to the current chunk
                chunk = {
                    k: v[start:end] if isinstance(v, torch.Tensor) else v
                    for k, v in sf.items()
                }

                # Frozen model embeddings — no gradient computation needed
                with torch.no_grad():
                    frozen_output = self.frozen_model(chunk)
                    frozen_emb = frozen_output["sentence_embedding"]

                # Current model embeddings — gradient needed for LoRA backprop
                current_output = self.model(chunk)
                current_emb = current_output["sentence_embedding"]

                # Cosine distance: 1 - cos_sim(current, frozen), sample-weighted across chunks
                cos_sim = F.cosine_similarity(current_emb, frozen_emb, dim=-1)
                total_distance = total_distance + (1.0 - cos_sim).sum()
                total_samples += current_emb.size(0)

        return total_distance / max(total_samples, 1)

    def get_config_dict(self) -> dict:
        """Return configuration for logging/serialization."""
        return {
            "distill_weight": self.distill_weight,
            "mini_batch_size": self.cached_mnrl.mini_batch_size
            if hasattr(self.cached_mnrl, "mini_batch_size")
            else "unknown",
            "distill_anchor_positive_only": self.distill_anchor_positive_only,
            "model_name": getattr(self.model, "model_card_data", {}).get(
                "base_model", "unknown"
            ),
        }


def create_loss(
    model: SentenceTransformer,
    use_distillation: bool = True,
    distill_weight: float = 0.10,
    mini_batch_size: int = 16,
    scale: float = 20.0,
    frozen_model: Optional[SentenceTransformer] = None,
    distill_anchor_positive_only: bool = True,
) -> nn.Module:
    """
    Factory function to create the appropriate loss.

    If use_distillation=True, returns DistillationCachedMNRL (requires frozen_model).
    If use_distillation=False, returns plain CachedMultipleNegativesRankingLoss.

    This separation allows easy A/B testing of the distillation contribution.
    """
    if use_distillation:
        if frozen_model is None:
            raise ValueError(
                "frozen_model is required when use_distillation=True. "
                "Load a separate copy of the base model with all parameters frozen."
            )
        logger.info(
            f"Creating DistillationCachedMNRL "
            f"(distill_weight={distill_weight}, mini_batch_size={mini_batch_size})"
        )
        return DistillationCachedMNRL(
            model=model,
            frozen_model=frozen_model,
            mini_batch_size=mini_batch_size,
            distill_weight=distill_weight,
            scale=scale,
            distill_anchor_positive_only=distill_anchor_positive_only,
        )
    else:
        logger.info(
            f"Creating CachedMultipleNegativesRankingLoss "
            f"(mini_batch_size={mini_batch_size})"
        )
        return CachedMultipleNegativesRankingLoss(
            model=model,
            mini_batch_size=mini_batch_size,
            scale=scale,
        )
