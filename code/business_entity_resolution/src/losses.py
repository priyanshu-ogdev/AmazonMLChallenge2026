"""
Custom loss functions for bi-encoder LoRA fine-tuning.

Primary loss: CachedMultipleNegativesRankingLoss (MNRL)
  — contrastive loss with gradient caching to decouple effective batch size
    from physical VRAM. Optimizes ranking (retrieval quality).

Secondary loss: Self-distillation anchor loss
  — cosine distance between fine-tuned and frozen base model embeddings.
    Penalizes drift from the pretrained multilingual embedding space.
    This is the second layer of the anti-forgetting stack (after LoRA itself).

Design decisions (from docs_final/training.md):
  - MNRL over raw triplet loss: triplet loss can cause severe collapse
    depending on LR/setup. MNRL optimizes ranking directly.
  - Self-distillation weight=0.10: mid-point of 0.05-0.15 range.
    A compliant, no-external-data substitute for replay training.
  - The distillation term adds ~50% compute overhead (extra forward pass).
    Acceptable for short sequences (80 tokens) on 568M model.

IMPORTANT: MNRL optimizes ranking, not absolute cosine values. The GBM
  consumes raw cosine similarity as a feature, so calibration (isotonic or
  Platt) happens downstream at Stage 3, not here.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
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
        over all input texts in the batch.

    Architecture:
        - self.model: the training model (BGE-M3 + LoRA adapter)
        - self.frozen_model: frozen copy of base BGE-M3 (no LoRA, no grad)
        - self.cached_mnrl: the contrastive loss with gradient caching

    Memory overhead: frozen model adds ~1.14 GB (568M × 2 bytes bf16).
    Compute overhead: one extra forward pass per batch (~50%).

    Args:
        model: The SentenceTransformer model being fine-tuned.
        frozen_model: A frozen copy of the base model (before LoRA).
        mini_batch_size: GradCache chunk size for CachedMNRL.
        distill_weight: Weight for the distillation loss term.
        scale: Temperature scaling for the contrastive loss.
    """

    def __init__(
        self,
        model: SentenceTransformer,
        frozen_model: SentenceTransformer,
        mini_batch_size: int = 16,
        distill_weight: float = 0.10,
        scale: float = 20.0,
    ):
        super().__init__()
        self.model = model
        self.distill_weight = distill_weight

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
        Compute combined MNRL + self-distillation loss.

        Flow:
        1. Ensure frozen model is on the correct device
        2. Compute self-distillation loss (forward through both models)
        3. Compute CachedMNRL loss (uses gradient caching internally)
        4. Return weighted sum

        Distillation is computed FIRST to avoid any potential mutation of
        sentence_features by CachedMNRL's internal gradient caching mechanism.
        """
        sentence_features = list(sentence_features)
        self._ensure_device_sync(sentence_features)

        # --- Self-distillation loss ---
        # Clone features to avoid any mutation issues from subsequent MNRL processing
        distill_features = [
            {k: v.clone() for k, v in sf.items() if isinstance(v, torch.Tensor)}
            for sf in sentence_features
        ]
        distill_loss = self._compute_distillation(distill_features)

        # --- Main contrastive loss ---
        mnrl_loss = self.cached_mnrl(sentence_features, labels)

        # --- Combined loss ---
        total_loss = mnrl_loss + self.distill_weight * distill_loss

        return total_loss

    def _compute_distillation(
        self, features: List[Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """
        Compute cosine distance between current and frozen embeddings.

        For each text column in the batch:
        1. Get frozen model embedding (no grad — just reference)
        2. Get current model embedding (with grad — backprop updates LoRA)
        3. Compute mean cosine distance (1 - cos_sim)

        This directly penalizes drift from the pretrained embedding space,
        including on tokens/patterns the training data never exercises
        (like French-specific vocabulary).
        """
        total_distance = torch.tensor(0.0, device=features[0]["input_ids"].device)
        count = 0

        for sf in features:
            # Frozen model embeddings — no gradient computation needed
            with torch.no_grad():
                frozen_output = self.frozen_model(sf)
                frozen_emb = frozen_output["sentence_embedding"]

            # Current model embeddings — gradient needed for LoRA backprop
            current_output = self.model(sf)
            current_emb = current_output["sentence_embedding"]

            # Cosine distance: 1 - cos_sim(current, frozen)
            cos_sim = F.cosine_similarity(current_emb, frozen_emb, dim=-1)
            distance = (1.0 - cos_sim).mean()
            total_distance = total_distance + distance
            count += 1

        return total_distance / max(count, 1)

    def get_config_dict(self) -> dict:
        """Return configuration for logging/serialization."""
        return {
            "distill_weight": self.distill_weight,
            "mini_batch_size": self.cached_mnrl.mini_batch_size
            if hasattr(self.cached_mnrl, "mini_batch_size")
            else "unknown",
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
