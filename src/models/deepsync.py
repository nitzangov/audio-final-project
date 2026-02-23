"""DeepSync classifier: phased architecture from Mel-only to dual-stream + attention."""

import torch
import torch.nn as nn

from src.models.attention import TemporalAttention
from src.models.backbone import CNNBackbone


class DeepSyncClassifier(nn.Module):
    """Multi-phase genre classifier.

    Phase 1: Mel-only CNN -> GlobalAvgPool -> FC head
    Phase 2: Mel CNN + CQT CNN -> Concat -> GlobalAvgPool -> FC head
    Phase 3: Mel CNN + CQT CNN -> Concat -> TemporalAttention -> FC head
    """

    def __init__(
        self,
        num_classes: int = 8,
        phase: int = 1,
        backbone_channels: list[int] | None = None,
        num_groups: int = 8,
        dropout: float = 0.3,
        attention_dim: int = 128,
        classifier_hidden: int = 64,
    ):
        super().__init__()
        self.phase = phase
        self.num_classes = num_classes
        backbone_channels = backbone_channels or [32, 64, 128]

        self.mel_backbone = CNNBackbone(channels=backbone_channels, num_groups=num_groups)

        if phase >= 2:
            self.cqt_backbone = CNNBackbone(channels=backbone_channels, num_groups=num_groups)
            feature_dim = self.mel_backbone.out_channels * 2
        else:
            self.cqt_backbone = None
            feature_dim = self.mel_backbone.out_channels

        self.use_attention = phase >= 3
        if self.use_attention:
            self.attention = TemporalAttention(feature_dim, attention_dim)

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_classes),
        )

        self._attention_weights = None

    def forward(self, mel: torch.Tensor, cqt: torch.Tensor | None = None):
        """
        Args:
            mel: (B, 1, n_mels, T)
            cqt: (B, 1, n_cqt_bins, T) — required for phase >= 2, ignored otherwise
        Returns:
            logits: (B, num_classes)
        """
        mel_feat = self.mel_backbone(mel)  # (B, C, T')

        if self.phase >= 2:
            if cqt is None:
                raise ValueError("CQT input required for phase >= 2")
            cqt_feat = self.cqt_backbone(cqt)  # (B, C, T')

            # Align temporal dims (they should match, but handle edge cases)
            min_t = min(mel_feat.shape[2], cqt_feat.shape[2])
            mel_feat = mel_feat[:, :, :min_t]
            cqt_feat = cqt_feat[:, :, :min_t]

            combined = torch.cat([mel_feat, cqt_feat], dim=1)  # (B, 2C, T')
        else:
            combined = mel_feat  # (B, C, T')

        if self.use_attention:
            # Attention expects (B, T, D)
            combined_t = combined.permute(0, 2, 1)  # (B, T', D)
            pooled, self._attention_weights = self.attention(combined_t)
        else:
            # Global average pool over temporal dimension
            pooled = combined.mean(dim=2)  # (B, D)

        logits = self.classifier(pooled)
        return logits

    def get_attention_weights(self) -> torch.Tensor | None:
        """Return the last computed attention weights for visualization."""
        return self._attention_weights

    @staticmethod
    def from_config(config) -> "DeepSyncClassifier":
        """Construct a DeepSyncClassifier from a Config object."""
        return DeepSyncClassifier(
            num_classes=config.model.num_classes,
            phase=config.model.phase,
            backbone_channels=config.model.backbone_channels,
            num_groups=config.model.num_groups,
            dropout=config.model.dropout,
            attention_dim=getattr(config.model, "attention_dim", 128),
            classifier_hidden=config.model.classifier_hidden,
        )
