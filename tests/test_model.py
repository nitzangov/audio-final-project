"""Smoke tests for DeepSyncClassifier: forward pass, output shapes, no NaN."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.deepsync import DeepSyncClassifier

BATCH = 4
N_MELS = 128
N_CQT = 84
T_FRAMES = 200
NUM_CLASSES = 8


@pytest.fixture
def mel_input():
    return torch.randn(BATCH, 1, N_MELS, T_FRAMES)


@pytest.fixture
def cqt_input():
    return torch.randn(BATCH, 1, N_CQT, T_FRAMES)


class TestPhase1:
    def test_forward_shape(self, mel_input):
        model = DeepSyncClassifier(num_classes=NUM_CLASSES, phase=1)
        logits = model(mel_input)
        assert logits.shape == (BATCH, NUM_CLASSES)

    def test_no_nan(self, mel_input):
        model = DeepSyncClassifier(num_classes=NUM_CLASSES, phase=1)
        logits = model(mel_input)
        assert not torch.isnan(logits).any()

    def test_softmax_sums_to_one(self, mel_input):
        model = DeepSyncClassifier(num_classes=NUM_CLASSES, phase=1)
        logits = model(mel_input)
        probs = torch.softmax(logits, dim=1)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(BATCH), atol=1e-5)

    def test_grad_flows(self, mel_input):
        model = DeepSyncClassifier(num_classes=NUM_CLASSES, phase=1)
        logits = model(mel_input)
        loss = logits.sum()
        loss.backward()
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None


class TestPhase2:
    def test_forward_shape(self, mel_input, cqt_input):
        model = DeepSyncClassifier(num_classes=NUM_CLASSES, phase=2)
        logits = model(mel_input, cqt_input)
        assert logits.shape == (BATCH, NUM_CLASSES)

    def test_no_nan(self, mel_input, cqt_input):
        model = DeepSyncClassifier(num_classes=NUM_CLASSES, phase=2)
        logits = model(mel_input, cqt_input)
        assert not torch.isnan(logits).any()

    def test_requires_cqt(self, mel_input):
        model = DeepSyncClassifier(num_classes=NUM_CLASSES, phase=2)
        with pytest.raises(ValueError, match="CQT input required"):
            model(mel_input)


class TestPhase3:
    def test_forward_shape(self, mel_input, cqt_input):
        model = DeepSyncClassifier(num_classes=NUM_CLASSES, phase=3)
        logits = model(mel_input, cqt_input)
        assert logits.shape == (BATCH, NUM_CLASSES)

    def test_attention_weights(self, mel_input, cqt_input):
        model = DeepSyncClassifier(num_classes=NUM_CLASSES, phase=3)
        model(mel_input, cqt_input)
        weights = model.get_attention_weights()
        assert weights is not None
        assert weights.shape[0] == BATCH
        # Attention weights should sum to 1 along time dim
        sums = weights.sum(dim=1)
        assert torch.allclose(sums, torch.ones(BATCH), atol=1e-5)

    def test_no_nan(self, mel_input, cqt_input):
        model = DeepSyncClassifier(num_classes=NUM_CLASSES, phase=3)
        logits = model(mel_input, cqt_input)
        assert not torch.isnan(logits).any()


class TestAttentionGradientHealth:
    """Verify attention module gradients are healthy (not vanishing)."""

    def test_attention_grads_nonzero(self, mel_input, cqt_input):
        model = DeepSyncClassifier(num_classes=NUM_CLASSES, phase=3)
        logits = model(mel_input, cqt_input)
        loss = logits.sum()
        loss.backward()

        attn = model.attention
        for name, param in attn.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Attention param {name} has no gradient"
                grad_norm = param.grad.norm().item()
                assert grad_norm > 1e-8, (
                    f"Attention param {name} gradient is vanishing: {grad_norm:.2e}"
                )

    def test_gate_starts_near_zero(self):
        from src.models.attention import TemporalAttention

        attn = TemporalAttention(feature_dim=128)
        gate_value = torch.sigmoid(attn.gate).item()
        assert gate_value < 0.05, (
            f"Gate should start near 0 for mean-pool init, got {gate_value:.4f}"
        )


class TestFromConfig:
    def test_from_config(self):
        class MockConfig:
            class model:
                num_classes = 8
                phase = 1
                backbone_channels = [32, 64, 128]
                num_groups = 8
                dropout = 0.3
                attention_dim = 128
                classifier_hidden = 64

        model = DeepSyncClassifier.from_config(MockConfig())
        assert model.num_classes == 8
        assert model.phase == 1

    def test_param_count_phase1(self):
        model = DeepSyncClassifier(num_classes=8, phase=1)
        n = sum(p.numel() for p in model.parameters())
        assert n < 500_000, f"Phase 1 model should be lightweight, got {n:,} params"
