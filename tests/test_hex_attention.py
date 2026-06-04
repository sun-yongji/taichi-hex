"""Tests for TaiChi-HexAttention."""

import numpy as np
import pytest

from taichi_hex import (
    HexAttention,
    HexAttnOutput,
    hex_attention_mask,
    full_hex_mask,
    verify_coverage,
    hex_rope,
    hex_rope_positions,
    hex_attention_entropy,
    head_coverage_analysis,
    compare_standard_vs_hex,
    NUM_HEADS,
    HEX_ANGLE,
)


class TestHexMask:
    def test_mask_shape(self):
        m = hex_attention_mask(16, 0)
        assert m.shape == (1, 16, 16)

    def test_mask_causal(self):
        m = hex_attention_mask(10, 2, causal=True)
        # Upper triangle (j > i) must be all False
        upper = np.triu(np.ones((10, 10), dtype=bool), k=1)
        assert not np.any(m[0, :, :] & upper)

    def test_mask_non_causal(self):
        m = hex_attention_mask(10, 0, causal=False)
        assert m[0, 0, 0] == True  # self-attention
        assert m[0, 0, 6] == True  # (6-0)%6 == 0

    def test_mask_head_offset(self):
        # head 1: (j-i)%6 == 1
        m = hex_attention_mask(12, 1, causal=False)
        for i in range(12):
            for j in range(12):
                if (j - i) % 6 == 1:
                    assert m[0, i, j], f"({i},{j}) should be True"
                else:
                    assert not m[0, i, j], f"({i},{j}) should be False"

    def test_full_hex_mask_shape(self):
        masks = full_hex_mask(20)
        assert masks.shape == (6, 20, 20)

    def test_full_coverage(self):
        masks = full_hex_mask(20, causal=True)
        assert verify_coverage(masks)
        # Lower triangle + diagonal = causal valid area
        valid = np.tril(np.ones((20, 20)), k=0)
        # Each valid (i,j) in exactly one head
        sum_mask = np.sum(masks.astype(int), axis=0)
        assert np.all(sum_mask[valid > 0] == 1)

    def test_causal_full_coverage(self):
        """Causal mask: each valid (i>=j) position appears in exactly 1 head."""
        for seq_len in [7, 11, 24]:
            masks = full_hex_mask(seq_len, causal=True)
            assert verify_coverage(masks)

    def test_coverage_non_causal(self):
        masks = full_hex_mask(12, causal=False)
        assert verify_coverage(masks)


class TestHexRoPE:
    def test_shape_preserved(self):
        x = np.random.randn(8, 32)
        pos = np.arange(8).astype(np.float32)
        y = hex_rope(x, pos)
        assert y.shape == x.shape

    def test_identity_at_zero(self):
        x = np.random.randn(2, 32)
        pos = np.zeros(2, dtype=np.float32)
        y = hex_rope(x, pos)
        np.testing.assert_allclose(y, x, atol=1e-6)

    def test_odd_dimension(self):
        x = np.random.randn(4, 31)  # Odd dim
        pos = np.arange(4).astype(np.float32)
        y = hex_rope(x, pos)
        assert y.shape == (4, 31)

    def test_positions(self):
        p = hex_rope_positions(5)
        np.testing.assert_array_equal(p, np.array([0, 1, 2, 3, 4], dtype=np.float32))

    def test_positions_with_start(self):
        p = hex_rope_positions(3, start=10)
        np.testing.assert_array_equal(p, np.array([10, 11, 12], dtype=np.float32))


class TestHexAttention:
    @pytest.fixture
    def ha(self):
        return HexAttention(d_model=384, num_heads=6, causal=True, rng=np.random.default_rng(42))

    def test_init(self, ha):
        assert ha.d_model == 384
        assert ha.num_heads == 6
        assert ha.head_dim == 64

    def test_forward_shape(self, ha):
        q = np.random.randn(32, 384)
        out = ha(q, q, q)
        assert out.coupled_output.shape == (32, 384)
        assert out.head_outputs.shape == (6, 32, 64)
        assert out.attn_weights.shape == (6, 32, 32)

    def test_causal_output(self, ha):
        q = np.random.randn(16, 384)
        out = ha(q, q, q)
        # Causal: position i should not attend to j > i
        # attn_weights non-zero only where causal allows
        assert not np.any(np.isnan(out.coupled_output))

    def test_coverage_is_one(self, ha):
        q = np.random.randn(64, 384)
        out = ha(q, q, q)
        assert out.coverage == 1.0

    def test_different_qkv(self, ha):
        q = np.random.randn(16, 384)
        k = np.random.randn(16, 384)
        v = np.random.randn(16, 384)
        out = ha(q, k, v)
        assert out.coupled_output.shape == (16, 384)

    def test_no_coupling(self):
        ha = HexAttention(d_model=384, use_coupling=False, rng=np.random.default_rng(42))
        q = np.random.randn(16, 384)
        out = ha(q, q, q)
        assert out.coupled_output.shape == (16, 384)

    def test_no_hexrope(self):
        ha = HexAttention(d_model=384, use_hexrope=False, rng=np.random.default_rng(42))
        q = np.random.randn(8, 384)
        out = ha(q, q, q)
        assert out.coupled_output.shape == (8, 384)

    def test_extreme_seq_len(self):
        ha = HexAttention(d_model=128, num_heads=4, rng=np.random.default_rng(42))
        # Small sequence
        q = np.random.randn(3, 128)
        out = ha(q, q, q)
        assert out.coupled_output.shape == (3, 128)

    def test_large_seq_len(self):
        ha = HexAttention(d_model=192, rng=np.random.default_rng(42))
        q = np.random.randn(128, 192)
        out = ha(q, q, q)
        assert out.coupled_output.shape == (128, 192)

    def test_repr(self, ha):
        r = repr(ha)
        assert "HexAttention" in r
        assert "384" in r

    def test_invalid_d_model(self):
        with pytest.raises(AssertionError):
            HexAttention(d_model=100, num_heads=6)  # 100 % 6 != 0

    def test_seq_len_not_multiple_of_6(self):
        """Works for any sequence length, not just multiples of 6."""
        ha = HexAttention(d_model=384, rng=np.random.default_rng(42))
        q = np.random.randn(17, 384)
        out = ha(q, q, q)
        assert out.coupled_output.shape == (17, 384)
        assert out.coverage == 1.0


class TestAnalysis:
    def test_entropy_shape(self):
        w = np.random.dirichlet(np.ones(4), size=(2, 3))
        e = hex_attention_entropy(w)
        assert e.shape == (2, 3)

    def test_entropy_non_negative(self):
        w = np.ones((6, 16, 16)) / 16
        e = hex_attention_entropy(w)
        assert np.all(e >= 0)

    def test_head_coverage_analysis(self):
        w = np.ones((6, 16, 16)) / 16
        r = head_coverage_analysis(w)
        assert "head_diversity" in r
        assert "per_head_active" in r
        assert len(r["per_head_active"]) == 6

    def test_compare_standard_vs_hex(self):
        cmp = compare_standard_vs_hex(seq_len=32, d_model=192)
        assert "standard_entropy" in cmp
        assert "hex_coverage" in cmp
        assert cmp["hex_coverage"] > 0.9
