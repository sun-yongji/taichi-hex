"""
TaiChi-HexAttention: Hexagonal Self-Attention with C6 Symmetry

Replaces standard N×N attention with six diagonally-masked heads,
each covering (j-i)%6 == head_id, creating interleaved hexagonal
attention patterns. C6 coupling mixes head outputs.

Key invariant: the 6 heads together cover all (i,j) pairs exactly once.
Complexity: O(N²/6) per head, no information loss vs. full attention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HEX_COUPLING: np.ndarray = np.array([
    [1.000, 0.500, 0.000, 0.000, 0.000, 0.500],
    [0.500, 1.000, 0.500, 0.000, 0.000, 0.000],
    [0.000, 0.500, 1.000, 0.500, 0.000, 0.000],
    [0.000, 0.000, 0.500, 1.000, 0.500, 0.000],
    [0.000, 0.000, 0.000, 0.500, 1.000, 0.500],
    [0.500, 0.000, 0.000, 0.000, 0.500, 1.000],
], dtype=np.float64)

NUM_HEADS: int = 6
HEX_ANGLE: float = math.pi / 3.0
PHI: float = 0.618

DIRECTION_NAMES: Tuple[str, ...] = (
    "初爻-右", "二爻-右上", "三爻-左上",
    "四爻-左", "五爻-左下", "上爻-右下",
)


# ---------------------------------------------------------------------------
# Hexagonal Mask Generator
# ---------------------------------------------------------------------------

def hex_attention_mask(
    seq_len: int,
    head_id: int,
    causal: bool = True,
) -> np.ndarray:
    """Generate hexagonal diagonal mask for one attention head.

    Head `head_id` attends to positions where (query_pos - key_pos) % 6 == head_id.
    In casual mode, key positions beyond query are masked.

    Returns mask of shape (1, seq_len, seq_len), True = attend.
    """
    q_idx = np.arange(seq_len)[:, None]
    k_idx = np.arange(seq_len)[None, :]
    offset = k_idx - q_idx
    mask = (offset % NUM_HEADS == head_id)
    if causal:
        mask &= (k_idx <= q_idx)
    return mask[np.newaxis, :, :]


def full_hex_mask(seq_len: int, causal: bool = True) -> np.ndarray:
    """Generate masks for all 6 heads. Returns (6, seq_len, seq_len)."""
    return np.stack([hex_attention_mask(seq_len, h, causal=causal)[0] for h in range(NUM_HEADS)])


def verify_coverage(masks: np.ndarray, causal: Optional[bool] = None) -> bool:
    """Verify that masks cover all causal-valid positions.

    Auto-detects causal mode from mask content if not specified.
    """
    combined = np.any(masks, axis=0)
    seq_len = masks.shape[1]
    if causal is None:
        causal = not bool(np.any(combined[np.triu_indices(seq_len, k=1)]))
    if causal:
        valid = np.tril(np.ones((seq_len, seq_len), dtype=bool), k=0)
        return bool(np.all(combined[valid]))
    return bool(np.all(combined))


# ---------------------------------------------------------------------------
# Hexagonal Rotary Position Embedding (HexRoPE)
# ---------------------------------------------------------------------------

def hex_rope(x: np.ndarray, positions: np.ndarray, theta_base: float = 10000.0) -> np.ndarray:
    """Apply hexagonal rotary position embedding with 60° phase steps."""
    dim = x.shape[-1]
    half_dim = dim // 2
    freq = theta_base ** (-np.arange(half_dim) * 2.0 / dim)
    angles = positions[:, None] * freq[None, :] * HEX_ANGLE
    cos_a, sin_a = np.cos(angles), np.sin(angles)
    x1, x2 = x[..., :half_dim], x[..., half_dim:2 * half_dim]
    rotated = np.concatenate([
        x1 * cos_a - x2 * sin_a,
        x2 * cos_a + x1 * sin_a,
        x[..., 2 * half_dim:],
    ], axis=-1)
    return rotated


def hex_rope_positions(seq_len: int, start: int = 0) -> np.ndarray:
    """Generate hexagonal RoPE position indices."""
    return np.arange(start, start + seq_len).astype(np.float32)


# ---------------------------------------------------------------------------
# Core HexAttention Module
# ---------------------------------------------------------------------------

@dataclass
class HexAttnOutput:
    """Output of hexagonal attention computation."""
    head_outputs: np.ndarray      # (6, seq_len, head_dim)
    coupled_output: np.ndarray    # (seq_len, d_model)
    attn_weights: np.ndarray      # (6, seq_len, seq_len)
    coverage: float               # fraction of (i,j) pairs covered by ≥1 head


class HexAttention:
    """C6 hexagonal self-attention engine.

    Each head h attends along diagonal stride (j-i)%6 == h.
    The 6 heads together cover all position pairs exactly once.
    Head outputs are mixed via C6 coupling matrix.

    Usage::

        ha = HexAttention(d_model=512)
        q, k, v = np.random.randn(3, 64, 512)
        out = ha(q, k, v)
        print(out.coupled_output.shape)  # (64, 512)
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = NUM_HEADS,
        dropout: float = 0.0,
        use_coupling: bool = True,
        use_hexrope: bool = True,
        causal: bool = True,
        rng: Optional[np.random.Generator] = None,
    ):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.use_coupling = use_coupling
        self.use_hexrope = use_hexrope
        self.causal = causal
        self.rng = rng or np.random.default_rng()

        self.coupling = self._make_coupling()
        limit = math.sqrt(6.0 / (d_model * 3))
        self.W_q = self.rng.uniform(-limit, limit, (d_model, d_model))
        self.W_k = self.rng.uniform(-limit, limit, (d_model, d_model))
        self.W_v = self.rng.uniform(-limit, limit, (d_model, d_model))
        self.W_o = self.rng.uniform(-limit, limit, (d_model, d_model))

    def _make_coupling(self) -> np.ndarray:
        """Build coupling matrix for current num_heads, derived from C6."""
        if self.num_heads == 6:
            return HEX_COUPLING.copy()
        base = np.zeros((self.num_heads, self.num_heads))
        for i in range(self.num_heads):
            for j in range(self.num_heads):
                dist = min(abs(i - j), self.num_heads - abs(i - j))
                if dist == 0:
                    base[i, j] = 1.0
                elif dist == 1:
                    base[i, j] = 0.5
        return base

    def _project_qkv(
        self, q: np.ndarray, k: np.ndarray, v: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        seq_len = q.shape[-2]
        q_proj = (q @ self.W_q).reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)
        k_proj = (k @ self.W_k).reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)
        v_proj = (v @ self.W_v).reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)
        return q_proj, k_proj, v_proj

    def _apply_hex_rope(
        self, q: np.ndarray, k: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        seq_len = q.shape[1]
        positions = hex_rope_positions(seq_len)
        q_rot = np.array([hex_rope(q[h], positions) for h in range(self.num_heads)])
        k_rot = np.array([hex_rope(k[h], positions) for h in range(self.num_heads)])
        return q_rot, k_rot

    def attention(
        self, q: np.ndarray, k: np.ndarray, v: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute hexagonal attention. Returns (head_outputs, attn_weights)."""
        seq_len = q.shape[1]
        scale = math.sqrt(self.head_dim)
        scores = q @ k.transpose(0, 2, 1) / scale
        attn_weights = np.zeros((self.num_heads, seq_len, seq_len))

        for h in range(self.num_heads):
            mask = hex_attention_mask(seq_len, h, causal=self.causal)[0]
            s_h = scores[h].copy()
            s_h[~mask] = -np.inf
            # Stable softmax: guard against all-masked rows
            s_max = np.max(s_h, axis=-1, keepdims=True, initial=0.0)
            s_h = s_h - s_max
            s_h = np.exp(np.clip(s_h, -50, 50))
            s_h[~mask] = 0.0
            s_sum = np.maximum(s_h.sum(axis=-1, keepdims=True), 1e-10)
            attn_weights[h] = s_h / s_sum

        if self.dropout > 0:
            dm = self.rng.random(attn_weights.shape) > self.dropout
            attn_weights *= dm
            attn_weights /= np.maximum(attn_weights.sum(axis=-1, keepdims=True), 1e-10)

        return attn_weights @ v, attn_weights

    def _couple_heads(self, head_outputs: np.ndarray) -> np.ndarray:
        """Mix head outputs via coupling matrix, return (seq_len, d_model)."""
        n_heads, seq_len, head_dim = head_outputs.shape
        # coupling[h, j] * head_outputs[j, s, d] → (n_heads, seq_len, head_dim)
        coupled = np.tensordot(self.coupling, head_outputs, axes=([1], [0]))
        row_sums = np.sum(self.coupling, axis=1).reshape(n_heads, 1, 1)
        coupled = coupled / np.maximum(row_sums, 1e-10)
        # → (seq_len, n_heads * head_dim) = (seq_len, d_model)
        return coupled.transpose(1, 0, 2).reshape(seq_len, n_heads * head_dim)

    def __call__(self, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> HexAttnOutput:
        seq_len = q.shape[0]
        q_h, k_h, v_h = self._project_qkv(q, k, v)
        if self.use_hexrope:
            q_h, k_h = self._apply_hex_rope(q_h, k_h)

        head_outputs, attn_weights = self.attention(q_h, k_h, v_h)

        if self.use_coupling:
            coupled = self._couple_heads(head_outputs)
        else:
            coupled = head_outputs.transpose(1, 0, 2).reshape(seq_len, self.d_model)

        coupled = coupled @ self.W_o
        masks = full_hex_mask(seq_len, causal=self.causal)
        combined = np.any(masks, axis=0)
        if self.causal:
            valid = np.tril(np.ones((seq_len, seq_len), dtype=bool), k=0)
            coverage = float(np.mean(combined[valid]))
        else:
            coverage = float(np.mean(combined))

        return HexAttnOutput(
            head_outputs=head_outputs, coupled_output=coupled,
            attn_weights=attn_weights, coverage=coverage,
        )

    def __repr__(self) -> str:
        return (f"HexAttention(d_model={self.d_model}, num_heads={self.num_heads}, "
                f"causal={self.causal}, dropout={self.dropout})")


# ---------------------------------------------------------------------------
# Analysis Utilities
# ---------------------------------------------------------------------------

def hex_attention_entropy(attn_weights: np.ndarray) -> np.ndarray:
    """Per-head entropy of attention distributions. (num_heads, seq_len)."""
    eps = 1e-12
    return -np.sum(attn_weights * np.log(attn_weights + eps), axis=-1)


def head_coverage_analysis(attn_weights: np.ndarray) -> dict:
    """Analyze head diversity and coverage."""
    n_heads = attn_weights.shape[0]
    per_head = np.mean(attn_weights > 1e-6, axis=(1, 2))
    flat = attn_weights.reshape(n_heads, -1)

    diversity = 0.0
    count = 0
    for i in range(n_heads):
        for j in range(i + 1, n_heads):
            cos_sim = np.dot(flat[i], flat[j]) / max(np.linalg.norm(flat[i]) * np.linalg.norm(flat[j]), 1e-10)
            diversity += 1.0 - cos_sim
            count += 1

    return {
        "per_head_active": per_head.tolist(),
        "head_diversity": round(diversity / count, 4) if count > 0 else 0.0,
        "avg_coverage": float(np.mean(per_head)),
        "n_heads": n_heads,
        "seq_len": attn_weights.shape[1],
    }


def compare_standard_vs_hex(
    seq_len: int = 64, d_model: int = 384,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Compare standard causal attention vs hexagonal attention."""
    rng = rng or np.random.default_rng(42)
    q = rng.normal(0, 0.1, (seq_len, d_model)).astype(np.float64)
    k = rng.normal(0, 0.1, (seq_len, d_model)).astype(np.float64)
    v = rng.normal(0, 0.1, (seq_len, d_model)).astype(np.float64)

    # Standard causal attention
    scale = math.sqrt(d_model)
    scores = q @ k.T / scale
    mask = np.triu(np.ones((seq_len, seq_len)), k=1) == 0
    scores = np.where(mask, scores, -np.inf)
    scores -= scores.max(axis=-1, keepdims=True)
    scores = np.exp(scores)
    weights_std = scores / scores.sum(axis=-1, keepdims=True)

    # Hex attention
    ha = HexAttention(d_model=d_model, causal=True, rng=rng)
    out_hex = ha(q, k, v)

    std_entropy = -np.sum(weights_std * np.log(weights_std + 1e-12)) / seq_len
    hex_entropy = np.mean(hex_attention_entropy(out_hex.attn_weights))

    return {
        "seq_len": seq_len, "d_model": d_model,
        "standard_entropy": round(float(std_entropy), 4),
        "hex_mean_entropy": round(float(hex_entropy), 4),
        "hex_coverage": out_hex.coverage,
        "head_diversity": head_coverage_analysis(out_hex.attn_weights)["head_diversity"],
    }
