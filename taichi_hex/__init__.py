"""
TaiChi-HexAttention: Hexagonal Self-Attention with C6 Symmetry

Six diagonally-masked heads (j-i)%6 == head_id, C6 coupling,
and 60° HexRoPE — sparse per-head, full-coverage together.
"""

from .hex_attention import (
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
    HEX_COUPLING,
    NUM_HEADS,
    HEX_ANGLE,
    PHI,
    DIRECTION_NAMES,
)

__all__ = [
    "HexAttention",
    "HexAttnOutput",
    "hex_attention_mask",
    "full_hex_mask",
    "verify_coverage",
    "hex_rope",
    "hex_rope_positions",
    "hex_attention_entropy",
    "head_coverage_analysis",
    "compare_standard_vs_hex",
    "HEX_COUPLING",
    "NUM_HEADS",
    "HEX_ANGLE",
    "PHI",
    "DIRECTION_NAMES",
]

__version__ = "0.1.0"
