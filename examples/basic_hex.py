"""
TaiChi-HexAttention demo: visualize hexagonal attention patterns.
"""

import numpy as np
from taichi_hex import (
    HexAttention,
    full_hex_mask,
    hex_attention_entropy,
    head_coverage_analysis,
    compare_standard_vs_hex,
    DIRECTION_NAMES,
)


def show_mask_patterns(seq_len: int = 18):
    """Display the 6 hexagonal diagonal mask patterns."""
    masks = full_hex_mask(seq_len, causal=True)

    print("=" * 70)
    print("  Hexagonal Attention Mask Patterns (causal, lower triangle)")
    print("  ● = attend, · = masked")
    print("=" * 70)

    for h in range(6):
        print(f"\n  Head {h} ({DIRECTION_NAMES[h]})  stride={(h,)}  "
              f"condition: (j-i)%6 == {h}")
        print("     ", end="")
        for j in range(seq_len):
            print(f"{j:3d}", end="")
        print()

        for i in range(seq_len):
            print(f"  {i:3d}", end="")
            for j in range(seq_len):
                if j > i:
                    print("  ·", end="")
                else:
                    print("  ●" if masks[h, i, j] else "  ·", end="")
            print()
    print()


def show_attention_distribution(seq_len: int = 24):
    """Demonstrate hexagonal attention in action."""
    rng = np.random.default_rng(42)
    ha = HexAttention(d_model=384, causal=True, rng=rng)

    q = rng.normal(0, 0.1, (seq_len, 384)).astype(np.float64)
    out = ha(q, q, q)

    print("=" * 70)
    print("  HexAttention Forward Pass Results")
    print("=" * 70)
    print(f"  Seq length:     {seq_len}")
    print(f"  Model dim:      384")
    print(f"  Heads:          6")
    print(f"  Head dim:       64")
    print(f"  Coverage:       {out.coverage:.1%}")
    print()

    # Per-head statistics
    entropy = hex_attention_entropy(out.attn_weights)
    coverage_info = head_coverage_analysis(out.attn_weights)

    print("  Per-Head Attention Statistics:")
    print(f"  {'Head':>6s} {'Direction':>10s} {'Active%':>8s} {'Entropy':>9s}")
    print("  " + "-" * 40)
    for h in range(6):
        active = coverage_info["per_head_active"][h] * 100
        ent = float(np.mean(entropy[h]))
        print(f"  {h:6d} {DIRECTION_NAMES[h]:>10s} {active:7.1f}% {ent:9.4f}")

    print(f"\n  Head diversity (cosine distance): {coverage_info['head_diversity']:.4f}")
    print(f"  Avg coverage per head:            {coverage_info['avg_coverage']:.1%}")
    print()


def compare():
    """Compare standard vs hexagonal attention."""
    print("=" * 70)
    print("  Standard vs Hexagonal Attention Comparison")
    print("=" * 70)

    for seq_len in [16, 32, 64]:
        cmp = compare_standard_vs_hex(seq_len=seq_len, d_model=192)
        print(f"\n  seq_len={seq_len}:")
        print(f"    Standard entropy: {cmp['standard_entropy']:.4f}")
        print(f"    Hex mean entropy: {cmp['hex_mean_entropy']:.4f}")
        print(f"    Hex coverage:     {cmp['hex_coverage']:.1%}")
        print(f"    Head diversity:   {cmp['head_diversity']:.4f}")

    print()


def main():
    show_mask_patterns(seq_len=18)
    show_attention_distribution(seq_len=24)
    compare()
    print("  TaiChi-HexAttention M4 demo complete.")


if __name__ == "__main__":
    main()
