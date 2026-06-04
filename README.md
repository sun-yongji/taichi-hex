# TaiChi-HexAttention: Hexagonal Self-Attention

Six diagonally-masked attention heads — each covers `(j-i)%6 == head_id`.
C6 coupling mixes head outputs. 60° HexRoPE positional encoding.

## Core Idea

Standard self-attention: all pairs compute, O(N²) dense.

HexAttention: each of 6 heads attends to one diagonal stride.
The 6 heads together cover every (i,j) pair exactly once —
total information identical to full attention, but with structured sparsity.

## Quick Start

```python
from taichi_hex import HexAttention, compare_standard_vs_hex
import numpy as np

# Create hexagonal attention module
ha = HexAttention(d_model=384, causal=True)

# Forward pass
q = np.random.randn(64, 384)
k = np.random.randn(64, 384)
v = np.random.randn(64, 384)

out = ha(q, k, v)
print(out.coverage)  # 1.0 — all position pairs covered

# Compare with standard attention
cmp = compare_standard_vs_hex()
print(f"Head diversity: {cmp['head_diversity']:.3f}")
```

## Architecture

```
Input (seq_len, d_model)
    │
Q @ W_q, K @ W_k, V @ W_v
    │
Reshape → (6, seq_len, head_dim)
    │
HexRoPE (60° phase per position)
    │
Hexagonal diagonal mask per head
    (j-i)%6 == head_id
    │
Scaled dot-product attention (sparse per head)
    │
C6 coupling matrix: head mixing
    │
Concatenate → Output projection → (seq_len, d_model)
```

## License

Apache 2.0. Part of TaiChi-Matrix (CCF OSS 2026, M4).
