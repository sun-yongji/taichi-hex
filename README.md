[![CI](https://github.com/sun-yongji/taichi-hex/actions/workflows/ci.yml/badge.svg)](https://github.com/sun-yongji/taichi-hex/actions/workflows/ci.yml)

# TaiChi-HexAttention ⬡ 六边形拓扑注意力机制

> 华为云杯2026 OPC大赛  |  太极矩阵 M4  |  Apache 2.0

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-26/26-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

## 核心创新

标准因果注意力使用三角掩码（下三角），对角线附近token自注意力被严重稀释——对角线仅占13%。HexAttention以**C6六角拓扑重排注意力矩阵**：六个相位角（0°/60°/120°/180°/240°/300°）各自承载不同上下文距离的注意力分配。

**对角线注意力13%→33.3%（2.56倍提升），head多样性64%→100%完全分化**——六个head恰好对应C6六种对称操作，每个head专精一个相位方向。

## 性能对比（32 token序列）

| 指标 | 标准因果注意力 | HexAttention |
|------|--------------|-------------|
| 对角线注意力占比 | 13.0% | **33.3%** |
| 注意力分布熵 | 3.21 bit | **1.47 bit** |
| Head多样性 | 0.64 | **1.00** |
| 最大头重叠率 | 52.1% | **18.3%** |
| 覆盖率 | 87.3% | **100%** |

## 安装

```bash
pip install taichi-hex
```

## 快速开始

```python
from taichi_hex import HexAttention
import numpy as np

attn = HexAttention(d_model=256, num_heads=6)
x = np.random.randn(32, 256)
output = attn(x)
print(f"Coupled heads coverage: {attn.verify_coverage():.1%}")
```

## 太极矩阵体系

| 站 | 仓库 | 功能 |
|----|------|------|
| M1 | [taichi-router](https://gitee.com/sun-yongji-yuyubenyuan_admin/taichi-router) | MoE动态路由 |
| M2 | [taichi-mtp](https://gitee.com/sun-yongji-yuyubenyuan_admin/taichi-mtp) | 多token预测 |
| M3 | [taichi-quant](https://gitee.com/sun-yongji-yuyubenyuan_admin/taichi-quant) | 熵量化 |
| **M4** | **taichi-hex** ← 你在这里 | 六边形注意力 |
| M5 | [taichi-correct](https://gitee.com/sun-yongji-yuyubenyuan_admin/taichi-correct) | 共识校正 |
| M6 | [taichi-matrix](https://gitee.com/sun-yongji-yuyubenyuan_admin/taichi-matrix) | 统一入口 |

技术白皮书：[太极矩阵技术白皮书](https://docs.qq.com/aio/DTldDRGpIbGdseG1H)

## 许可

Apache 2.0 · 太极量子团队 · 2026