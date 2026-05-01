# Sparse Attention 推理加速

## 概述

Sparse Attention（稀疏注意力）是一种推理加速技术，通过有选择地计算部分 Query Token 的注意力输出，显著减少 Self-Attention 的计算量。

## 工作原理

视频生成中，每帧的 Spatial Token 包含大量冗余信息（相邻位置或相似区域的注意力模式高度相关）。Sparse Attention 利用这一特性：

1. **Query 选择**：在每次 Self-Attention 计算中，只选择 `sparse_attention_ratio` 比例的 Query Token 进行计算
2. **KV 保持完整**：Key 和 Value 保持全部 Token，保留完整上下文信息
3. **最近邻填充**：未被选择的 Query Token 的注意力输出，从同一时间帧内最近的计算位置复制

### Token 选择模式

- **uniform（均匀采样）**：在空间网格上均匀间隔采样，确保覆盖整个空间区域
- **random（随机采样）**：基于固定种子的确定性随机选择（seed=42，确保可复现性）

## 使用方法

### 命令行参数

```bash
# 以 50% 比例启用稀疏注意力
python generate_infinitetalk.py \
    --ckpt_dir ./weights \
    --infinitetalk_dir ./weights/InfiniteTalk \
    --input_json examples.json \
    --use_sparse_attention \
    --sparse_attention_ratio 0.5

# 更激进的加速（30% token 计算）
python generate_infinitetalk.py \
    --ckpt_dir ./weights \
    --infinitetalk_dir ./weights/InfiniteTalk \
    --input_json examples.json \
    --use_sparse_attention \
    --sparse_attention_ratio 0.3
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use_sparse_attention` | flag | False | 启用稀疏注意力 |
| `--sparse_attention_ratio` | float | 0.5 | Query Token 保留比例 (0.0-1.0)，越低越快但质量损失越大 |

## 与 TeaCache 的兼容性

Sparse Attention 与 TeaCache 实现了**不同的加速维度**：
- **TeaCache**：跳过整个 Block 的计算（时间维度上的缓存）
- **Sparse Attention**：减少单次 Self-Attention 计算量（空间维度上的稀疏化）

两者可以同时使用，加速效果叠加。

## 性能预期

| Ratio (r) | 理论 Self-Attention 加速 | 推荐场景 |
|-----------|------------------------|---------|
| 1.0       | 1x (无加速)               | 质量优先 |
| 0.5       | ~2x                    | 平衡模式 |
| 0.3       | ~3x                    | 快速预览 |

**注意**：实际收益受 Video Length 影响。长视频生成中（流式模式），Sparse Attention 在每一帧的优化累计效果更显著。

## 实现文件

- `wan/modules/attention.py`：`SparseAttentionConfig` 配置类、`create_sparse_q_mask()` 遮罩生成、`sparse_attention_wrapper()` 核心函数
- `wan/modules/multitalk_model.py`：`WanSelfAttention.forward` 中集成稀疏注意力调用
- `wan/multitalk.py`：`generate_infinitetalk` 中初始化和关闭逻辑

## 已知限制

- 当前仅作用于 Self-Attention（视觉 Tokens 的自注意力），不影响 Cross-Attention
- 仅在推理时生效，不影响训练
- 极低 Ratio（< 0.2）可能导致明显的质量退化
