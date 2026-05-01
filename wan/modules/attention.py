# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import torch.nn as nn
from einops import rearrange, repeat
from ..utils.multitalk_utils import RotaryPositionalEmbedding1D, normalize_and_scale, split_token_counts_and_frame_ids
from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)
import xformers.ops

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

__all__ = [
    'flash_attention',
    'attention',
    'SparseAttentionConfig',
    'create_sparse_q_mask',
    'sparse_attention_wrapper',
]


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out
    

class SingleStreamAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        encoder_hidden_states_dim: int,
        num_heads: int,
        qkv_bias: bool,
        qk_norm: bool,
        norm_layer: nn.Module,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.encoder_hidden_states_dim = encoder_hidden_states_dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qk_norm = qk_norm

        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim, eps=eps) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim,eps=eps) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.kv_linear = nn.Linear(encoder_hidden_states_dim, dim * 2, bias=qkv_bias)

        self.add_q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.add_k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

    def forward(self, x: torch.Tensor, encoder_hidden_states: torch.Tensor, shape=None, enable_sp=False, kv_seq=None) -> torch.Tensor:
       
        N_t, N_h, N_w = shape
        if not enable_sp:
            x = rearrange(x, "B (N_t S) C -> (B N_t) S C", N_t=N_t)

        # get q for hidden_state
        B, N, C = x.shape
        q = self.q_linear(x)
        q_shape = (B, N, self.num_heads, self.head_dim)
        q = q.view(q_shape).permute((0, 2, 1, 3))

        if self.qk_norm:
            q = self.q_norm(q)
        
        # get kv from encoder_hidden_states
        _, N_a, _ = encoder_hidden_states.shape
        encoder_kv = self.kv_linear(encoder_hidden_states)
        encoder_kv_shape = (B, N_a, 2, self.num_heads, self.head_dim)
        encoder_kv = encoder_kv.view(encoder_kv_shape).permute((2, 0, 3, 1, 4)) 
        encoder_k, encoder_v = encoder_kv.unbind(0)

        if self.qk_norm:
            encoder_k = self.add_k_norm(encoder_k)


        q = rearrange(q, "B H M K -> B M H K")
        encoder_k = rearrange(encoder_k, "B H M K -> B M H K")
        encoder_v = rearrange(encoder_v, "B H M K -> B M H K")

        if enable_sp:
            # context parallel
            sp_size = get_sequence_parallel_world_size()
            sp_rank = get_sequence_parallel_rank()
            visual_seqlen, _ = split_token_counts_and_frame_ids(N_t, N_h * N_w, sp_size, sp_rank)
            assert kv_seq is not None, f"kv_seq should not be None."
            attn_bias = xformers.ops.fmha.attn_bias.BlockDiagonalMask.from_seqlens(visual_seqlen, kv_seq)
        else:
            attn_bias = None
        x = xformers.ops.memory_efficient_attention(q, encoder_k, encoder_v, attn_bias=attn_bias, op=None,)
        x = rearrange(x, "B M H K -> B H M K") 

        # linear transform
        x_output_shape = (B, N, C)
        x = x.transpose(1, 2) 
        x = x.reshape(x_output_shape) 
        x = self.proj(x)
        x = self.proj_drop(x)

        if not enable_sp:
            # reshape x to origin shape
            x = rearrange(x, "(B N_t) S C -> B (N_t S) C", N_t=N_t)

        return x

class SingleStreamMutiAttention(SingleStreamAttention):
    def __init__(
        self,
        dim: int,
        encoder_hidden_states_dim: int,
        num_heads: int,
        qkv_bias: bool,
        qk_norm: bool,
        norm_layer: nn.Module,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        eps: float = 1e-6,
        class_range: int = 24,
        class_interval: int = 4,
    ) -> None:
        super().__init__(
            dim=dim,
            encoder_hidden_states_dim=encoder_hidden_states_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            norm_layer=norm_layer,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            eps=eps,
        )
        self.class_interval = class_interval
        self.class_range = class_range
        self.rope_h1  = (0, self.class_interval)
        self.rope_h2  = (self.class_range - self.class_interval, self.class_range)
        self.rope_bak = int(self.class_range // 2)

        self.rope_1d = RotaryPositionalEmbedding1D(self.head_dim)

    def forward(self, 
                x: torch.Tensor, 
                encoder_hidden_states: torch.Tensor, 
                shape=None, 
                x_ref_attn_map=None,
                human_num=None) -> torch.Tensor:
        
        encoder_hidden_states = encoder_hidden_states.squeeze(0)
        if human_num == 1:
            return super().forward(x, encoder_hidden_states, shape)

        N_t, _, _ = shape 
        x = rearrange(x, "B (N_t S) C -> (B N_t) S C", N_t=N_t) 

        # get q for hidden_state
        B, N, C = x.shape
        q = self.q_linear(x) 
        q_shape = (B, N, self.num_heads, self.head_dim) 
        q = q.view(q_shape).permute((0, 2, 1, 3))

        if self.qk_norm:
            q = self.q_norm(q)

  
        max_values = x_ref_attn_map.max(1).values[:, None, None] 
        min_values = x_ref_attn_map.min(1).values[:, None, None] 
        max_min_values = torch.cat([max_values, min_values], dim=2)

        human1_max_value, human1_min_value = max_min_values[0, :, 0].max(), max_min_values[0, :, 1].min()
        human2_max_value, human2_min_value = max_min_values[1, :, 0].max(), max_min_values[1, :, 1].min()

        human1 = normalize_and_scale(x_ref_attn_map[0], (human1_min_value, human1_max_value), (self.rope_h1[0], self.rope_h1[1]))
        human2 = normalize_and_scale(x_ref_attn_map[1], (human2_min_value, human2_max_value), (self.rope_h2[0], self.rope_h2[1]))
        back   = torch.full((x_ref_attn_map.size(1),), self.rope_bak, dtype=human1.dtype).to(human1.device)
        max_indices = x_ref_attn_map.argmax(dim=0)
        normalized_map = torch.stack([human1, human2, back], dim=1)
        normalized_pos = normalized_map[range(x_ref_attn_map.size(1)), max_indices] # N 

        q = rearrange(q, "(B N_t) H S C -> B H (N_t S) C", N_t=N_t)
        q = self.rope_1d(q, normalized_pos)
        q = rearrange(q, "B H (N_t S) C -> (B N_t) H S C", N_t=N_t)

        _, N_a, _ = encoder_hidden_states.shape 
        encoder_kv = self.kv_linear(encoder_hidden_states) 
        encoder_kv_shape = (B, N_a, 2, self.num_heads, self.head_dim)
        encoder_kv = encoder_kv.view(encoder_kv_shape).permute((2, 0, 3, 1, 4)) 
        encoder_k, encoder_v = encoder_kv.unbind(0) 

        if self.qk_norm:
            encoder_k = self.add_k_norm(encoder_k)

        
        per_frame = torch.zeros(N_a, dtype=encoder_k.dtype).to(encoder_k.device)
        per_frame[:per_frame.size(0)//2] = (self.rope_h1[0] + self.rope_h1[1]) / 2
        per_frame[per_frame.size(0)//2:] = (self.rope_h2[0] + self.rope_h2[1]) / 2
        encoder_pos = torch.concat([per_frame]*N_t, dim=0)
        encoder_k = rearrange(encoder_k, "(B N_t) H S C -> B H (N_t S) C", N_t=N_t)
        encoder_k = self.rope_1d(encoder_k, encoder_pos)
        encoder_k = rearrange(encoder_k, "B H (N_t S) C -> (B N_t) H S C", N_t=N_t)

 
        q = rearrange(q, "B H M K -> B M H K")
        encoder_k = rearrange(encoder_k, "B H M K -> B M H K")
        encoder_v = rearrange(encoder_v, "B H M K -> B M H K")
        x = xformers.ops.memory_efficient_attention(q, encoder_k, encoder_v, attn_bias=None, op=None,)
        x = rearrange(x, "B M H K -> B H M K")

        # linear transform
        x_output_shape = (B, N, C)
        x = x.transpose(1, 2) 
        x = x.reshape(x_output_shape) 
        x = self.proj(x) 
        x = self.proj_drop(x)

        # reshape x to origin shape
        x = rearrange(x, "(B N_t) S C -> B (N_t S) C", N_t=N_t) 

        return x


@dataclass
class SparseAttentionConfig:
    """
    Configuration for sparse attention.

    Attributes:
        enabled: Whether sparse attention is enabled.
        ratio: Fraction of query tokens to compute full attention for (0.0-1.0).
               Lower = faster but more approximate.
        pattern: Sampling pattern for query token selection.
                 'uniform' = evenly spaced grid sampling.
                 'random' = random sampling (seed-deterministic).
    """
    enabled: bool = False
    ratio: float = 0.5
    pattern: str = 'uniform'


def create_sparse_q_mask(
    lq: int,
    grid_sizes: torch.Tensor,
    ratio: float,
    pattern: str = 'uniform',
) -> torch.Tensor:
    """
    Create a boolean mask for selecting a subset of query positions.

    For video tokens, the layout is: [t0_spatial, t1_spatial, ..., tF_spatial]
    where each temporal slice has H*W spatial tokens.

    Args:
        lq: Total number of query tokens.
        grid_sizes: [B, 3] tensor with (F, H, W) grid dimensions.
        ratio: Fraction of tokens to keep (0.0-1.0).
        pattern: 'uniform' (grid subsampling) or 'random'.

    Returns:
        bool mask of shape [lq], True for tokens to compute.
    """
    device = grid_sizes.device
    f, h, w = grid_sizes[0].tolist()
    spatial_per_frame = h * w
    total_needed = f * spatial_per_frame

    # If lq is padded (larger than actual tokens), create mask only for actual tokens
    actual_lq = min(lq, total_needed)
    num_selected = max(1, int(actual_lq * ratio))

    mask = torch.zeros(lq, dtype=torch.bool, device=device)

    if pattern == 'uniform':
        # Uniform spatial subsampling: keep every Nth spatial position
        tokens_per_frame = spatial_per_frame
        keep_per_frame = max(1, int(tokens_per_frame * ratio))
        if keep_per_frame >= tokens_per_frame:
            mask[:actual_lq] = True
        else:
            step = max(1, tokens_per_frame // keep_per_frame)
            for t in range(f):
                offset = t * tokens_per_frame
                for idx in range(0, tokens_per_frame, step):
                    pos = offset + idx
                    if pos < actual_lq:
                        mask[pos] = True
    elif pattern == 'random':
        # Deterministic random selection (reproducible)
        rng_state = torch.random.get_rng_state()
        torch.manual_seed(42)
        perm = torch.randperm(actual_lq, device=device)
        torch.random.set_rng_state(rng_state)
        selected_indices = perm[:num_selected]
        mask[selected_indices] = True
    else:
        # Default: keep all
        mask[:actual_lq] = True

    return mask


def sparse_attention_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sparse_config: SparseAttentionConfig,
    grid_sizes: torch.Tensor,
    **flash_kwargs,
) -> torch.Tensor:
    """
    Attention wrapper that applies sparse query selection.

    Full keys and values are kept for context quality, but only a subset of
    query positions (controlled by ratio) have their attention computed.
    Uncomputed positions copy the output from the nearest computed neighbor
    in the same spatial grid position.

    Args:
        q: [B, Lq, Nq, C1] query tensor.
        k: [B, Lk, Nk, C1] key tensor.
        v: [B, Lk, Nk, C2] value tensor.
        sparse_config: Configuration controlling sparsity.
        grid_sizes: [B, 3] tensor with (F, H, W).
        **flash_kwargs: Additional kwargs passed to flash_attention.

    Returns:
        x: [B, Lq, Nq, C2] full attention output.
    """
    if not sparse_config.enabled or sparse_config.ratio >= 1.0:
        return flash_attention(q, k, v, **flash_kwargs)

    b, lq, nq, c1 = q.shape
    c2 = v.shape[-1]

    # Create sparse query mask
    q_mask = create_sparse_q_mask(lq, grid_sizes, sparse_config.ratio, sparse_config.pattern)
    selected_indices = q_mask.nonzero(as_tuple=True)[0]  # [num_selected]

    if len(selected_indices) == 0:
        # Fallback: compute all
        return flash_attention(q, k, v, **flash_kwargs)

    # Select query positions
    q_selected = q[:, selected_indices, :, :]  # [B, num_selected, Nq, C1]

    # Compute attention only for selected queries, full K/V context
    flash_kwargs_copy = dict(flash_kwargs)
    # When q is modified, drop q_lens to let flash_attention handle it uniformly
    if 'q_lens' in flash_kwargs_copy:
        flash_kwargs_copy.pop('q_lens')

    out_selected = flash_attention(
        q=q_selected,
        k=k,
        v=v,
        **flash_kwargs_copy,
    )  # [B, num_selected, Nq, C2]

    # Expand output back to full length
    out = torch.zeros(b, lq, nq, c2, dtype=out_selected.dtype, device=out_selected.device)

    # Place computed outputs
    out[:, selected_indices, :, :] = out_selected

    # Fill uncomputed positions by copying nearest computed neighbor
    unselected_mask = ~q_mask
    unselected_indices = unselected_mask.nonzero(as_tuple=True)[0]

    if len(unselected_indices) > 0:
        # Use spatial-aware nearest neighbor: same temporal frame, nearest spatial
        f, h, w = grid_sizes[0].tolist()
        tokens_per_frame = h * w

        for pos in unselected_indices:
            pos_int = pos.item()
            if pos_int >= lq:
                continue
            # Find nearest selected index in the same temporal frame
            frame_idx = pos_int // tokens_per_frame
            frame_start = frame_idx * tokens_per_frame
            frame_end = min(frame_idx * tokens_per_frame + tokens_per_frame, lq)

            # Selected indices within this frame
            in_frame_mask = (selected_indices >= frame_start) & (selected_indices < frame_end)
            in_frame = selected_indices[in_frame_mask]

            if len(in_frame) > 0:
                nearest = in_frame[torch.argmin(torch.abs(in_frame - pos_int))]
                out[:, pos_int:pos_int + 1, :, :] = out[:, nearest:nearest + 1, :, :]
            else:
                # Fallback: overall nearest selected
                nearest = selected_indices[torch.argmin(torch.abs(selected_indices - pos_int))]
                out[:, pos_int:pos_int + 1, :, :] = out[:, nearest:nearest + 1, :, :]

    return out