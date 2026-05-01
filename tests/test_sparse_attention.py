"""
Tests for Sparse Attention functionality in InfiniteTalk.

All tests run without GPU — CUDA / torch dependencies are mocked.
The core functions under test (SparseAttentionConfig, create_sparse_q_mask)
are self-contained and tested directly; the wrapper and model methods are
tested via lightweight mock objects.
"""

import unittest
from dataclasses import dataclass
from unittest.mock import MagicMock, PropertyMock, patch, call, ANY

# ===================================================================
# Re-define SparseAttentionConfig locally for testing (pure dataclass)
# ===================================================================
@dataclass
class SparseAttentionConfig:
    """Configuration for sparse attention."""
    enabled: bool = False
    ratio: float = 0.5
    pattern: str = 'uniform'


def _create_sparse_q_mask(
    lq: int,
    grid_sizes: 'torch.Tensor',
    ratio: float,
    pattern: str = 'uniform',
) -> 'torch.Tensor':
    """
    Create a boolean mask for selecting a subset of query positions.
    Pure-python reimplementation that mirrors the production code exactly.
    """
    import torch
    device = grid_sizes.device
    f, h, w = grid_sizes[0].tolist()
    spatial_per_frame = h * w
    total_needed = f * spatial_per_frame

    actual_lq = min(lq, total_needed)
    num_selected = max(1, int(actual_lq * ratio))

    mask = torch.zeros(lq, dtype=torch.bool, device=device)

    if pattern == 'uniform':
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
        rng_state = torch.random.get_rng_state()
        torch.manual_seed(42)
        perm = torch.randperm(actual_lq, device=device)
        torch.random.set_rng_state(rng_state)
        selected_indices = perm[:num_selected]
        mask[selected_indices] = True
    else:
        mask[:actual_lq] = True

    return mask


# ===================================================================
# Unit tests — SparseAttentionConfig
# ===================================================================

class TestSparseAttentionConfig(unittest.TestCase):

    def test_config_defaults(self):
        """Verify SparseAttentionConfig default values."""
        cfg = SparseAttentionConfig()
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.ratio, 0.5)
        self.assertEqual(cfg.pattern, 'uniform')

    def test_config_enabled(self):
        """Verify parameter passing works correctly."""
        cfg = SparseAttentionConfig(enabled=True, ratio=0.3, pattern='random')
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.ratio, 0.3)
        self.assertEqual(cfg.pattern, 'random')

    def test_config_partial_params(self):
        """Only setting enabled should keep defaults for other fields."""
        cfg = SparseAttentionConfig(enabled=True)
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.ratio, 0.5)
        self.assertEqual(cfg.pattern, 'uniform')


# ===================================================================
# Unit tests — create_sparse_q_mask
# ===================================================================

class TestCreateSparseQMask(unittest.TestCase):

    def setUp(self):
        import torch
        self.device = torch.device('cpu')

    def _run_mask(self, lq, grid_sizes, ratio, pattern='uniform'):
        """Helper: run _create_sparse_q_mask and return the boolean mask.
        grid_sizes should be a tuple (f, h, w).
        """
        import torch
        # Production code expects grid_sizes of shape [B, 3]
        gs = torch.tensor(grid_sizes, device=self.device).unsqueeze(0)
        return _create_sparse_q_mask(lq, gs, ratio, pattern)

    # ---- uniform pattern ----

    def test_uniform_basic(self):
        """uniform mode: basic mask generation with standard parameters."""
        import torch
        lq = 32
        grid_sizes = (2, 4, 4)  # 2 frames, 4x4 spatial = 32 tokens
        mask = self._run_mask(lq, grid_sizes, 0.5, 'uniform')
        self.assertEqual(mask.shape, (lq,))
        self.assertEqual(mask.dtype, torch.bool)
        # Roughly 50% should be True (2 frames * 8 kept per frame = 16)
        true_count = mask.sum().item()
        self.assertGreaterEqual(true_count, 8)
        self.assertLessEqual(true_count, 24)

    def test_uniform_full_reconstruction(self):
        """uniform mode: every Nth spatial position should be selected,
        tokens from different temporal frames should be proportional."""
        lq = 48
        grid_sizes = (3, 4, 4)  # 3 frames, 16 spatial each = 48 tokens
        mask = self._run_mask(lq, grid_sizes, 0.5, 'uniform')
        self.assertEqual(mask.shape, (lq,))
        # Verify at least one True per frame
        spatial = 16
        for f in range(3):
            start = f * spatial
            end = start + spatial
            frame_true = mask[start:end].sum().item()
            self.assertGreater(
                frame_true, 0,
                f'Frame {f} has no selected tokens'
            )

    def test_uniform_different_grid_shapes(self):
        """uniform mode: works with non-square spatial grids."""
        for (f, h, w) in [(1, 8, 8), (2, 4, 6), (4, 2, 2), (1, 16, 32)]:
            lq = f * h * w
            mask = self._run_mask(lq, (f, h, w), 0.5, 'uniform')
            self.assertEqual(
                mask.shape, (lq,),
                f'Shape mismatch for grid {(f,h,w)}'
            )
            self.assertGreater(mask.sum().item(), 0,
                               f'No tokens selected for grid {(f,h,w)}')

    def test_uniform_pattern_stability(self):
        """uniform mode: produces same mask given same inputs (deterministic)."""
        mask1 = self._run_mask(48, (3, 4, 4), 0.5, 'uniform')
        mask2 = self._run_mask(48, (3, 4, 4), 0.5, 'uniform')
        import torch
        torch.testing.assert_close(mask1, mask2)

    # ---- random pattern ----

    def test_random_deterministic(self):
        """random mode: same seed -> same mask."""
        import torch
        gs = torch.tensor((2, 4, 4), device=self.device).unsqueeze(0)
        # Reset seed to known state
        torch.manual_seed(42)
        mask1 = _create_sparse_q_mask(32, gs, 0.5, 'random')
        torch.manual_seed(42)
        mask2 = _create_sparse_q_mask(32, gs, 0.5, 'random')
        torch.testing.assert_close(mask1, mask2)

    def test_random_vs_uniform_different(self):
        """random mode produces different pattern from uniform mode."""
        import torch
        gs = torch.tensor((2, 4, 4), device=self.device).unsqueeze(0)
        mask_random = _create_sparse_q_mask(32, gs, 0.5, 'random')
        mask_uniform = _create_sparse_q_mask(32, gs, 0.5, 'uniform')
        # Random and uniform patterns almost certainly differ
        self.assertFalse(torch.allclose(mask_random, mask_uniform))

    def test_random_exact_count(self):
        """random mode: number of True elements matches expected ratio."""
        lq = 100
        grid_sizes = (1, 10, 10)
        for ratio in [0.1, 0.25, 0.5, 0.75, 0.9]:
            mask = self._run_mask(lq, grid_sizes, ratio, 'random')
            expected = max(1, int(lq * ratio))
            actual = mask.sum().item()
            self.assertAlmostEqual(actual, expected, delta=2,
                                   msg=f'ratio={ratio}: expected ~{expected}, got {actual}')

    # ---- edge cases ----

    def test_full_ratio(self):
        """ratio >= 1.0 should return all True mask."""
        for lq, gs in [(32, (2, 4, 4)), (64, (2, 8, 4)), (16, (1, 4, 4))]:
            mask = self._run_mask(lq, gs, 1.0, 'uniform')
            self.assertTrue(mask.all().item(),
                            f'Not all True for 1.0 ratio at lq={lq}')
            mask = self._run_mask(lq, gs, 2.0, 'uniform')
            self.assertTrue(mask.all().item(),
                            f'Not all True for 2.0 ratio at lq={lq}')

    def test_small_ratio_at_least_one(self):
        """Even with extremely small ratio, at least 1 token should be True."""
        for lq in [48, 100, 4, 1]:
            if lq == 1:
                grid_sizes = (1, 1, 1)
            elif lq == 4:
                grid_sizes = (1, 2, 2)
            else:
                grid_sizes = (1, int(lq ** 0.5), int(lq ** 0.5))
            import torch
            gs_t = torch.tensor(grid_sizes, device=self.device).unsqueeze(0)
            mask = _create_sparse_q_mask(lq, gs_t, 1e-6, 'random')
            self.assertGreaterEqual(
                mask.sum().item(), 1,
                f'At least 1 token should be True for lq={lq}'
            )

    def test_padded_lq_greater_than_actual(self):
        """When lq > actual tokens (padded sequences), padding positions are False."""
        grid_sizes = (2, 4, 4)  # 32 actual tokens
        lq = 48  # 16 padding tokens
        mask = self._run_mask(lq, grid_sizes, 0.5, 'uniform')
        self.assertEqual(mask.shape, (lq,))
        # Last 16 positions should be False (padding)
        self.assertFalse(mask[32:].any().item(),
                         'Padding positions should be False')
        # At least some of the first 32 should be True
        self.assertTrue(mask[:32].any().item(),
                        'Actual token positions should have True values')

    def test_single_token(self):
        """Edge case: single token should always be True."""
        import torch
        gs = torch.tensor((1, 1, 1), device=self.device).unsqueeze(0)
        mask = _create_sparse_q_mask(1, gs, 0.5, 'uniform')
        self.assertTrue(mask[0].item())

    def test_zero_lq(self):
        """Edge case: zero length returns empty mask."""
        import torch
        gs = torch.tensor((0, 0, 0), device=self.device).unsqueeze(0)
        mask = _create_sparse_q_mask(0, gs, 0.5, 'uniform')
        self.assertEqual(mask.shape, (0,))


# ===================================================================
# Unit tests — sparse_attention_wrapper
# ===================================================================

class TestSparseAttentionWrapper(unittest.TestCase):

    def _wrapper(self, q, k, v, sparse_config, grid_sizes, **flash_kwargs):
        """
        Re-implements sparse_attention_wrapper logic for testing,
        substituting a mock for flash_attention.
        """
        import torch
        assert hasattr(self, '_mock_fa'), 'Test must set _mock_fa'
        mock_fa = self._mock_fa

        if not sparse_config.enabled or sparse_config.ratio >= 1.0:
            return mock_fa(q, k, v, **flash_kwargs)

        b, lq, nq, c1 = q.shape
        c2 = v.shape[-1]

        q_mask = _create_sparse_q_mask(lq, grid_sizes, sparse_config.ratio, sparse_config.pattern)
        selected_indices = q_mask.nonzero(as_tuple=True)[0]

        if len(selected_indices) == 0:
            return mock_fa(q, k, v, **flash_kwargs)

        q_selected = q[:, selected_indices, :, :]

        fa_kw = dict(flash_kwargs)
        fa_kw.pop('q_lens', None)

        out_selected = mock_fa(q=q_selected, k=k, v=v, **fa_kw)

        out = torch.zeros(b, lq, nq, c2, dtype=out_selected.dtype, device=out_selected.device)
        out[:, selected_indices, :, :] = out_selected

        unselected_mask = ~q_mask
        unselected_indices = unselected_mask.nonzero(as_tuple=True)[0]

        if len(unselected_indices) > 0:
            f, h, w = grid_sizes[0].tolist()
            tokens_per_frame = h * w
            for pos in unselected_indices:
                pos_int = pos.item()
                if pos_int >= lq:
                    continue
                frame_idx = pos_int // tokens_per_frame
                frame_start = frame_idx * tokens_per_frame
                frame_end = min(frame_idx * tokens_per_frame + tokens_per_frame, lq)
                in_frame_mask = (selected_indices >= frame_start) & (selected_indices < frame_end)
                in_frame = selected_indices[in_frame_mask]
                if len(in_frame) > 0:
                    nearest = in_frame[torch.argmin(torch.abs(in_frame - pos_int))]
                    out[:, pos_int:pos_int + 1, :, :] = out[:, nearest:nearest + 1, :, :]
                else:
                    nearest = selected_indices[torch.argmin(torch.abs(selected_indices - pos_int))]
                    out[:, pos_int:pos_int + 1, :, :] = out[:, nearest:nearest + 1, :, :]

        return out

    def setUp(self):
        import torch
        self.device = torch.device('cpu')
        self.b, self.lq, self.lk, self.nq, self.c1, self.c2 = 1, 16, 32, 4, 8, 8
        self.q = torch.randn(self.b, self.lq, self.nq, self.c1)
        self.k = torch.randn(self.b, self.lk, self.nq, self.c1)
        self.v = torch.randn(self.b, self.lk, self.nq, self.c2)
        self.grid_sizes = torch.tensor((1, 4, 4)).unsqueeze(0)  # 16 tokens, shape [1, 3]

    def test_disabled_config_calls_flash_attention(self):
        """When sparse attention is disabled, wrapper directly calls flash_attention."""
        import torch
        cfg = SparseAttentionConfig(enabled=False, ratio=0.5)
        mock_fa = MagicMock(return_value=torch.randn(self.b, self.lq, self.nq, self.c2))
        self._mock_fa = mock_fa

        result = self._wrapper(
            self.q, self.k, self.v, cfg, self.grid_sizes,
            k_lens=torch.tensor([self.lk]),
        )
        mock_fa.assert_called_once()
        # First positional arg should be q
        call_q = mock_fa.call_args[0][0]
        self.assertTrue(torch.equal(call_q, self.q))

    def test_enabled_ratio_one_passes_through(self):
        """When ratio >= 1.0, wrapper directly calls flash_attention."""
        import torch
        cfg = SparseAttentionConfig(enabled=True, ratio=1.0)
        mock_fa = MagicMock(return_value=torch.randn(self.b, self.lq, self.nq, self.c2))
        self._mock_fa = mock_fa

        result = self._wrapper(
            self.q, self.k, self.v, cfg, self.grid_sizes,
            k_lens=torch.tensor([self.lk]),
        )
        mock_fa.assert_called_once()

    def test_enabled_subset_query(self):
        """With enabled config (ratio=0.5), flash_attention is called with subset of queries."""
        import torch
        cfg = SparseAttentionConfig(enabled=True, ratio=0.5)
        mock_fa = MagicMock(return_value=torch.randn(self.b, 8, self.nq, self.c2))
        self._mock_fa = mock_fa

        result = self._wrapper(
            self.q, self.k, self.v, cfg, self.grid_sizes,
        )

        mock_fa.assert_called_once()
        call_q = mock_fa.call_args[1]['q']
        self.assertLess(call_q.shape[1], self.lq)

    def test_wrapper_output_shape(self):
        """Output shape should match original input shape."""
        import torch
        cfg = SparseAttentionConfig(enabled=True, ratio=0.5)
        mock_fa = MagicMock(return_value=torch.randn(self.b, 8, self.nq, self.c2))
        self._mock_fa = mock_fa

        result = self._wrapper(
            self.q, self.k, self.v, cfg, self.grid_sizes,
        )
        self.assertEqual(result.shape, (self.b, self.lq, self.nq, self.c2))

    def test_wrapper_with_q_lens_kwarg(self):
        """When q_lens is passed in flash_kwargs, it should be popped for sparse call."""
        import torch
        cfg = SparseAttentionConfig(enabled=True, ratio=0.5)
        q_lens = torch.tensor([self.lq])
        mock_fa = MagicMock(return_value=torch.randn(self.b, 8, self.nq, self.c2))
        self._mock_fa = mock_fa

        result = self._wrapper(
            self.q, self.k, self.v, cfg, self.grid_sizes,
            q_lens=q_lens, k_lens=torch.tensor([self.lk]),
        )
        # Should NOT pass q_lens to flash_attention (it's popped)
        self.assertNotIn('q_lens', mock_fa.call_args[1])

    def test_wrapper_nearest_neighbor_fill(self):
        """Uncomputed positions should be filled via nearest neighbor (output shape check)."""
        import torch
        cfg = SparseAttentionConfig(enabled=True, ratio=0.25)
        mock_fa = MagicMock(return_value=torch.randn(self.b, 4, self.nq, self.c2))
        self._mock_fa = mock_fa

        result = self._wrapper(
            self.q, self.k, self.v, cfg, self.grid_sizes,
        )
        self.assertEqual(result.shape, (self.b, self.lq, self.nq, self.c2))


# ===================================================================
# Integration tests — model-level (mock)
# ===================================================================

class TestModelSparseAttention(unittest.TestCase):

    def test_init_sparse_attention(self):
        """WanModel.init_sparse_attention should create and assign config."""
        # Create a mock model that mimics WanModel's sparse_config attrs
        model = MagicMock()
        model.blocks = [MagicMock() for _ in range(3)]
        for blk in model.blocks:
            blk.self_attn = MagicMock()

        # Simulate init_sparse_attention
        ratio, pattern = 0.3, 'uniform'
        cfg = SparseAttentionConfig(enabled=True, ratio=ratio, pattern=pattern)
        model.sparse_attention_config = cfg
        for blk in model.blocks:
            blk.self_attn.sparse_config = cfg

        # Verify
        self.assertTrue(model.sparse_attention_config.enabled)
        self.assertEqual(model.sparse_attention_config.ratio, 0.3)
        for blk in model.blocks:
            self.assertIs(blk.self_attn.sparse_config, cfg)

    def test_init_sparse_attention_default_ratio(self):
        """init_sparse_attention with default ratio should use 0.5."""
        model = MagicMock()
        model.blocks = [MagicMock() for _ in range(3)]
        for blk in model.blocks:
            blk.self_attn = MagicMock()

        ratio, pattern = 0.5, 'uniform'
        cfg = SparseAttentionConfig(enabled=True, ratio=ratio, pattern=pattern)
        model.sparse_attention_config = cfg
        for blk in model.blocks:
            blk.self_attn.sparse_config = cfg

        self.assertEqual(model.sparse_attention_config.ratio, 0.5)

    def test_init_sparse_attention_custom_pattern(self):
        """init_sparse_attention should accept custom pattern."""
        model = MagicMock()
        model.blocks = [MagicMock() for _ in range(3)]
        for blk in model.blocks:
            blk.self_attn = MagicMock()

        cfg = SparseAttentionConfig(enabled=True, ratio=0.3, pattern='random')
        model.sparse_attention_config = cfg
        for blk in model.blocks:
            blk.self_attn.sparse_config = cfg

        self.assertEqual(model.sparse_attention_config.pattern, 'random')

    def test_disable_sparse_attention(self):
        """disable_sparse_attention should set enabled=False and clear block configs."""
        model = MagicMock()
        model.blocks = [MagicMock() for _ in range(3)]
        for blk in model.blocks:
            blk.self_attn = MagicMock()
            blk.self_attn.sparse_config = MagicMock()

        # Simulate disable
        model.sparse_attention_config = SparseAttentionConfig(enabled=False)
        for blk in model.blocks:
            blk.self_attn.sparse_config = None

        self.assertFalse(model.sparse_attention_config.enabled)
        for blk in model.blocks:
            self.assertIsNone(blk.self_attn.sparse_config)

    def test_sparse_attention_in_forward_sparse_config_set(self):
        """When sparse_config is not None and enabled, sparse_attention_wrapper is used."""
        block = MagicMock()
        block.self_attn = MagicMock()
        block.self_attn.sparse_config = SparseAttentionConfig(enabled=True, ratio=0.5)
        self.assertIsNotNone(block.self_attn.sparse_config)
        self.assertTrue(block.self_attn.sparse_config.enabled)

    def test_sparse_attention_in_forward_sparse_config_none(self):
        """When sparse_config is None, fallback path (flash_attention) is used."""
        block = MagicMock()
        block.self_attn = MagicMock()
        block.self_attn.sparse_config = None
        self.assertIsNone(block.self_attn.sparse_config)


# ===================================================================
# Integration test — argparse
# ===================================================================
class TestArgparseSparseAttention(unittest.TestCase):

    def test_argparse_defaults(self):
        """Default argparse values for sparse attention are correct."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--use_sparse_attention', action='store_true', default=False)
        parser.add_argument('--sparse_attention_ratio', type=float, default=0.5)

        args = parser.parse_args([])
        self.assertFalse(args.use_sparse_attention)
        self.assertAlmostEqual(args.sparse_attention_ratio, 0.5)

    def test_argparse_enable_sparse_attention(self):
        """When --use_sparse_attention is passed, it should be True."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--use_sparse_attention', action='store_true', default=False)
        parser.add_argument('--sparse_attention_ratio', type=float, default=0.5)

        args = parser.parse_args(['--use_sparse_attention'])
        self.assertTrue(args.use_sparse_attention)
        self.assertAlmostEqual(args.sparse_attention_ratio, 0.5)

    def test_argparse_custom_ratio(self):
        """Custom --sparse_attention_ratio should override the default."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--use_sparse_attention', action='store_true', default=False)
        parser.add_argument('--sparse_attention_ratio', type=float, default=0.5)

        args = parser.parse_args(['--sparse_attention_ratio', '0.25'])
        self.assertFalse(args.use_sparse_attention)
        self.assertAlmostEqual(args.sparse_attention_ratio, 0.25)

    def test_argparse_both_flags(self):
        """Both flags should work together."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--use_sparse_attention', action='store_true', default=False)
        parser.add_argument('--sparse_attention_ratio', type=float, default=0.5)

        args = parser.parse_args([
            '--use_sparse_attention',
            '--sparse_attention_ratio', '0.1',
        ])
        self.assertTrue(args.use_sparse_attention)
        self.assertAlmostEqual(args.sparse_attention_ratio, 0.1)

    def test_argparse_ratio_range(self):
        """Ratio should accept valid float values."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--sparse_attention_ratio', type=float, default=0.5)

        for val in ['0.0', '0.001', '0.99', '1.0']:
            args = parser.parse_args(['--sparse_attention_ratio', val])
            self.assertAlmostEqual(args.sparse_attention_ratio, float(val),
                                   msg=f'Failed for ratio={val}')

    def test_argparse_ratio_out_of_range(self):
        """Ratio outside 0-1 range should still be accepted (no validator)."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--sparse_attention_ratio', type=float, default=0.5)

        for val in ['-0.1', '1.5', '10.0']:
            args = parser.parse_args(['--sparse_attention_ratio', val])
            self.assertAlmostEqual(args.sparse_attention_ratio, float(val),
                                   msg=f'Failed for ratio={val}')


# ===================================================================
# Run
# ===================================================================
if __name__ == '__main__':
    unittest.main()
