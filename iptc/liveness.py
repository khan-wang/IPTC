from . import base as gate
import csv
from pathlib import Path
import shutil
import sys
import types
import torch
import torch.nn.functional as F

class Intervention(gate.Intervention):

    def __init__(self, model, method, record, seed, diagnose=False):
        self.label = method
        base = 'put' if method.endswith('_put') or method == 'put' else 'sbvc_original' if method.endswith('_sbvc') or method == 'sbvc_original' else 'dirichlet_original'
        super().__init__(model, base, record, seed, diagnose=False)
        self.audit = []
        self.calls = 0
        self.debug = method.startswith('check_')
        self.cached_state = None
        self.cached_known = None

    def install(self):
        if self.method == 'put':
            self.model.safe_tome_enabled = False
            self.model.safe_tome_r = 0
            self.model.boundary_split_enabled = False
            self.model.ga_spg_guidance_enabled = False
            self.model.ga_spg_lite_mode = False
        super().install()
        if self.label in {'put', 'dirichlet_original', 'sbvc_original'} or self.label.startswith('route_only'):
            return
        self.model._live_output_forward = types.MethodType(lambda owner, *args: self.head(*args), self.model)
        if 'race' in self.label:
            self.model._live_sample_forward = types.MethodType(lambda owner, *args: self.sample_live(*args), self.model)
        if 'tail' in self.label:
            self.model._live_tail_forward = types.MethodType(lambda owner, *args: self.tail(*args), self.model)

    def state(self, emb, split, guidance):
        if self.cached_state is None:
            if '_r256_' in self.label:
                self.model.safe_tome_r = 256
            elif '_align_' in self.label:
                available = int(split['safe_candidate_token'].sum()) // 2
                self.model.safe_tome_r = min(256, available) // 128 * 128
        if 'cache' in self.label and self.cached_state is not None:
            self.step += 1
            state = dict(self.cached_state)
            compressed = state['compressed_embedding'].clone()
            flat = emb.reshape(emb.shape[0], -1, emb.shape[-1])
            for i, plan in enumerate(state['plans']):
                if plan['actual_r']:
                    if self.debug:
                        assert torch.equal(flat[i, plan['eligible_idx']], self.cached_known[i])
                    compressed[i, 0, :plan['frozen_len']] = flat[i, plan['frozen_idx']]
                else:
                    compressed[i, 0, :plan['compressed_len']] = flat[i]
            state['compressed_embedding'] = compressed
            if self.debug:
                fresh = self.original_state(emb, split, guidance)
                for old_plan, new_plan in zip(state['plans'], fresh['plans']):
                    assert torch.equal(old_plan['frozen_idx'], new_plan['frozen_idx'])
                    if old_plan['actual_r']:
                        ids = torch.arange(old_plan['compressed_len'] - old_plan['frozen_len'], device=emb.device, dtype=emb.dtype).view(1, -1, 1)
                        assert torch.equal(old_plan['eligible_unmerge'](ids), new_plan['eligible_unmerge'](ids))
                self.audit.append(dict(sample_id=self.record['sample_id'], method=self.label, step=self.step, component='route', active=compressed.shape[2], total=state['num_tokens'], max_error=float((compressed - fresh['compressed_embedding']).abs().max())))
                torch.testing.assert_close(compressed, fresh['compressed_embedding'], atol=1e-06, rtol=1e-05)
            removed = state['profile']['removed_tokens']
            self.model.safe_tome_profile = state['profile']
            self.route_rows.append(dict(sample_id=self.record['sample_id'], method=self.label, step=self.step, removed=removed, target_r=removed, ring_tokens=-1, protected_tokens=-1))
            return state
        state = super().state(emb, split, guidance)
        if 'cache' in self.label:
            self.cached_state = state
            if self.debug:
                flat = emb.reshape(emb.shape[0], -1, emb.shape[-1])
                self.cached_known = [flat[i, p['eligible_idx']].clone() for i, p in enumerate(state['plans'])]
        self.route_rows[-1]['method'] = self.label
        return state

    def head(self, emb, live, ratio, kind, temperature, need_logits):
        from image_synthesis.modeling.models.masked_image_inpainting_transformer import logits_top_k
        self.calls += 1
        shape = (*emb.shape[:-1], self.model.num_cls)
        indices = live.reshape(-1).nonzero().flatten()
        active = self.model.to_logits(emb.reshape(-1, emb.shape[-1]).index_select(0, indices))
        filtered = logits_top_k(active, filter_ratio=ratio, minimum=1, filter_type=kind)
        probability = (filtered * temperature).softmax(-1)
        logits = logits_filter = scores = None
        if 'compact' not in self.label or need_logits:
            logits = active.new_zeros(live.numel(), shape[-1])
            logits.index_copy_(0, indices, active)
            logits_filter = active.new_zeros(live.numel(), shape[-1])
            logits_filter.index_copy_(0, indices, filtered)
        else:
            scores = active.new_full((live.numel(),), -torch.inf)
            scores.index_copy_(0, indices, filtered.max(-1).values)
            scores = scores.view_as(live)
        probs = active.new_zeros(live.numel(), shape[-1])
        probs[:, 0] = 1
        probs.index_copy_(0, indices, probability)
        if self.debug:
            dense = self.model.to_logits(emb).reshape(-1, shape[-1]).index_select(0, indices)
            self.audit.append(dict(sample_id=self.record['sample_id'], method=self.label, step=self.calls, component='head', active=len(indices), total=live.numel(), max_error=float((active - dense).abs().max())))
            torch.testing.assert_close(active, dense, atol=0.0001, rtol=0.0001)
        return (logits.view(shape) if logits is not None else None, logits_filter.view(shape) if logits_filter is not None else None, probs.view(shape), scores)

    def sample_live(self, emb, known, count, ratio, kind, temperature, raster, need_logits):
        from image_synthesis.modeling.models.masked_image_inpainting_transformer import logits_top_k
        assert emb.shape[0] == 1 and (not raster)
        assert emb.dtype == torch.float32, 'RNG equivalence is validated for float32.'
        self.calls += 1
        indices = (~known).flatten().nonzero().flatten()
        active = self.model.to_logits(emb.reshape(-1, emb.shape[-1]).index_select(0, indices))
        scores = active.new_full((known.numel(),), -torch.inf)
        scores.index_copy_(0, indices, active.max(-1).values)
        if count == -1 or count >= known.numel():
            selected = indices
        else:
            selected = scores.view(1, -1).topk(count, dim=1).indices.flatten()
            assert len(selected) <= len(indices)
        selected_active = torch.searchsorted(indices, selected)
        chosen_logits = active.index_select(0, selected_active)
        filtered = logits_top_k(chosen_logits, filter_ratio=ratio, minimum=1, filter_type=kind)
        probability = (filtered * temperature).softmax(-1)
        state_before = torch.cuda.get_rng_state() if self.debug else None
        noise = active.new_empty((known.numel(), active.shape[-1])).exponential_(1)
        sampled = (probability / noise.index_select(0, selected)).argmax(-1)
        if self.debug:
            state_after = torch.cuda.get_rng_state()
            full_filtered = logits_top_k(active, filter_ratio=ratio, minimum=1, filter_type=kind)
            full_probs = active.new_zeros(known.numel(), active.shape[-1])
            full_probs[:, 0] = 1
            full_probs.index_copy_(0, indices, (full_filtered * temperature).softmax(-1))
            torch.cuda.set_rng_state(state_before)
            expected = torch.multinomial(full_probs, 1).flatten().index_select(0, selected)
            assert torch.equal(sampled, expected)
            assert torch.equal(state_after, torch.cuda.get_rng_state())
            self.audit.append(dict(sample_id=self.record['sample_id'], method=self.label, step=self.calls, component='sampling', active=len(selected), total=known.numel(), max_error=0.0))
        sample = torch.zeros(known.numel(), device=emb.device, dtype=torch.long)
        sample.index_copy_(0, selected, sampled)
        logits = None
        if need_logits:
            logits = active.new_zeros(known.numel(), active.shape[-1])
            logits.index_copy_(0, indices, active)
            logits = logits.view(*known.shape, active.shape[-1])
        return (logits, scores.view_as(known), sample.view_as(known))

    def tail(self, emb, mask, live, state):
        block = self.model.blocks[-1]
        assert not block.training
        assert block.attn.window_size is None
        assert emb.shape[0] == 1
        live_flat = live.flatten()
        assert live_flat.numel() == (state['num_tokens'] if state else emb.shape[1] * emb.shape[2])
        if state is not None:
            plan = state['plans'][0]
            if plan['actual_r']:
                compressed_live = torch.zeros(emb.shape[1] * emb.shape[2], device=emb.device, dtype=torch.bool)
                compressed_live[:plan['frozen_len']] = live_flat[plan['frozen_idx']]
                assert not bool(live_flat[plan['eligible_idx']].any())
                live_flat = compressed_live
        indices = live_flat.nonzero().flatten()
        flat = emb.reshape(1, -1, emb.shape[-1])
        normalized = block.norm1(flat)
        attn = block.attn
        bias = None
        if attn.q_bias is not None:
            bias = torch.cat((attn.q_bias, torch.zeros_like(attn.v_bias), attn.v_bias))
        qkv = F.linear(normalized, attn.qkv.weight, bias)
        q, k, v = qkv.reshape(1, -1, 3, attn.num_heads, emb.shape[-1] // attn.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
        q = q.index_select(-2, indices)
        logits = q @ k.transpose(-2, -1) * attn.scale
        if getattr(self, 'mass_bias', None) is not None:
            logits = logits + self.mass_bias
        if mask is not None:
            logits = logits.masked_fill(~mask.reshape(1, 1, 1, -1), -torch.inf)
        attention = attn.attn_drop(logits.softmax(-1)) @ v
        attention = attention.transpose(1, 2).reshape(1, len(indices), emb.shape[-1])
        active = flat.index_select(1, indices) + block.drop_path(attn.proj_drop(attn.proj(attention)))
        active = active + block.drop_path(block.mlp(block.norm2(active)))
        output = flat.clone()
        output.index_copy_(1, indices, active)
        if self.debug:
            expected, _ = block(emb, mask=mask)
            expected = expected.reshape_as(flat).index_select(1, indices)
            self.audit.append(dict(sample_id=self.record['sample_id'], method=self.label, step=self.calls + 1, component='tail', active=len(indices), total=flat.shape[1], max_error=float((active - expected).abs().max())))
            torch.testing.assert_close(active, expected, atol=0.0001, rtol=0.0001)
        return (output.reshape_as(emb), None)

    def remove(self):
        for key in ('_live_tail_forward', '_live_output_forward', '_live_sample_forward'):
            if hasattr(self.model, key):
                delattr(self.model, key)
        if self.audit:
            path = Path(sys.argv[sys.argv.index('--output') + 1]) / 'liveness_audit.csv'
            exists = path.exists()
            with path.open('a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(self.audit[0]))
                if not exists:
                    writer.writeheader()
                writer.writerows(self.audit)
        super().remove()
        self.cached_state = self.cached_known = None
