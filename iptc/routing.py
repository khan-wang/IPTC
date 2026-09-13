from . import execution as strong
import importlib.util
from pathlib import Path
import shutil
import sys
import types
import torch

class Intervention(strong.Intervention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.saved_guidance = self.model._ga_spg_guidance
        self.lean_checks = []

    def state(self, emb, split, guidance):
        state = super().state(emb, split, guidance)
        if '_nullmask_' in self.label:
            assert len(state['plans']) == 1 and emb.shape[0] == 1
            assert state['plans'][0]['compressed_len'] == state['compressed_embedding'].shape[2]
            state = dict(state)
            state['compressed_mask'] = None
        return state

    def install(self):
        super().install()
        if '_lean_' in self.label:
            assert self.model.safe_tome_score_mode == 'dirichlet'
            assert not self.model.safe_tome_debug_enabled and (not self.model.ga_spg_audit_enabled)
            self.model._ga_spg_guidance = types.MethodType(lambda owner, emb, split: self.lean_guidance(emb, split), self.model)
            self.model._build_experimental_pair_metric = types.MethodType(lambda owner, **kwargs: self.lean_pair(**kwargs), self.model)
        if '_cosine_' in self.label or '_feature_' in self.label:
            self.model._build_experimental_pair_metric = types.MethodType(lambda owner, **kwargs: self.score_control(**kwargs), self.model)

    def score_control(self, **kwargs):
        result = self.original_pair(**kwargs)
        cosine = result['base_metric']
        if '_cosine_' in self.label:
            result['adjusted_metric'] = cosine
        else:
            result['adjusted_metric'] = -((1.0 - cosine).clamp(min=0) * result['pair_feature_l2_raw'].square())
        return result

    def lean_guidance(self, emb, split):
        safe = split['safe_candidate_token'].bool()
        protect = split['protect_mask_token'].bool()
        b, h, w, _ = emb.shape
        distance, score = self.model.similarity_scorer._distance_rule_score(protect, safe)
        yy, xx = torch.meshgrid(torch.arange(h, device=emb.device, dtype=emb.dtype), torch.arange(w, device=emb.device, dtype=emb.dtype), indexing='ij')
        coords = torch.stack([yy, xx], -1).unsqueeze(0).expand(b, -1, -1, -1).contiguous()
        result = dict(mode='dirichlet', distance_raw=distance, distance_score=score, distance_raw_flat=distance.view(b, -1), token_coords=coords, token_coords_flat=coords.view(b, -1, 2), local_variance_risk=torch.zeros_like(distance))
        if '_audit_' in self.label:
            reference = self.saved_guidance(emb, split)
            for key in ('distance_raw', 'distance_score', 'token_coords_flat'):
                assert torch.equal(result[key], reference[key]), key
            self.lean_checks.append({'component': 'guidance_live_fields', 'exact': True})
        return result

    def lean_pair(self, **kwargs):
        model = self.model
        seq = kwargs['eligible_seq']
        runtime = kwargs['runtime_plan']
        cosine = model._pairwise_cosine(seq[::2], seq[1::2])
        l2 = model._pairwise_l2(seq[::2], seq[1::2])
        mu = max(float(model.experimental_dirichlet_mu), 0.0)
        delta = (1.0 - cosine).clamp(min=0.0) * l2.square()
        delta = delta + mu * (1.0 / (runtime['src_distance'].unsqueeze(1) + 1e-06) + 1.0 / (runtime['dst_distance'].unsqueeze(0) + 1e-06))
        result = dict(adjusted_metric=-delta, feasible_pair_count=int(cosine.shape[-2]))
        if '_audit_' in self.label:
            original = self.original_pair(**kwargs)
            assert torch.equal(result['adjusted_metric'], original['adjusted_metric'])
            self.lean_checks.append({'component': 'pair_score', 'exact': True})
        return result

    def remove(self):
        self.model._ga_spg_guidance = self.saved_guidance
        if self.lean_checks:
            import json
            output = Path(sys.argv[sys.argv.index('--output') + 1])
            with (output / 'lean_checks.jsonl').open('a') as stream:
                for check in self.lean_checks:
                    stream.write(json.dumps(dict(sample_id=self.record['sample_id'], method=self.label, **check)) + '\n')
        super().remove()
