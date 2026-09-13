from __future__ import annotations
import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import types
import numpy as np
import torch
import torch.nn.functional as F

def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def stable_seed(name, seed):
    return (int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) + seed) % 2 ** 31

def project_qkv(block, x):
    attn = block.attn
    bias = None
    if attn.q_bias is not None:
        bias = torch.cat((attn.q_bias, torch.zeros_like(attn.v_bias), attn.v_bias))
    y = F.linear(block.norm1(x), attn.qkv.weight, bias)
    return y.reshape(-1, 3, attn.num_heads, x.shape[-1] // attn.num_heads).permute(1, 2, 0, 3)

def pair_output_error(block, x, pairs, probes, qkv=None):
    """Exact single-pair first-attention error for fixed, unmerged query tokens.

    Includes LN-before-QKV of the actual mean embedding and softmax denominator
    change. No token-mass compensation, matching this PUT merge implementation.
    This is not the error of simultaneous multi-pair merging or the full network.
    """
    if qkv is None:
        qkv = project_qkv(block, x)
    q, k, v = qkv
    a = (q[:, probes] @ k.transpose(-1, -2) * block.attn.scale).softmax(-1)
    out = a @ v
    logz = torch.logsumexp(q[:, probes] @ k.transpose(-1, -2) * block.attn.scale, -1)
    chunks = []
    for pair in pairs.split(128):
        i, j = pair.unbind(-1)
        _, km, vm = project_qkv(block, (x[i] + x[j]) * 0.5)
        am = torch.exp(q[:, probes] @ km.transpose(-1, -2) * block.attn.scale - logz[..., None])
        ai, aj = (a[:, :, i], a[:, :, j])
        denom = (1 - ai - aj + am).clamp_min(1e-08)
        diff = (am[..., None] * vm[:, None] - ai[..., None] * v[:, None, i] - aj[..., None] * v[:, None, j] + (ai + aj - am)[..., None] * out[:, :, None]) / denom[..., None]
        diff = diff.permute(1, 2, 0, 3).flatten(2)
        diff = F.linear(diff, block.attn.proj.weight, None)
        chunks.append(diff.square().mean((0, 2)))
    return torch.cat(chunks)

class Intervention:

    def __init__(self, model, method, record, seed, diagnose=False):
        self.model, self.method, self.record, self.seed = (model, method, record, seed)
        self.diagnose = diagnose
        self.diag_rows, self.route_rows = ([], [])
        self.step = 0
        self.original_state = model._build_safe_tome_state
        self.original_pair = model._build_experimental_pair_metric
        self.original_count = model._safe_tome_selected_count
        self.original_score = model._safe_tome_selection_score
        self.original_r = model.safe_tome_r
        self.current_x = None
        self.missing = None
        self.cache_split = None
        self.generator = torch.Generator(device='cpu').manual_seed(seed + (100003 if method == 'random2' else 0))

    def install(self):
        if self.method == 'put':
            return
        model = self.model
        model._build_safe_tome_state = types.MethodType(lambda owner, emb, split, guidance: self.state(emb, split, guidance), model)
        if self.method not in {'sbvc_original', 'dirichlet_original'}:
            model._safe_tome_selected_count = types.MethodType(lambda owner, n: int(n), model)
            model._safe_tome_selection_score = types.MethodType(lambda owner, split, guidance: torch.zeros_like(split['token_visible_ratio']), model)
            model._build_experimental_pair_metric = types.MethodType(lambda owner, **kwargs: self.pair(**kwargs), model)

    def remove(self):
        m = self.model
        m._build_safe_tome_state = self.original_state
        m._build_experimental_pair_metric = self.original_pair
        m._safe_tome_selected_count = self.original_count
        m._safe_tome_selection_score = self.original_score
        m.safe_tome_r = self.original_r
        self.current_x = None

    def make_split(self, emb, split):
        split = {k: v.clone() if torch.is_tensor(v) else v for k, v in split.items()}
        valid = split['valid_token'].flatten()
        missing = ~valid
        ring = split['boundary_token'].flatten()
        n_ring = int(ring.sum())
        self.target_r = min(224, int((valid & ~ring).sum()) // 2)
        chosen = ring.clone()
        if self.method.startswith('random'):
            idx = valid.nonzero().flatten()
            order = torch.randperm(len(idx), generator=self.generator).to(idx.device)
            chosen = torch.zeros_like(ring)
            chosen[idx[order[:n_ring]]] = True
        elif self.method == 'texture':
            x = emb.permute(0, 3, 1, 2)
            mu = F.avg_pool2d(x, 3, 1, 1, count_include_pad=False)
            var = (F.avg_pool2d(x.square(), 3, 1, 1, count_include_pad=False) - mu.square()).mean(1).flatten()
            var = var.masked_fill(~valid, -torch.inf)
            chosen = torch.zeros_like(ring)
            chosen[var.topk(n_ring).indices] = True
        elif self.method in {'missing_only', 'global'}:
            chosen = torch.zeros_like(ring)
        protected = missing | chosen
        if self.method == 'global':
            protected = torch.zeros_like(ring)
        split['protect_mask_token'] = protected.reshape_as(split['protect_mask_token'])
        split['safe_candidate_token'] = (~protected).reshape_as(split['safe_candidate_token'])
        self.missing = missing.nonzero().flatten()
        self.ring_count = n_ring
        self.protected_count = int(protected.sum())
        return split

    def state(self, emb, split, guidance):
        self.step += 1
        if self.method in {'sbvc_original', 'dirichlet_original'}:
            state = self.original_state(emb, split, guidance)
        else:
            if self.cache_split is None:
                self.cache_split = self.make_split(emb, split)
            self.current_x = emb.reshape(-1, emb.shape[-1])
            self.model.safe_tome_r = self.target_r
            state = self.original_state(emb, self.cache_split, guidance)
        p = self.model.safe_tome_profile
        removed = int(p.get('removed_tokens', -1))
        if self.method not in {'sbvc_original', 'dirichlet_original'}:
            assert removed == self.target_r, (removed, self.target_r, p)
        self.route_rows.append({'sample_id': self.record['sample_id'], 'method': self.method, 'step': self.step, 'removed': removed, 'target_r': getattr(self, 'target_r', removed), 'ring_tokens': getattr(self, 'ring_count', -1), 'protected_tokens': getattr(self, 'protected_count', -1)})
        return state

    def pair(self, **kwargs):
        result = self.original_pair(**kwargs)
        x = self.current_x
        idx = kwargs['selection_plan']['eligible_idx']
        left, right = (idx[::2], idx[1::2])
        cos = result['base_metric']
        pairs = None
        if self.method in {'dirichlet', 'query', 'attention'} or self.diagnose:
            shortlist = cos.topk(min(4, cos.shape[1]), dim=1).indices
            src = torch.arange(len(left), device=x.device)[:, None].expand_as(shortlist)
            pairs = torch.stack((left[src.flatten()], right[shortlist.flatten()]), -1)
        if self.method == 'dirichlet':
            score = torch.full_like(cos, -torch.inf)
            score.scatter_(1, shortlist, result['adjusted_metric'].gather(1, shortlist))
        elif self.method in {'query', 'attention'}:
            block = self.model.blocks[0]
            qkv = project_qkv(block, x)
            probes = self.missing[torch.linspace(0, len(self.missing) - 1, min(8, len(self.missing)), device=x.device).long()]
            score = torch.full_like(cos, -torch.inf)
            if self.method == 'query':
                cost = pair_output_error(block, x, pairs, probes, qkv)
            else:
                q, k, _ = qkv
                mass = (q[:, probes] @ k.transpose(-1, -2) * block.attn.scale).softmax(-1).mean((0, 1))
                cost = mass[pairs].sum(-1)
            score.scatter_(1, shortlist, -cost.reshape_as(shortlist))
        else:
            score = cos
        if self.diagnose and self.step in {1, 4} and (len(self.missing) >= 4):
            block = self.model.blocks[0]
            qkv = project_qkv(block, x)
            take = torch.linspace(0, len(pairs) - 1, min(64, len(pairs)), device=x.device).long()
            selected = pairs[take]
            missing_order = self.missing[torch.randperm(len(self.missing), generator=self.generator).to(x.device)]
            n = min(8, len(missing_order) // 2)
            probes, holdout = (missing_order[:n], missing_order[n:n + 32])
            train_error = pair_output_error(block, x, selected, probes, qkv)
            test_error = pair_output_error(block, x, selected, holdout, qkv)
            q, k, _ = qkv
            mass = (q[:, probes] @ k.transpose(-1, -2) * block.attn.scale).softmax(-1).mean((0, 1))
            ii, jj = (src.flatten()[take], shortlist.flatten()[take])
            vals = torch.stack((1 - cos[ii, jj], result['pair_feature_l2_raw'][ii, jj], -result['adjusted_metric'][ii, jj], mass[selected].sum(-1), train_error, test_error), 1).cpu().numpy()
            for pair, row in zip(selected.cpu().tolist(), vals.tolist()):
                self.diag_rows.append(dict(sample_id=self.record['sample_id'], step=self.step, src=pair[0], dst=pair[1], cosine_distance=row[0], feature_l2=row[1], dirichlet=row[2], attention_mass=row[3], query_proxy=row[4], heldout_error=row[5]))
        result['adjusted_metric'] = score
        return result
