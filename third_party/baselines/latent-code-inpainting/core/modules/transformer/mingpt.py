"""
taken from: https://github.com/karpathy/minGPT/
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging
import os
import time

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import top_k_top_p_filtering

logger = logging.getLogger(__name__)

_LATENT_SBVC_STATS = []
_LATENT_SBVC_ROUTE_CACHE = {}


def reset_latent_sbvc_stats():
    _LATENT_SBVC_STATS.clear()
    _LATENT_SBVC_ROUTE_CACHE.clear()


def get_latent_sbvc_stats():
    return list(_LATENT_SBVC_STATS)


def _sbvc_env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _sbvc_env_int(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _sbvc_layer_enabled(layer_index):
    spec = os.environ.get("LATENT_SBVC_LAYER_IDS", "").strip()
    if not spec or spec.lower() == "all":
        return True
    if layer_index < 0:
        return True
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = chunk.split("-", 1)
            try:
                if int(start) <= layer_index <= int(end):
                    return True
            except ValueError:
                continue
        else:
            try:
                if int(chunk) == layer_index:
                    return True
            except ValueError:
                continue
    return False


def _sbvc_now(device, enabled):
    if not enabled:
        return None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _sbvc_elapsed_ms(device, start, enabled):
    if not enabled or start is None:
        return 0.0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (time.perf_counter() - start) * 1000.0


def _sbvc_gather_tokens(x, index):
    # x: [B, H, T, D], index: [B, N]
    return x.gather(2, index[:, None, :, None].expand(-1, x.shape[1], -1, x.shape[3]))


def _sbvc_merge_and_compact_one(tensor, src_pos, dst_pos, keep_idx):
    B, H, T, D = tensor.shape
    r = src_pos.shape[1]
    counts = tensor.new_ones((B, T))
    counts.scatter_add_(1, dst_pos, tensor.new_ones((B, r)))
    merged = tensor.clone()
    src_values = _sbvc_gather_tokens(tensor, src_pos)
    merged.scatter_add_(2, dst_pos[:, None, :, None].expand(B, H, r, D), src_values)
    merged = merged / counts[:, None, :, None].clamp_min(1.0)
    return _sbvc_gather_tokens(merged, keep_idx)


def _sbvc_merge_and_compact(q, k, v, src_pos, dst_pos, keep_idx):
    return (
        _sbvc_merge_and_compact_one(q, src_pos, dst_pos, keep_idx),
        _sbvc_merge_and_compact_one(k, src_pos, dst_pos, keep_idx),
        _sbvc_merge_and_compact_one(v, src_pos, dst_pos, keep_idx),
    )


def _sbvc_restore_tokens(y_compact, src_pos, dst_pos, keep_idx, original_tokens):
    B, H, _Tc, D = y_compact.shape
    r = src_pos.shape[1]
    restored = y_compact.new_zeros((B, H, original_tokens, D))
    restored.scatter_(2, keep_idx[:, None, :, None].expand(-1, H, -1, D), y_compact)
    dst_values = _sbvc_gather_tokens(restored, dst_pos)
    restored.scatter_(2, src_pos[:, None, :, None].expand(B, H, r, D), dst_values)
    return restored


def _sbvc_token_positions(token_indices, grid_size):
    pos = (token_indices - 1).clamp_min(0)
    return torch.stack((pos // grid_size, pos % grid_size), dim=-1).float()


def _sbvc_bipartite_route(features, key_mask, target_r, route_mode="safe_similarity"):
    # Default route is unchanged: only visible latent tokens may merge, and token 0 stays protected.
    B, T, _C = features.shape
    device = features.device
    if target_r <= 0 or T <= 3:
        return None
    route_mode = str(route_mode or "safe_similarity").strip().lower()
    if route_mode not in {"safe_similarity", "safe_distance", "global_similarity"}:
        route_mode = "safe_similarity"
    if key_mask is None and route_mode != "global_similarity":
        return None

    src_idx = torch.arange(1, T, 2, device=device)
    dst_idx = torch.arange(2, T, 2, device=device)
    if src_idx.numel() == 0 or dst_idx.numel() == 0:
        return None

    if route_mode == "global_similarity":
        safe = torch.ones((B, T), dtype=torch.bool, device=device)
    else:
        safe = key_mask.to(dtype=torch.bool).clone()
    safe[:, 0] = False
    src_safe = safe.index_select(1, src_idx)
    dst_safe = safe.index_select(1, dst_idx)

    if route_mode == "safe_distance":
        grid_size = int(round(math.sqrt(max(T - 1, 1))))
        if grid_size * grid_size != T - 1:
            grid_size = max(T - 1, 1)
        src_pos_2d = _sbvc_token_positions(src_idx, grid_size)
        dst_pos_2d = _sbvc_token_positions(dst_idx, grid_size)
        dist2 = (src_pos_2d[:, None, :] - dst_pos_2d[None, :, :]).pow(2).sum(dim=-1)
        scores = -dist2[None, :, :].expand(B, -1, -1)
    else:
        normed = F.normalize(features.float(), dim=-1)
        src_feat = normed.index_select(1, src_idx)
        dst_feat = normed.index_select(1, dst_idx)
        scores = src_feat @ dst_feat.transpose(-2, -1)
    valid = src_safe[:, :, None] & dst_safe[:, None, :]
    scores = scores.masked_fill(~valid, -1.0e9)

    best_scores, best_dst_rel = scores.max(dim=-1)
    valid_counts = (best_scores > -1.0e8).sum(dim=1)
    actual_r = min(int(target_r), int(valid_counts.min().item()))
    if actual_r <= 0:
        return None

    _top_scores, src_rel = best_scores.topk(actual_r, dim=1)
    dst_rel = best_dst_rel.gather(1, src_rel)
    src_pos = src_idx[src_rel]
    dst_pos = dst_idx[dst_rel]

    keep_mask = torch.ones((B, T), dtype=torch.bool, device=device)
    keep_mask.scatter_(1, src_pos, False)
    all_idx = torch.arange(T, device=device).expand(B, T)
    keep_idx = all_idx[keep_mask].view(B, T - actual_r)
    return src_pos, dst_pos, keep_idx, safe


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)


class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        mask = torch.tril(torch.ones(config.block_size,
                                     config.block_size))
        if hasattr(config, "n_unmasked"):
            mask[:config.n_unmasked, :config.n_unmasked] = 1
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self._sbvc_layer_index = -1

    def _attention(self, q, k, v, autoregressive=True, mask=None, query_positions=None, key_positions=None):
        q_tokens = q.size(-2)
        k_tokens = k.size(-2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if autoregressive:
            if q_tokens == k_tokens:
                causal_mask = self.mask[:, :, :q_tokens, :k_tokens]
            else:
                causal_mask = torch.ones(q_tokens, k_tokens, device=q.device, dtype=torch.bool)
                causal_mask = torch.tril(causal_mask, diagonal=k_tokens - q_tokens)[None, None]
            att = att.masked_fill(causal_mask == 0, float('-inf'))

        if not autoregressive and mask is not None:
            mask_diag = mask.unsqueeze(1).unsqueeze(1).expand(-1, -1, q_tokens, -1)
            if key_positions is None:
                eyes = torch.eye(q_tokens, k_tokens, device=mask.device, dtype=torch.bool)[None, None]
                eyes = eyes.expand(q.shape[0], -1, -1, -1)
            else:
                if query_positions is None:
                    query_positions = torch.arange(q_tokens, device=mask.device).expand(q.shape[0], q_tokens)
                eyes = query_positions[:, :, None].eq(key_positions[:, None, :])[:, None, :, :]
            mask_diag = torch.logical_or(mask_diag.to(dtype=torch.bool), eyes)
            att = att.masked_fill(mask_diag == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        return att @ v

    def forward(self, x, autoregressive=True, layer_past=None, mask=None):
        B, T, C = x.size()
        device = x.device
        profile = _sbvc_env_flag("LATENT_SBVC_PROFILE", False)
        collect_stats = profile or _sbvc_env_flag("LATENT_SBVC_COLLECT_STATS", False)
        target_r = _sbvc_env_int("LATENT_SBVC_R", 0)
        sbvc_mode = os.environ.get("LATENT_SBVC_MODE", "qkv").strip().lower()
        if sbvc_mode not in {"qkv", "kv"}:
            sbvc_mode = "qkv"
        route_mode = os.environ.get("LATENT_SBVC_ROUTE_MODE", "safe_similarity").strip().lower()
        if route_mode not in {"safe_similarity", "safe_distance", "global_similarity"}:
            route_mode = "safe_similarity"
        layer_index = getattr(self, "_sbvc_layer_index", -1)
        sbvc_enabled = (
            _sbvc_env_flag("LATENT_SBVC_ENABLE", False)
            and not self.training
            and not autoregressive
            and layer_past is None
            and mask is not None
            and target_r > 0
            and _sbvc_layer_enabled(layer_index)
        )

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        present = torch.stack((k, v))
        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        pair_ms = 0.0
        merge_ms = 0.0
        attention_ms = 0.0
        restore_ms = 0.0
        route_cache_hit = False
        compact_tokens = T
        removed_tokens = 0
        safe_tokens = 0
        protect_tokens = T

        if sbvc_enabled:
            cache_key = None
            if _sbvc_env_flag("LATENT_SBVC_ROUTE_CACHE", False):
                cache_key = (id(mask), int(target_r), int(T), int(B), str(device), route_mode)
                route = _LATENT_SBVC_ROUTE_CACHE.get(cache_key)
                route_cache_hit = route is not None
            else:
                route = None
            if route is None:
                start = _sbvc_now(device, profile)
                route = _sbvc_bipartite_route(x, mask, target_r, route_mode=route_mode)
                pair_ms = _sbvc_elapsed_ms(device, start, profile)
                if cache_key is not None and route is not None:
                    _LATENT_SBVC_ROUTE_CACHE[cache_key] = route
            if route is not None:
                src_pos, dst_pos, keep_idx, safe = route
                removed_tokens = int(src_pos.shape[1])
                safe_tokens = int(safe.sum(dim=1).float().mean().item())
                protect_tokens = T - safe_tokens
                compact_tokens = T - removed_tokens
                start = _sbvc_now(device, profile)
                if sbvc_mode == "kv":
                    q_compact = q
                    k_compact = _sbvc_merge_and_compact_one(k, src_pos, dst_pos, keep_idx)
                    v_compact = _sbvc_merge_and_compact_one(v, src_pos, dst_pos, keep_idx)
                    query_positions = None
                    key_positions = keep_idx
                else:
                    q_compact, k_compact, v_compact = _sbvc_merge_and_compact(q, k, v, src_pos, dst_pos, keep_idx)
                    query_positions = keep_idx
                    key_positions = keep_idx
                mask_compact = mask.gather(1, keep_idx)
                merge_ms = _sbvc_elapsed_ms(device, start, profile)
                start = _sbvc_now(device, profile)
                y_compact = self._attention(
                    q_compact,
                    k_compact,
                    v_compact,
                    autoregressive=False,
                    mask=mask_compact,
                    query_positions=query_positions,
                    key_positions=key_positions,
                )
                attention_ms = _sbvc_elapsed_ms(device, start, profile)
                if sbvc_mode == "kv":
                    y = y_compact
                else:
                    start = _sbvc_now(device, profile)
                    y = _sbvc_restore_tokens(y_compact, src_pos, dst_pos, keep_idx, T)
                    restore_ms = _sbvc_elapsed_ms(device, start, profile)
            else:
                sbvc_enabled = False

        if not sbvc_enabled:
            if mask is not None:
                safe_for_stats = mask.to(dtype=torch.bool).clone()
                safe_for_stats[:, 0] = False
                safe_tokens = int(safe_for_stats.sum(dim=1).float().mean().item())
                protect_tokens = T - safe_tokens
            start = _sbvc_now(device, profile)
            y = self._attention(q, k, v, autoregressive=autoregressive, mask=mask)
            attention_ms = _sbvc_elapsed_ms(device, start, profile)

        if collect_stats:
            _LATENT_SBVC_STATS.append({
                "layer_index": int(layer_index),
                "enabled": bool(sbvc_enabled),
                "target_r": int(target_r),
                "original_tokens": int(T),
                "compact_tokens": int(compact_tokens),
                "removed_tokens": int(removed_tokens),
                "safe_tokens": int(safe_tokens),
                "protect_tokens": int(protect_tokens),
                "score_elements_before": int(B * self.n_head * T * T),
                "score_elements_after": int(B * self.n_head * (T if sbvc_mode == "kv" else compact_tokens) * compact_tokens),
                "pair_ms": float(pair_ms),
                "merge_ms": float(merge_ms),
                "attention_ms": float(attention_ms),
                "restore_ms": float(restore_ms),
                "route_cache_hit": bool(route_cache_hit),
                "mode": sbvc_mode,
                "route_mode": route_mode,
            })

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, present   # TODO: check that this does not break anything


class Block(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),  # nice
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, autoregressive=True, layer_past=None, return_present=False, mask=None):
        # TODO: check that training still works
        if return_present: assert not self.training
        # layer past: tuple of length two with B, nh, T, hs
        attn, present = self.attn(self.ln1(x), autoregressive=autoregressive, layer_past=layer_past, mask=mask)

        x = x + attn
        x = x + self.mlp(self.ln2(x))
        if layer_past is not None or return_present:
            return x, present
        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """
    def __init__(self, vocab_size, block_size, n_layer=12, n_head=8, n_embd=256,
                 embd_pdrop=0., resid_pdrop=0., attn_pdrop=0., n_unmasked=0):
        super().__init__()
        config = GPTConfig(vocab_size=vocab_size, block_size=block_size,
                           embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                           n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                           n_unmasked=n_unmasked)
        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        for idx, block in enumerate(self.blocks):
            block.attn._sbvc_layer_index = idx
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.config = config
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, embeddings=None, targets=None, mask=None, autoregressive=True):
        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        if embeddings is not None: # prepend explicit embeddings
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)

        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector

        x = self.drop(token_embeddings + position_embeddings)
        for idx, block in enumerate(self.blocks):
            # if idx <= len(self.blocks) // 2:
            x = block(x, autoregressive=autoregressive, mask=mask)

        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def forward_with_past(self, idx, embeddings=None, targets=None, past=None, past_length=None):
        # inference only
        assert not self.training
        token_embeddings = self.tok_emb(idx)    # each index maps to a (learnable) vector
        if embeddings is not None:              # prepend explicit embeddings
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)

        if past is not None:
            assert past_length is not None
            past = torch.cat(past, dim=-2)   # n_layer, 2, b, nh, len_past, dim_head
            past_shape = list(past.shape)
            expected_shape = [self.config.n_layer, 2, idx.shape[0], self.config.n_head, past_length, self.config.n_embd//self.config.n_head]
            assert past_shape == expected_shape, f"{past_shape} =/= {expected_shape}"
            position_embeddings = self.pos_emb[:, past_length, :]  # each position maps to a (learnable) vector
        else:
            position_embeddings = self.pos_emb[:, :token_embeddings.shape[1], :]

        x = self.drop(token_embeddings + position_embeddings)
        presents = []  # accumulate over layers
        for i, block in enumerate(self.blocks):
            x, present = block(x, layer_past=past[i, ...] if past is not None else None, return_present=True)
            presents.append(present)

        x = self.ln_f(x)
        logits = self.head(x)
        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, torch.stack(presents)  # _, _, n_layer, 2, b, nh, 1, dim_head



class DummyGPT(nn.Module):
    # for debugging
    def __init__(self, add_value=1):
        super().__init__()
        self.add_value = add_value

    def forward(self, idx):
        return idx + self.add_value, None


class CondGPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """
    def __init__(self, vocab_size, vocab_size_cond, block_size, n_layer=12, n_head=8, n_embd=256,
                 embd_pdrop=0., resid_pdrop=0., attn_pdrop=0., n_unmasked=0):
        super().__init__()
        config = GPTConfig(vocab_size=vocab_size, block_size=block_size,
                           embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                           n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                           n_unmasked=n_unmasked)
        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.cond_emb = nn.Embedding(vocab_size_cond, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.config = config
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, cidx, embeddings=None, targets=None):
        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        cond_embeddings = self.cond_emb(cidx)

        # concat cond embeddings before token_embeddings
        token_embeddings = torch.cat([cond_embeddings, token_embeddings], dim=1)

        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def forward_with_past(self, idx, embeddings=None, targets=None, past=None, past_length=None):
        # inference only
        assert not self.training
        token_embeddings = self.tok_emb(idx)    # each index maps to a (learnable) vector
        if embeddings is not None:              # prepend explicit embeddings
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)

        if past is not None:
            assert past_length is not None
            past = torch.cat(past, dim=-2)   # n_layer, 2, b, nh, len_past, dim_head
            past_shape = list(past.shape)
            expected_shape = [self.config.n_layer, 2, idx.shape[0], self.config.n_head, past_length, self.config.n_embd//self.config.n_head]
            assert past_shape == expected_shape, f"{past_shape} =/= {expected_shape}"
            position_embeddings = self.pos_emb[:, past_length, :]  # each position maps to a (learnable) vector
        else:
            position_embeddings = self.pos_emb[:, :token_embeddings.shape[1], :]

        x = self.drop(token_embeddings + position_embeddings)
        presents = []  # accumulate over layers
        for i, block in enumerate(self.blocks):
            x, present = block(x, layer_past=past[i, ...] if past is not None else None, return_present=True)
            presents.append(present)

        x = self.ln_f(x)
        logits = self.head(x)
        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, torch.stack(presents)  # _, _, n_layer, 2, b, nh, 1, dim_head

#### sampling utils

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
        logits, _ = model(x_cond)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)

    return x


@torch.no_grad()
def sample_with_past(x, model, steps, temperature=1., sample_logits=True,
                     top_k=None, top_p=None, callback=None):
    # x is conditioning
    sample = x
    cond_len = x.shape[1]
    past = None
    for n in range(steps):
        if callback is not None:
            callback(n)
        logits, _, present = model.forward_with_past(x, past=past, past_length=(n+cond_len-1))
        if past is None:
            past = [present]
        else:
            past.append(present)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

        probs = F.softmax(logits, dim=-1)
        if not sample_logits:
            _, x = torch.topk(probs, k=1, dim=-1)
        else:
            x = torch.multinomial(probs, num_samples=1)
        # append to the sequence and continue
        sample = torch.cat((sample, x), dim=1)
    del past
    sample = sample[:, cond_len:]  # cut conditioning off
    return sample


#### clustering utils

class KMeans(nn.Module):
    def __init__(self, ncluster=512, nc=3, niter=10):
        super().__init__()
        self.ncluster = ncluster
        self.nc = nc
        self.niter = niter
        self.shape = (3,32,32)
        self.register_buffer("C", torch.zeros(self.ncluster,nc))
        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def is_initialized(self):
        return self.initialized.item() == 1

    @torch.no_grad()
    def initialize(self, x):
        N, D = x.shape
        assert D == self.nc, D
        c = x[torch.randperm(N)[:self.ncluster]] # init clusters at random
        for i in range(self.niter):
            # assign all pixels to the closest codebook element
            a = ((x[:, None, :] - c[None, :, :])**2).sum(-1).argmin(1)
            # move each codebook element to be the mean of the pixels that assigned to it
            c = torch.stack([x[a==k].mean(0) for k in range(self.ncluster)])
            # re-assign any poorly positioned codebook elements
            nanix = torch.any(torch.isnan(c), dim=1)
            ndead = nanix.sum().item()
            print('done step %d/%d, re-initialized %d dead clusters' % (i+1, self.niter, ndead))
            c[nanix] = x[torch.randperm(N)[:ndead]] # re-init dead clusters

        self.C.copy_(c)
        self.initialized.fill_(1)


    def forward(self, x, reverse=False, shape=None):
        if not reverse:
            # flatten
            bs,c,h,w = x.shape
            assert c == self.nc
            x = x.reshape(bs,c,h*w,1)
            C = self.C.permute(1,0)
            C = C.reshape(1,c,1,self.ncluster)
            a = ((x-C)**2).sum(1).argmin(-1) # bs, h*w indices
            return a
        else:
            # flatten
            bs, HW = x.shape
            """
            c = self.C.reshape( 1, self.nc,  1, self.ncluster)
            c = c[bs*[0],:,:,:]
            c = c[:,:,HW*[0],:]
            x =      x.reshape(bs,       1, HW,             1)
            x = x[:,3*[0],:,:]
            x = torch.gather(c, dim=3, index=x)
            """
            x = self.C[x]
            x = x.permute(0,2,1)
            shape = shape if shape is not None else self.shape
            x = x.reshape(bs, *shape)

            return x
