from . import liveness as live
import csv
import json
from pathlib import Path
import shutil
import sys
import types
import torch
import torch.nn.functional as F

class Intervention(live.Intervention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_originals = []
        self.block_originals = []

    def install(self):
        super().install()
        if 'sdpa' in self.label:
            for block in self.model.blocks:
                self.attention_originals.append((block.attn, block.attn.forward))
                block.attn.forward = types.MethodType(lambda owner, x, mask=None: self.sdpa(owner, x, mask), block.attn)
        if 'bf16' in self.label:
            for block in self.model.blocks[:-1]:
                original = block.forward
                self.block_originals.append((block, original))
                block.forward = types.MethodType(lambda owner, x, mask=None, original=original: self.mixed_block(x, mask, original), block)

    @staticmethod
    def mixed_block(x, mask, original):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            return original(x, mask=mask)

    @staticmethod
    def sdpa(attn, x, mask):
        assert attn.window_size is None
        bias = None if attn.q_bias is None else torch.cat((attn.q_bias, torch.zeros_like(attn.v_bias), attn.v_bias))
        qkv = F.linear(x, attn.qkv.weight, bias)
        q, k, v = qkv.reshape(x.shape[0], -1, 3, attn.num_heads, x.shape[-1] // attn.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
        mask = None if mask is None else mask.reshape(q.shape[0], 1, 1, k.shape[-2])
        result = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, scale=attn.scale)
        result = result.transpose(1, 2).reshape_as(x)
        return (attn.proj_drop(attn.proj(result)), None)

    def tail(self, emb, mask, active, state):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled='bf16' in self.label):
            return self.tail_impl(emb, mask, active, state)

    def tail_impl(self, emb, mask, active, state):
        if 'sdpa' not in self.label:
            return super().tail(emb, mask, active, state)
        block = self.model.blocks[-1]
        live_mask = active.flatten()
        if state is not None and state['plans'][0]['actual_r']:
            plan = state['plans'][0]
            compressed = torch.zeros(emb.shape[1] * emb.shape[2], dtype=torch.bool, device=emb.device)
            compressed[:plan['frozen_len']] = live_mask[plan['frozen_idx']]
            live_mask = compressed
        indices = live_mask.nonzero().flatten()
        flat = emb.reshape(1, -1, emb.shape[-1])
        attn = block.attn
        bias = None if attn.q_bias is None else torch.cat((attn.q_bias, torch.zeros_like(attn.v_bias), attn.v_bias))
        qkv = F.linear(block.norm1(flat), attn.qkv.weight, bias)
        q, k, v = qkv.reshape(1, -1, 3, attn.num_heads, emb.shape[-1] // attn.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
        q = q.index_select(-2, indices)
        mask = None if mask is None else mask.reshape(1, 1, 1, -1)
        update = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, scale=attn.scale)
        update = update.transpose(1, 2).reshape(1, len(indices), emb.shape[-1])
        selected = flat.index_select(1, indices) + block.drop_path(attn.proj_drop(attn.proj(update)))
        selected = selected + block.drop_path(block.mlp(block.norm2(selected)))
        result = flat.clone()
        result.index_copy_(1, indices, selected)
        return (result.reshape_as(emb), None)

    def remove(self):
        for module, original in self.attention_originals:
            module.forward = original
        for module, original in self.block_originals:
            module.forward = original
        super().remove()
