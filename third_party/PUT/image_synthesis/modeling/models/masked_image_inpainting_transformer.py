"""
This transformer model is for PUT in to be published in Journal
"""
import torch
import math
import time
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
import os
import numpy as np
from PIL import Image, ImageDraw
from timm.models.layers import DropPath, trunc_normal_
from timm.models.layers import trunc_normal_ as __call_trunc_normal_


from image_synthesis.utils.misc import instantiate_from_config
from image_synthesis.modeling.utils.misc import get_token_type
from image_synthesis.distributed.distributed import get_local_rank
from image_synthesis.modeling.utils.misc import logits_top_k, pixel_unshuffle, pixel_shuffle
from image_synthesis.modeling.modules.losses.poly_loss import PolyLoss
from image_synthesis.modeling.modules.losses.label_smoothing_loss import LabelSmoothingLoss


def _tome_do_nothing(x, mode=None):
    return x


def _tome_bipartite_soft_matching(metric, r, return_metadata=False):
    """
    Minimal global ToMe matching adapted from facebookresearch/ToMe.

    Input size is [batch, tokens, channels]. The returned merge removes r
    tokens at most, with a maximum reduction of 50% of tokens.
    """
    t = metric.shape[1]
    r = min(int(r), t // 2)
    if r <= 0:
        if return_metadata:
            return _tome_do_nothing, _tome_do_nothing, 0, None
        return _tome_do_nothing, _tome_do_nothing, 0

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]
        src_idx = edge_idx[..., :r, :]
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

    def merge(x, mode="mean"):
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
        return torch.cat([unm, dst], dim=1)

    def unmerge(x):
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))
        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)
        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)
        return out

    if return_metadata:
        metadata = {
            "unm_idx": unm_idx,
            "src_idx": src_idx,
            "dst_idx": dst_idx,
            "num_tokens": int(metric.shape[1]),
        }
        return merge, unmerge, r, metadata
    return merge, unmerge, r


def _tome_bipartite_soft_matching_scores(scores, num_tokens, r, return_metadata=False):
    """
    ToMe matching on a precomputed bipartite score matrix.

    Args:
        scores: [batch, even_tokens, odd_tokens]
        num_tokens: original token count before even/odd split
        r: requested removed token count
    """
    if scores.dim() != 3:
        raise ValueError("scores must have shape [batch, even_tokens, odd_tokens].")

    max_pairs = min(int(num_tokens // 2), int(scores.shape[-1]))
    r = min(int(r), max_pairs)
    if r <= 0 or scores.shape[-2] == 0 or scores.shape[-1] == 0:
        if return_metadata:
            return _tome_do_nothing, _tome_do_nothing, 0, None
        return _tome_do_nothing, _tome_do_nothing, 0

    with torch.no_grad():
        node_max, node_idx = scores.max(dim=-1)
        finite_mask = torch.isfinite(node_max)
        feasible_count = int(finite_mask[0].sum().detach().cpu())
        actual_r = min(r, feasible_count)
        if actual_r <= 0:
            if return_metadata:
                return _tome_do_nothing, _tome_do_nothing, 0, None
            return _tome_do_nothing, _tome_do_nothing, 0

        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        unm_idx = edge_idx[..., actual_r:, :]
        src_idx = edge_idx[..., :actual_r, :]
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

    def merge(x, mode="mean"):
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - actual_r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, actual_r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, actual_r, c), src, reduce=mode)
        return torch.cat([unm, dst], dim=1)

    def unmerge(x):
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, actual_r, c))
        out = torch.zeros(n, num_tokens, c, device=x.device, dtype=x.dtype)
        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, actual_r, c), src=src)
        return out

    if return_metadata:
        metadata = {
            "unm_idx": unm_idx,
            "src_idx": src_idx,
            "dst_idx": dst_idx,
            "num_tokens": int(num_tokens),
            "feasible_count": feasible_count,
        }
        return merge, unmerge, actual_r, metadata
    return merge, unmerge, actual_r


def _tome_merge_wavg(merge, x, size=None):
    if size is None:
        size = torch.ones_like(x[..., 0, None])
    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")
    x = x / size
    return x, size


class BoundaryDetector(nn.Module):
    """GPU-only token split for boundary-preserved valid-token compression."""

    def __init__(self, ring_radius=1, safe_threshold=0.999, ablate_valid_token_restriction=False):
        super().__init__()
        self.ring_radius = int(ring_radius)
        self.safe_threshold = float(safe_threshold)
        self.ablate_valid_token_restriction = bool(ablate_valid_token_restriction)

    def _mask_to_token_ratio(self, mask, token_shape):
        token_shape = tuple(int(x) for x in token_shape)
        mask = mask.float()
        if tuple(mask.shape[-2:]) == token_shape:
            return mask

        h, w = mask.shape[-2:]
        th, tw = token_shape
        if h % th == 0 and w % tw == 0:
            kernel_size = (h // th, w // tw)
            return F.avg_pool2d(mask, kernel_size=kernel_size, stride=kernel_size)
        return F.interpolate(mask, size=token_shape, mode="area")

    def forward(self, mask, token_shape):
        """
        Args:
            mask: B x 1 x H x W, where 1 means known/valid pixels in PUT.
            token_shape: spatial shape of UQ-Transformer tokens.

        Returns:
            Boolean token maps in B x 1 x Ht x Wt.
        """
        if mask.dim() != 4 or mask.shape[1] != 1:
            raise ValueError("BoundaryDetector expects mask with shape [B, 1, H, W].")

        token_visible_ratio = self._mask_to_token_ratio(mask, token_shape).clamp(0.0, 1.0)
        valid_token = token_visible_ratio >= self.safe_threshold
        masked_or_partial_token = ~valid_token

        if self.ring_radius > 0:
            kernel_size = self.ring_radius * 2 + 1
            dilated_risk = F.max_pool2d(
                masked_or_partial_token.float(),
                kernel_size=kernel_size,
                stride=1,
                padding=self.ring_radius,
            ).bool()
            boundary_token = dilated_risk & valid_token
        else:
            boundary_token = torch.zeros_like(valid_token)

        protect_mask_token = masked_or_partial_token | boundary_token
        if self.ablate_valid_token_restriction:
            safe_candidate_token = torch.ones_like(valid_token, dtype=torch.bool)
        else:
            safe_candidate_token = valid_token & ~boundary_token

        return {
            "token_visible_ratio": token_visible_ratio,
            "valid_token": valid_token,
            "masked_or_partial_token": masked_or_partial_token,
            "boundary_token": boundary_token,
            "protect_mask_token": protect_mask_token,
            "safe_candidate_token": safe_candidate_token,
        }


class SimilarityScorer(nn.Module):
    """Local redundancy scorer over safe tokens only."""

    def __init__(self, kernel_size=3, alpha=1.0, beta=0.15, topk_ratio=0.25, learnable=False):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("SimilarityScorer requires an odd kernel size.")
        self.kernel_size = int(kernel_size)
        self.padding = self.kernel_size // 2
        self.topk_ratio = float(topk_ratio)
        self.learnable = bool(learnable)
        self.alpha_param = nn.Parameter(torch.tensor(float(alpha), dtype=torch.float32))
        self.beta_param = nn.Parameter(torch.tensor(float(beta), dtype=torch.float32))
        self.alpha_param.requires_grad = self.learnable
        self.beta_param.requires_grad = self.learnable

    @property
    def alpha(self):
        return float(self.alpha_param.detach().cpu())

    @property
    def beta(self):
        return float(self.beta_param.detach().cpu())

    def alpha_tensor(self):
        return torch.clamp(self.alpha_param, min=0.0)

    def beta_tensor(self):
        return torch.clamp(self.beta_param, min=0.0)

    @staticmethod
    def _masked_mean(values, mask, dim, keepdim=False):
        mask = mask.to(values.dtype)
        denom = mask.sum(dim=dim, keepdim=keepdim).clamp(min=1.0)
        return (values * mask).sum(dim=dim, keepdim=keepdim) / denom

    @staticmethod
    def _normalize_safe_map(score, safe_mask, zero_threshold=1e-8):
        score = score.clone()
        b = score.shape[0]
        flat_score = score.view(b, -1)
        flat_safe = safe_mask.view(b, -1)
        out = torch.zeros_like(flat_score)

        for batch_idx in range(b):
            valid = flat_safe[batch_idx]
            if valid.sum() == 0:
                continue
            score_valid = flat_score[batch_idx, valid]
            score_min = score_valid.min()
            score_max = score_valid.max()
            if float((score_max - score_min).detach().cpu()) < float(zero_threshold):
                out[batch_idx, valid] = 0.0
            else:
                out[batch_idx, valid] = (score_valid - score_min) / (score_max - score_min)
        return out.view_as(score)

    def _distance_rule_score(self, protect_mask_token, safe_candidate_token):
        protect = protect_mask_token.bool()
        safe = safe_candidate_token.bool()
        distance = torch.zeros_like(protect_mask_token, dtype=torch.float32)
        frontier = protect.float()
        visited = protect.clone()
        max_steps = int(protect_mask_token.shape[-2] + protect_mask_token.shape[-1])

        for step in range(1, max_steps + 1):
            dilated = F.max_pool2d(frontier, kernel_size=3, stride=1, padding=1)
            ring = (dilated > 0.5) & (~visited) & safe
            if not ring.any():
                break
            distance[ring] = float(step)
            frontier = ring.float()
            visited = visited | ring

        distance = distance * safe.float()
        distance_norm = self._normalize_safe_map(distance, safe)
        return distance, distance_norm

    def _topk_mask(self, score_norm, safe_mask):
        b = score_norm.shape[0]
        flat_score = score_norm.view(b, -1)
        flat_safe = safe_mask.view(b, -1)
        out = torch.zeros_like(flat_safe, dtype=torch.bool)

        for batch_idx in range(b):
            valid_idx = flat_safe[batch_idx].nonzero(as_tuple=False).flatten()
            if valid_idx.numel() == 0:
                continue
            k = max(int(math.ceil(valid_idx.numel() * self.topk_ratio)), 1)
            values = flat_score[batch_idx, valid_idx]
            _, topk = torch.topk(values, k=k, dim=0, largest=True)
            out[batch_idx, valid_idx[topk]] = True
        return out.view_as(safe_mask)

    def forward(self, x, protect_mask_token, safe_candidate_token):
        """
        Args:
            x: B x H x W x C
            protect_mask_token: B x 1 x H x W
            safe_candidate_token: B x 1 x H x W
        """
        if x.dim() != 4:
            raise ValueError("SimilarityScorer expects token features with shape [B, H, W, C].")

        feat = x.permute(0, 3, 1, 2).contiguous()
        safe = safe_candidate_token.bool()
        protect = protect_mask_token.bool()
        b, c, h, w = feat.shape
        num_neighbors = self.kernel_size * self.kernel_size
        center_idx = num_neighbors // 2

        feat_norm = F.normalize(feat, dim=1, eps=1e-6)
        feat_unfold = F.unfold(feat, kernel_size=self.kernel_size, padding=self.padding)
        feat_unfold = feat_unfold.view(b, c, num_neighbors, h, w)
        feat_norm_unfold = F.unfold(feat_norm, kernel_size=self.kernel_size, padding=self.padding)
        feat_norm_unfold = feat_norm_unfold.view(b, c, num_neighbors, h, w)

        center_norm = feat_norm.unsqueeze(dim=2)
        cos_sim = (feat_norm_unfold * center_norm).sum(dim=1)  # B x K x H x W

        center_feat = feat.unsqueeze(dim=2)
        local_diff = (feat_unfold - center_feat).pow(2).mean(dim=1)  # B x K x H x W

        safe_neighbors = F.unfold(safe.float(), kernel_size=self.kernel_size, padding=self.padding)
        safe_neighbors = safe_neighbors.view(b, 1, num_neighbors, h, w).bool().squeeze(dim=1)
        safe_neighbors[:, center_idx, :, :] = False

        neighbor_count = safe_neighbors.sum(dim=1, keepdim=True)
        mean_cos = self._masked_mean(cos_sim, safe_neighbors, dim=1, keepdim=True)
        local_variance = self._masked_mean(local_diff, safe_neighbors, dim=1, keepdim=True)

        safe_float = safe.float()
        safe_has_neighbor = (neighbor_count > 0).float()
        variance_safe_mask = safe & (neighbor_count > 0)
        local_variance_norm = self._normalize_safe_map(
            local_variance * safe_float * safe_has_neighbor,
            variance_safe_mask,
            zero_threshold=1e-6,
        )
        alpha = self.alpha_tensor().to(feat.dtype)
        beta = self.beta_tensor().to(feat.dtype)
        redundancy_score = (alpha * mean_cos - beta * local_variance_norm) * safe_float * safe_has_neighbor
        redundancy_score_norm = self._normalize_safe_map(redundancy_score, safe)

        distance_raw, distance_score = self._distance_rule_score(protect, safe)
        topk_similarity = self._topk_mask(redundancy_score_norm, safe)
        topk_distance = self._topk_mask(distance_score, safe)
        overlap = (topk_similarity & topk_distance & safe).sum(dim=(1, 2, 3)).float()
        denom = (topk_similarity & safe).sum(dim=(1, 2, 3)).float().clamp(min=1.0)
        topk_overlap_ratio = overlap / denom

        return {
            "mean_cos_sim": mean_cos,
            "local_variance": local_variance,
            "local_variance_norm": local_variance_norm,
            "safe_neighbor_count": neighbor_count.float(),
            "redundancy_score": redundancy_score,
            "redundancy_score_norm": redundancy_score_norm,
            "distance_score": distance_score,
            "distance_raw": distance_raw,
            "topk_similarity_mask": topk_similarity,
            "topk_distance_mask": topk_distance,
            "topk_overlap_ratio": topk_overlap_ratio,
        }


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


def window_partition(x, window_size, partition_type='partition'):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    if partition_type == 'partition':
        x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    elif partition_type == 'shuffle':
        nw = int(H * W / window_size[0] / window_size[1])
        windows = pixel_unshuffle(x.permute(0, 3, 1, 2), out_size=window_size, chunked=True).permute(0, 2, 3, 1) # B x wH x wW x C*nW
        windows = torch.chunk(windows, chunks=nw, dim=-1) # tuple(B x wH x wW x C), nw
        windows = torch.cat(windows, dim=0) # B*nW x wH x wW x C
    # import pdb; pdb.set_trace()
    return windows


def window_unpartition(windows, window_size, H, W, partition_type='partition'):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    if partition_type == 'partition':
        x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    else:
        nw = int(H * W / window_size[0] / window_size[1])
        x = torch.chunk(windows, chunks=nw, dim=0) # tuple(B x wH x wW x C), nw
        x = torch.cat(x, dim=-1) # B x wH x wW x C*nW
        x = pixel_shuffle(x.permute(0, 3, 1, 2), out_size=(H, W), chunked=True) # B x C x H x W
        x = x.permute(0, 2, 3, 1)
    return x



class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, 
                dim, 
                num_heads=8, 
                qkv_bias=False, 
                attn_drop=0., 
                proj_drop=0.,
        
                window_size=None, # the window size to perform attention, similar to swin transformer
                shift_size=(0,0), # the size to cycle shift the input feature, similar to swin transformer, only effective when window size is not None
                partition_type='shuffle',
                apply_window_mask=False,
    ):
        super().__init__()

        if window_size is not None and not isinstance(window_size, (tuple, list)):
            window_size = (window_size, window_size)
        if shift_size is not None and not isinstance(shift_size, (tuple, list)):
            shift_size = (shift_size, shift_size)
        self.window_size = tuple(window_size) if window_size is not None else None
        self.shift_size = tuple(shift_size) if shift_size is not None else None
        self.apply_window_mask = apply_window_mask
        self.window_mask = None

        for pt in partition_type.split(','):
            assert pt in ['shuffle', 'partition'], 'not implemented reduce type {}'.format(pt)
        self.partition_type = partition_type 

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        all_head_dim = head_dim * self.num_heads
        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def set_window_attn_mask(self, resolution, partition_type='partition', device=None):
        assert partition_type == 'partition', 'Following swin transformer, the mask is only needed for parition window!'
        H, W = resolution 
        img_mask = torch.zeros((1, H, W, 1))  # 1 x H x W x 1
        h_slices = (slice(0, -self.window_size[0]),
                    slice(-self.window_size[0], -self.shift_size[0]),
                    slice(-self.shift_size[0], None))
        w_slices = (slice(0, -self.window_size[1]),
                    slice(-self.window_size[1], -self.shift_size[1]),
                    slice(-self.shift_size[1], None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, window_size=self.window_size, partition_type='partition')  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1]) # nw x wH*wW
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2) # nw x wH*wW x wH*wH
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        if device is not None:
            attn_mask = attn_mask.to(device)
        self.window_mask = attn_mask
        # import pdb; pdb.set_trace()
        return attn_mask

    def atten(self, qkv, mask=None, window_mask=None):
        """
        qkv: 3 x B x H x N x C
        mask: B x N
        window_mask: nw x N x N
        """
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple) # B x H x N x C
        b, h, n, c = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale # B x H x N x N

        if window_mask is not None:
            nw = window_mask.shape[0]
            attn = attn.view(b//nw, nw, h, n, n) + window_mask.to(attn).unsqueeze(dim=1).unsqueeze(0)
            attn = attn.view(b, h, n, n)
        if mask is not None:
            mask = mask.view(b, 1, 1, n)
            attn = attn.masked_fill(~mask, float('-inf'))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn) # B x H x N x N
        x = attn @ v # B x H x N x C
        return x, attn


    def forward(self, x, mask=None):
        """
        x: B x H x W x C
        mask: None or B x H x W
        """
        B, H, W, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias)) # 3*C
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias) # B x H x W x 3*C

        if self.window_size is not None and self.window_size != (H, W):
            assert self.shift_size >= (0,0) and self.shift_size < (H, W), 'shift size should be in range (0,0)-(H,W)'
            if self.shift_size > (0,0):
                qkv = torch.roll(qkv, shifts=self.shift_size, dims=(1,2))
                if mask is not None:
                    mask = torch.roll(mask, shifts=self.shift_size, dims=(1,2))
            
            pt_list = self.partition_type.split(',')
            qkv_list = torch.chunk(qkv, chunks=len(pt_list), dim=-1) # B x H x W x 3*C/pt
            x = []
            for i in range(len(pt_list)):
                pt_ = pt_list[i]
                qkv_ = qkv_list[i]
                qkv_ = window_partition(qkv_, window_size=self.window_size, partition_type=pt_) # B*nW x wH x wW x 3*C/pt
                if mask is not None:
                    mask_ = window_partition(mask.unsqueeze(dim=-1), window_size=self.window_size, partition_type=pt_).squeeze(dim=-1) # B*nW x wH x wW
                else:
                    mask_ = None
                b, h, w, _ = qkv_.shape
                qkv_ = qkv_.reshape(b, h*w, 3, self.num_heads//len(pt_list), C // self.num_heads).permute(2, 0, 3, 1, 4) # b x hw x 3 x Head/2 x C/Head -> 3 x b x Head/pt x hw x C/Head
                if self.apply_window_mask and pt_ == 'partition':
                    if self.window_mask is None:
                        self.set_window_attn_mask(resolution=(H,W), partition_type=pt_, device=qkv_.device)
                    window_mask = self.window_mask
                else:
                    window_mask = None
                x_, attn_, = self.atten(qkv_, mask=mask_, window_mask=window_mask) # b x H/pt x hw x C/H, b x H/pt x hw x hw
                            
                x_ = x_.permute(0, 2, 1, 3).contiguous().view(b, h, w, C//len(pt_list)) # b x Head/pt x hw x C/Head -> b x hw x Head/pt x C/Head -> b x h x w x C/pt
                x_ = window_unpartition(x_, window_size=self.window_size, H=H, W=W, partition_type=pt_) # B x H x W x C/pt

                x.append(x_) # B x H x W x C
            x = torch.cat(x, dim=-1)
        
            if self.shift_size > (0,0):
                x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1,2))

            attn = None
        else:
            qkv = qkv.reshape(B, H*W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # B x H x W x 3C -> B x HW x 3 x H x C/Head -> 3 x B x Head x HW x C/Head
            x, attn = self.atten(qkv, mask=mask) # B x Head x HW x C/Head, B x Head x HW x HW
            x = x.permute(0, 2, 1, 3).contiguous().view(B, H, W, C) # B x Head x HW x C/Head -> B x HW x Head x C/Head -> B x H x W x C
            attn = attn.mean(1, keepdim=False).view(B, H, W, H, W)
        
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn


    @staticmethod
    def count_flops(m, x, y):
        from thop.vision import basic_hooks, calc_func
        m.total_params = m.qkv.total_params + m.proj.total_params
        if m.q_bias is not None:
            m.total_params += m.q_bias.numel()
            m.total_params += m.v_bias.numel()

        total_ops = 0

        B, H, W, C = x[0].shape
        # qkv linear
        total_ops += calc_func.calculate_linear(m.qkv.in_features, B*H*W*m.qkv.out_features)

        if m.window_size is not None and m.window_size != (H, W):
            raise NotImplementedError
        else:
            # atten
            head_dim = m.qkv.out_features // 3 // m.num_heads
            total_ops += (B*m.num_heads*H*W*H*W*head_dim) # bmm, matrix multiply
            total_ops += (B*m.num_heads*H*W*H*W) # x scale
            total_ops += calc_func.calculate_softmax(B*m.num_heads*H*W, H*W) # softmax
            total_ops += (B*m.num_heads*H*W*C*H*W)# bmm, matrix multiply
        
        total_ops += m.proj.total_ops
        m.total_ops += total_ops
        

class GELU2(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x * F.sigmoid(1.702 * x)
    
    @staticmethod
    def count_flops(m, x, y):
        m.total_ops += torch.DoubleTensor([2 * y.numel()])



class Block(nn.Module):

    def __init__(self, 
                dim, 
                num_heads, 
                mlp_ratio=4., 
                qkv_bias=False, 
                attn_drop=0.,
                drop_path=0., 
                drop=0.0,
                act_layer='GELU', 
                norm_layer=nn.LayerNorm,
                window_size=None, # the window size to perform attention, similar to swin transformer
                shift_size=(0,0), # the size to cycle shift the input feature, similar to swin transformer, only effective when window size is not None
                partition_type='shuffle',
                apply_window_mask=False,
        ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                              window_size=window_size, shift_size=shift_size, partition_type=partition_type, apply_window_mask=apply_window_mask)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if act_layer == 'GELU':
            act_layer = nn.GELU 
        elif act_layer == 'GELU2':
            act_layer = GELU2
        else:
            raise NotImplementedError('activation layer {} not implemented!'.format(act_layer))
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        object.__setattr__(self, '_phase4a_owner', None)

    def forward(self, x, mask=None):
        phase4a_owner = getattr(self, '_phase4a_owner', None)
        block_total_handle = None
        attention_handle = None
        if phase4a_owner is not None:
            block_total_handle = phase4a_owner._phase4a_start_timing('Time_Block_Total')
            attention_handle = phase4a_owner._phase4a_start_timing('Time_Attention')
        x_, attn = self.attn(self.norm1(x), mask=mask) # B x H x W x C, B x H x W x H x W
        if attention_handle is not None:
            phase4a_owner._phase4a_end_timing(attention_handle)
        x = x + self.drop_path(x_)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if block_total_handle is not None:
            phase4a_owner._phase4a_end_timing(block_total_handle)
        return x, attn


class MaskedImageInpaintingTransformer(nn.Module):
    def __init__(
        self,
        *,
        content_seq_len, # length of content sequences
        embd_pdrop=0., # embedding dropout prob
        
        n_layer, # number of layers in transformer
        dim, # the embed dim
        num_heads, # the number of heads
        attn_drop=0.0, # attention dropout prob
        drop_path=0.0, # drop path prob
        act_layer='GELU', # the activation layer in MLP of transformer block
        mlp_ratio=4, # the times of hidden dimension in the MLP of attetntion block
        qkv_bias=True, # the bias for qkv in attention block

        attn_window_size=[None],
        attn_shift_size=[(0,0)],
        attn_partition_type=['shuffle'],
        attn_window_mask=False,

        attn_content_with_mask=False,
        content_codec_config=None,

        content_ignore_token=-100,

        input_feature_type='origin',

        learn_mask_emb=False,
        mask_pixel_value=None,

        init_type='beit', # how to initialize the weight

        # args for training
        weight_decay=0.01,
        random_quantize=0.2, # random quantize the feature, only when the input feature is not quantized
        num_token=None, # i.e the numbr of classes
        content_patch_token_shape=[1, 1], # the shape of tokens in each patch, h, w
        ckpt_path=None, # The pretrained model to load 
        loss_config=None,
        loss_mask_type='binary', # binary: the patch with any pixel missed, other use the unmasked ratio as the mask
    ):
        super().__init__()
        
        content_patch_token_shape = tuple(int(ts) for ts in content_patch_token_shape)
        content_patch_seq_len = int(content_patch_token_shape[0] * content_patch_token_shape[1])
        assert dim % content_patch_seq_len == 0, 'The number of dimmension should be divisible by the number of tokens '
        
        # embeddings for content
        self.content_codec = instantiate_from_config(content_codec_config)
        self.emb_proj = nn.Linear(self.content_codec.embed_dim, dim//content_patch_seq_len)
        self.pos_emb = nn.Parameter(torch.zeros(1, content_seq_len, dim))
        if learn_mask_emb:
            self.mask_emb = nn.Parameter(torch.zeros(1, dim//content_patch_seq_len, 1, 1))
        else:
            self.mask_emb = None
        
        # drop for embedding
        if embd_pdrop > 0:
            self.drop = nn.Dropout(embd_pdrop)
        else:
            self.drop = None
                             
        # transformer
        attn_window_size = (math.ceil(float(n_layer)/len(attn_window_size)) * attn_window_size)[:n_layer]
        attn_shift_size = (math.ceil(float(n_layer)/len(attn_shift_size)) * attn_shift_size)[:n_layer]
        attn_partition_type = (math.ceil(float(n_layer)/len(attn_partition_type)) * attn_partition_type)[:n_layer]
        dpr = [x.item() for x in torch.linspace(0, drop_path, n_layer)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[Block(
                dim=dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, 
                attn_drop=attn_drop,
                drop_path=dpr[n], 
                act_layer=act_layer, 
                window_size=attn_window_size[n],
                shift_size=attn_shift_size[n],
                partition_type=attn_partition_type[n],
                apply_window_mask=attn_window_mask,
        ) for n in range(n_layer)])
        for block in self.blocks:
            object.__setattr__(block, '_phase4a_owner', self)

        # final prediction head
        self.norm = nn.LayerNorm(dim)
        self.num_cls = self.content_codec.get_number_of_tokens() if num_token is None else num_token
        self.to_logits = nn.Linear(dim//content_patch_seq_len, self.num_cls)
        
        self.dim = dim
        self.attn_content_with_mask = attn_content_with_mask
        self.content_seq_len = content_seq_len
        self.content_patch_seq_len = content_patch_seq_len
        self.content_patch_token_shape = content_patch_token_shape
        self.content_ignore_token = content_ignore_token
        self.input_feature_type = input_feature_type
        self.weight_decay = weight_decay
        self.random_quantize = random_quantize
        self.mask_pixel_value = mask_pixel_value
        self.global_tome_r = int(os.environ.get('PUT_GLOBAL_TOME_R', '0'))
        self.global_tome_profile = {
            'enabled': self.global_tome_r > 0,
            'requested_r': self.global_tome_r,
            'calls': 0,
            'original_tokens': 0,
            'merged_tokens': 0,
            'removed_tokens': 0,
            'restored_tokens': 0,
            'pair_count': 0,
        }
        self.boundary_split_enabled = os.environ.get('PUT_BOUNDARY_SPLIT', '0') == '1'
        self.boundary_detector = BoundaryDetector(
            ring_radius=int(os.environ.get('PUT_BOUNDARY_RING_RADIUS', '1')),
            safe_threshold=float(os.environ.get('PUT_BOUNDARY_SAFE_THRESHOLD', '0.999')),
            ablate_valid_token_restriction=os.environ.get('PUT_ABLATE_VALID_TOKEN_RESTRICTION', '0') == '1',
        )
        self.boundary_split_profile = {
            'enabled': self.boundary_split_enabled,
            'ring_radius': self.boundary_detector.ring_radius,
            'safe_threshold': self.boundary_detector.safe_threshold,
            'calls': 0,
            'token_shape': [],
            'total_tokens': 0,
            'valid_tokens': 0,
            'masked_or_partial_tokens': 0,
            'boundary_tokens': 0,
            'protect_tokens': 0,
            'safe_candidate_tokens': 0,
        }
        self.similarity_scorer_enabled = os.environ.get('PUT_SIMILARITY_SCORER', '0') == '1'
        self.similarity_scorer_learnable = os.environ.get('PUT_SIMILARITY_LEARNABLE', '0') == '1'
        self.similarity_guidance_source = os.environ.get(
            'PUT_SIMILARITY_GUIDANCE_SOURCE',
            'projected',
        ).strip().lower()
        if self.similarity_guidance_source not in {
            'projected',
            'codec_feature_detached',
            'codec_quantized_detached',
        }:
            raise ValueError(
                "PUT_SIMILARITY_GUIDANCE_SOURCE must be one of "
                "{'projected', 'codec_feature_detached', 'codec_quantized_detached'}, "
                f"got {self.similarity_guidance_source!r}."
            )
        self.similarity_scorer = SimilarityScorer(
            kernel_size=int(os.environ.get('PUT_SIMILARITY_KERNEL_SIZE', '3')),
            alpha=float(os.environ.get('PUT_SIMILARITY_ALPHA', '1.0')),
            beta=float(os.environ.get('PUT_SIMILARITY_BETA', '0.15')),
            topk_ratio=float(os.environ.get('PUT_SIMILARITY_TOPK_RATIO', '0.25')),
            learnable=self.similarity_scorer_learnable,
        )
        self.similarity_profile = {
            'enabled': self.similarity_scorer_enabled,
            'learnable': self.similarity_scorer.learnable,
            'guidance_source': self.similarity_guidance_source,
            'guidance_detached': self.similarity_guidance_source != 'projected',
            'kernel_size': self.similarity_scorer.kernel_size,
            'alpha': self.similarity_scorer.alpha,
            'beta': self.similarity_scorer.beta,
            'topk_ratio': self.similarity_scorer.topk_ratio,
            'calls': 0,
            'token_shape': [],
            'safe_tokens': 0,
            'protect_tokens': 0,
            'score_mean': 0.0,
            'score_min': 0.0,
            'score_max': 0.0,
            'score_std': 0.0,
            'distance_mean': 0.0,
            'distance_min': 0.0,
            'distance_max': 0.0,
            'distance_std': 0.0,
            'mean_cos_sim': 0.0,
            'mean_cos_sim_min': 0.0,
            'mean_cos_sim_max': 0.0,
            'mean_cos_sim_std': 0.0,
            'local_variance_mean': 0.0,
            'local_variance_min': 0.0,
            'local_variance_max': 0.0,
            'local_variance_std': 0.0,
            'local_variance_norm_mean': 0.0,
            'local_variance_norm_min': 0.0,
            'local_variance_norm_max': 0.0,
            'local_variance_norm_std': 0.0,
            'topk_overlap_ratio': 0.0,
        }
        self.safe_tome_r = int(os.environ.get('PUT_SAFE_TOME_R', '0'))
        self.safe_tome_enabled = self.safe_tome_r > 0
        gaspg_debug_override = os.environ.get('PUT_GASPG_DEBUG')
        if gaspg_debug_override is None:
            self.safe_tome_debug_enabled = os.environ.get('PUT_SAFE_TOME_DEBUG', '0') == '1'
            self.ga_spg_audit_enabled = os.environ.get('PUT_GA_SPG_AUDIT', '0') == '1'
        else:
            gaspg_debug_enabled = gaspg_debug_override == '1'
            self.safe_tome_debug_enabled = gaspg_debug_enabled and self.safe_tome_enabled
            self.ga_spg_audit_enabled = gaspg_debug_enabled
        self.safe_tome_score_mode = os.environ.get('PUT_SAFE_TOME_SCORE_MODE', 'similarity').strip().lower()
        if self.safe_tome_score_mode not in {'similarity', 'distance', 'ga_spg_soft', 'ga_spg_hardveto', 'ga_spg_lite'}:
            raise ValueError(
                "PUT_SAFE_TOME_SCORE_MODE must be one of "
                "{'similarity', 'distance', 'ga_spg_soft', 'ga_spg_hardveto', 'ga_spg_lite'}, "
                f"got {self.safe_tome_score_mode!r}."
            )
        self.ga_spg_pair_mode = self.safe_tome_score_mode in {'ga_spg_soft', 'ga_spg_hardveto'}
        self.ga_spg_lite_mode = self.safe_tome_score_mode == 'ga_spg_lite'
        self.ga_spg_mode = self.ga_spg_pair_mode or self.ga_spg_lite_mode
        self.ga_spg_guidance_enabled = self.ga_spg_mode or self.ga_spg_audit_enabled
        self.ga_spg_lite_optimized = os.environ.get('PUT_GA_SPG_LITE_OPTIMIZED', '0') == '1'
        self.ga_spg_lite_runtime_cache_enabled = os.environ.get(
            'PUT_GA_SPG_LITE_RUNTIME_CACHE',
            '1' if self.ga_spg_lite_optimized else '0',
        ) == '1'
        self.ga_spg_lite_pad_mode = os.environ.get('PUT_GA_SPG_LITE_PAD_MODE', 'replicate').strip().lower()
        if self.ga_spg_lite_pad_mode not in {'zero', 'replicate'}:
            raise ValueError(
                "PUT_GA_SPG_LITE_PAD_MODE must be one of {'zero', 'replicate'}, "
                f"got {self.ga_spg_lite_pad_mode!r}."
            )
        self.ga_spg_lite_explicit_border_risk = os.environ.get('PUT_GA_SPG_LITE_EXPLICIT_BORDER_RISK', '0') == '1'
        self.ga_spg_lambda_risk = float(os.environ.get('PUT_GA_SPG_LAMBDA_RISK', '0.35'))
        self.ga_spg_weights = {
            'cos_3x3': float(os.environ.get('PUT_GA_SPG_W1', '0.35')),
            'cos_9x9': float(os.environ.get('PUT_GA_SPG_W2', '0.35')),
            'context_variance': float(os.environ.get('PUT_GA_SPG_W3', '0.15')),
            'boundary_risk': float(os.environ.get('PUT_GA_SPG_W4', '0.15')),
        }
        self.ga_spg_lite_lambda_risk = float(os.environ.get('PUT_GA_SPG_LITE_LAMBDA', '0.2'))
        self.ga_spg_lite_weights = {
            'texture_risk': float(os.environ.get('PUT_GA_SPG_LITE_W_TEXTURE', '0.45')),
            'boundary_risk': float(os.environ.get('PUT_GA_SPG_LITE_W_BOUNDARY', '0.35')),
            'smoothness_risk': float(os.environ.get('PUT_GA_SPG_LITE_W_SMOOTHNESS', '0.20')),
        }
        self.ga_spg_lite_image_border_weight = float(os.environ.get('PUT_GA_SPG_LITE_W_IMAGE_BORDER', '0.20'))
        self.ga_spg_hard_risk_threshold = float(os.environ.get('PUT_GA_SPG_HARD_THRESHOLD', '0.65'))
        self.ga_spg_hard_penalty = float(os.environ.get('PUT_GA_SPG_HARD_PENALTY', '1000000.0'))
        ga_spg_weights = dict(self.ga_spg_lite_weights) if self.ga_spg_lite_mode else dict(self.ga_spg_weights)
        ga_spg_lambda = self.ga_spg_lite_lambda_risk if self.ga_spg_lite_mode else self.ga_spg_lambda_risk
        if self.ga_spg_lite_mode:
            ga_spg_weights['image_border_risk'] = float(self.ga_spg_lite_image_border_weight)
        self.ga_spg_profile = {
            'enabled': self.ga_spg_guidance_enabled,
            'mode': self.safe_tome_score_mode if self.ga_spg_mode else 'audit_only',
            'guidance_source': self.similarity_guidance_source,
            'optimized': bool(self.ga_spg_lite_mode and self.ga_spg_lite_optimized),
            'runtime_cache_enabled': bool(self.ga_spg_lite_mode and self.ga_spg_lite_runtime_cache_enabled),
            'pad_mode': self.ga_spg_lite_pad_mode if self.ga_spg_lite_mode else 'n/a',
            'explicit_image_border_risk': bool(self.ga_spg_lite_mode and self.ga_spg_lite_explicit_border_risk),
            'lambda_risk': ga_spg_lambda,
            'weights': ga_spg_weights,
            'hard_risk_threshold': self.ga_spg_hard_risk_threshold,
            'hard_penalty': self.ga_spg_hard_penalty,
            'calls': 0,
            'safe_tokens': 0,
            'protect_tokens': 0,
            'distance_mean': 0.0,
            'distance_min': 0.0,
            'distance_max': 0.0,
            'distance_std': 0.0,
            'local_variance_mean': 0.0,
            'local_variance_min': 0.0,
            'local_variance_max': 0.0,
            'local_variance_std': 0.0,
            'context_variance_mean': 0.0,
            'context_variance_min': 0.0,
            'context_variance_max': 0.0,
            'context_variance_std': 0.0,
            'boundary_risk_mean': 0.0,
            'boundary_risk_min': 0.0,
            'boundary_risk_max': 0.0,
            'boundary_risk_std': 0.0,
            'image_border_risk_mean': 0.0,
            'image_border_risk_min': 0.0,
            'image_border_risk_max': 0.0,
            'image_border_risk_std': 0.0,
            'smoothness_risk_mean': 0.0,
            'smoothness_risk_min': 0.0,
            'smoothness_risk_max': 0.0,
            'smoothness_risk_std': 0.0,
            'scalar_risk_mean': 0.0,
            'scalar_risk_min': 0.0,
            'scalar_risk_max': 0.0,
            'scalar_risk_std': 0.0,
        }
        self.safe_tome_profile = {
            'enabled': self.safe_tome_enabled,
            'requested_r': self.safe_tome_r,
            'score_mode': self.safe_tome_score_mode,
            'optimized': bool(self.ga_spg_lite_mode and self.ga_spg_lite_optimized),
            'runtime_cache_enabled': bool(self.ga_spg_lite_mode and self.ga_spg_lite_runtime_cache_enabled),
            'pad_mode': self.ga_spg_lite_pad_mode if self.ga_spg_lite_mode else 'n/a',
            'explicit_image_border_risk': bool(self.ga_spg_lite_mode and self.ga_spg_lite_explicit_border_risk),
            'calls': 0,
            'token_shape': [],
            'original_tokens': 0,
            'compressed_tokens': 0,
            'protect_tokens': 0,
            'safe_tokens': 0,
            'eligible_tokens': 0,
            'merged_tokens': 0,
            'removed_tokens': 0,
            'restored_tokens': 0,
            'pair_count': 0,
            'selected_score_mean': 0.0,
            'selected_score_min': 0.0,
            'selected_score_max': 0.0,
            'selected_score_std': 0.0,
            'actual_compression_ratio': 0.0,
            'protect_overlap_tokens': 0,
            'invalid_pair_count': 0,
            'high_risk_pair_count': 0,
            'mean_risk_penalty_selected': 0.0,
            'mean_base_metric_selected': 0.0,
            'mean_adjusted_metric_selected': 0.0,
            'under_compression': False,
            'fallback_used': False,
            'fallback_reason': '',
            'restore_alignment_ok': True,
        }
        self._safe_tome_selection_cache_signature = None
        self._safe_tome_selection_cache = None
        self._safe_tome_runtime_cache_signature = None
        self._safe_tome_runtime_cache = None
        self.phase5k_lean_profile_enabled = os.environ.get('PUT_PHASE5K_LEAN_PROFILE', '0') == '1'
        self.phase5k_audit_enabled = os.environ.get('PUT_PHASE5K_AUDIT', '0') == '1'
        self.phase5k_audit_profile = {
            'enabled': self.phase5k_audit_enabled,
            'records': [],
        }
        self.phase4a_timing_enabled = os.environ.get('PUT_PHASE4A_TIMING', '0') == '1'
        self.phase4a_timing_buckets = [
            'Time_Scoring',
            'Time_TopK_and_Pairing',
            'Time_Merge',
            'Time_Attention',
            'Time_Restore',
            'Time_Block_Total',
        ]
        self.phase4a_timing_profile = {
            'enabled': self.phase4a_timing_enabled,
            'calls': 0,
            'num_sampling_steps': 0,
            'timings_ms': {bucket: 0.0 for bucket in self.phase4a_timing_buckets},
            'event_counts': {bucket: 0 for bucket in self.phase4a_timing_buckets},
            'token_profile': {
                'original_tokens': 0,
                'eligible_tokens': 0,
                'removed_tokens': 0,
                'merged_tokens': 0,
                'restored_tokens': 0,
            },
        }
        self._phase4a_event_buckets = None
        self._phase4a_num_sampling_steps = 0
        self.phase5f_timing_enabled = os.environ.get('PUT_PHASE5F_TIMING', '0') == '1'
        self.phase5f_timing_buckets = [
            'boundary_split_time',
            'pure_distance_pool_time',
            'distance_map_time',
            'codec_feature_prepare_time',
            'token_risk_compute_time',
            'avgpool_3x3_time',
            'avgpool_9x9_time',
            'runtime_cache_build_time',
            'runtime_cache_hit_time',
            'pair_metric_base_time',
            'risk_penalty_matrix_time',
            'adjusted_metric_time',
            'topk_matching_time',
            'merge_time',
            'restore_time',
            'transformer_blocks_time',
            'total_inference_time',
        ]
        self.phase5f_timing_profile = {
            'enabled': self.phase5f_timing_enabled,
            'calls': 0,
            'timings_ms': {bucket: 0.0 for bucket in self.phase5f_timing_buckets},
            'event_counts': {bucket: 0 for bucket in self.phase5f_timing_buckets},
        }
        self._phase5f_event_buckets = None

        if self.content_patch_token_shape != (1, 1):
            assert not self.attn_content_with_mask, 'If there are more than one tokens in each embedding, the attn should not be controlled by the mask!'

        self.init_type = init_type
        self.apply(self._init_weights)
        self.fix_init_weight()
        if ckpt_path is not None:
            self.init_from_ckpt(path=ckpt_path)

        # reinitialize the codec, so that the pretrained model can be reloaded
        self.content_codec = instantiate_from_config(content_codec_config)

        self.loss_func = instantiate_from_config(loss_config)
        self.loss_mask_type = loss_mask_type

    def fix_init_weight(self):
        if self.init_type == 'beit':
            def rescale(param, layer_id):
                param.div_(math.sqrt(2.0 * layer_id))

            for layer_id, layer in enumerate(self.blocks):
                rescale(layer.attn.proj.weight.data, layer_id + 1)
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)
        elif self.init_type == 'mae':
            trunc_normal_(self.pos_emb)
        else:
            raise NotImplementedError('init type: {} not implemented!'.format(self.init_type))


    def _init_weights(self, module):
        if self.init_type == 'beit':
            if isinstance(module, (nn.Linear, nn.Embedding)):
                trunc_normal_(module.weight, std=0.02)
                module.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)       
            elif isinstance(module, nn.Conv2d):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        elif self.init_type == 'mae':
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
        else:
            raise NotImplementedError('init type: {} not implemented!'.format(self.init_type))



    def init_from_ckpt(self, path, ignore_keys=['content_codec.']):
        sd = torch.load(path, map_location="cpu")
        if 'model' in sd:
            sd = sd['model']
        else:
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("UQ-Transformer: Deleting key {} from the given state_dict.".format(k))
                    del sd[k]
        
        # interpolate positional embedding
        if tuple(sd['pos_emb'].shape) != tuple(self.pos_emb.shape):
            content_shape = (int(self.pos_emb.shape[1] ** 0.5), int(self.pos_emb.shape[1] ** 0.5))
            real_dim = self.pos_emb.shape[-1] // self.content_patch_seq_len

            provide_content_shape = (int(sd['pos_emb'].shape[1] ** 0.5), int(sd['pos_emb'].shape[1] ** 0.5))
            provide_content_patch_seq_len = sd['pos_emb'].shape[-1]//real_dim
            provide_content_token_patch_shape = (int(provide_content_patch_seq_len**0.5), int(provide_content_patch_seq_len**0.5))
            real_provide_content_shape = (provide_content_shape[0]*provide_content_token_patch_shape[0], provide_content_shape[1]*provide_content_token_patch_shape[1])
            pos_emb = sd['pos_emb'].permute(0, 2, 1).view(1, -1, provide_content_shape[0], provide_content_shape[1]) # 1 x H/cps*W/cps x D -> 1 x D x H/cps*W/cps -> 1 x D x H/cps x W/cps
            pos_emb = pixel_shuffle(pos_emb, out_size=real_provide_content_shape, chunked=True) # 1 x C x H x W

            real_content_shape = (content_shape[0]*self.content_patch_token_shape[0], content_shape[1]*self.content_patch_token_shape[1])
            pos_emb = F.interpolate(pos_emb, size=real_content_shape, mode='bilinear')
            
            pos_emb = pixel_unshuffle(pos_emb, out_size=content_shape, chunked=True) # 1 x D x H/cps x W/cps
            pos_emb = pos_emb.view(1, self.pos_emb.shape[-1], self.pos_emb.shape[1]).permute(0, 2, 1)
            sd['pos_emb'] = pos_emb
        
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print("UQ-Transformer: Load pretrained model from {}".format(path))
        print('UQ-Transformer: Missing keys in created model:\n', missing)
        print('UQ-Transformer: Unexpected keys in state dict:\n', unexpected)

    @property
    def device(self):
        return self.to_logits.weight.device

    def _boundary_split(self, mask, token_shape):
        if not self.boundary_split_enabled:
            return None

        timing_handle = self._phase5f_start_timing('boundary_split_time')
        split = self.boundary_detector(mask, token_shape=token_shape)
        self._phase5f_end_timing(timing_handle)
        profile = {
            'enabled': True,
            'ring_radius': self.boundary_detector.ring_radius,
            'safe_threshold': self.boundary_detector.safe_threshold,
            'calls': self.boundary_split_profile['calls'] + 1,
            'token_shape': [int(token_shape[0]), int(token_shape[1])],
            'total_tokens': int(split['protect_mask_token'].numel()),
            'valid_tokens': int(split['valid_token'].sum().detach().cpu()),
            'masked_or_partial_tokens': int(split['masked_or_partial_token'].sum().detach().cpu()),
            'boundary_tokens': int(split['boundary_token'].sum().detach().cpu()),
            'protect_tokens': int(split['protect_mask_token'].sum().detach().cpu()),
            'safe_candidate_tokens': int(split['safe_candidate_token'].sum().detach().cpu()),
        }
        self.boundary_split_profile = profile
        return split

    def _safe_stats(self, value_map, safe_mask):
        value_map = value_map.view(value_map.shape[0], -1)
        safe_mask = safe_mask.view(safe_mask.shape[0], -1)
        values = []
        for batch_idx in range(value_map.shape[0]):
            current = value_map[batch_idx, safe_mask[batch_idx]]
            if current.numel() > 0:
                values.append(current)

        if len(values) == 0:
            return {
                'mean': 0.0,
                'min': 0.0,
                'max': 0.0,
                'std': 0.0,
            }

        merged = torch.cat(values, dim=0)
        return {
            'mean': float(merged.mean().detach().cpu()),
            'min': float(merged.min().detach().cpu()),
            'max': float(merged.max().detach().cpu()),
            'std': float(merged.std(unbiased=False).detach().cpu()),
        }

    def _similarity_guidance_requires_quantized_feature(self):
        return self.similarity_guidance_source == 'codec_quantized_detached'

    def _to_similarity_guidance_tokens(self, feature_map, content_shape, detach=False):
        if feature_map is None:
            raise RuntimeError(
                f"similarity guidance source {self.similarity_guidance_source!r} "
                "requested a missing codec feature."
            )
        if detach:
            feature_map = feature_map.detach()
        if self.content_patch_token_shape != (1, 1):
            feature_map = pixel_unshuffle(feature_map, out_size=content_shape, chunked=True)
        return feature_map.permute(0, 2, 3, 1).contiguous()

    def _build_similarity_guidance_feature(self, data_mask, projected_feature, content_shape):
        timing_handle = self._phase5f_start_timing('codec_feature_prepare_time')
        if self.similarity_guidance_source == 'projected':
            if projected_feature is None:
                raise RuntimeError('projected similarity guidance requires projected transformer input features.')
            out = self._to_similarity_guidance_tokens(projected_feature, content_shape, detach=False)
            self._phase5f_end_timing(timing_handle)
            return out
        if self.similarity_guidance_source == 'codec_feature_detached':
            out = self._to_similarity_guidance_tokens(
                data_mask.get('feature'),
                content_shape,
                detach=True,
            )
            self._phase5f_end_timing(timing_handle)
            return out
        if self.similarity_guidance_source == 'codec_quantized_detached':
            out = self._to_similarity_guidance_tokens(
                data_mask.get('feature_quantize'),
                content_shape,
                detach=True,
            )
            self._phase5f_end_timing(timing_handle)
            return out
        self._phase5f_end_timing(timing_handle)
        raise RuntimeError(f'unknown similarity guidance source: {self.similarity_guidance_source!r}')

    @staticmethod
    def _normalize_valid_matrix(matrix, valid_mask, zero_threshold=1e-6):
        out = torch.zeros_like(matrix)
        if not bool(valid_mask.any().detach().cpu()):
            return out
        values = matrix[valid_mask]
        value_min = values.min()
        value_max = values.max()
        if float((value_max - value_min).detach().cpu()) < float(zero_threshold):
            out[valid_mask] = 0.0
        else:
            out[valid_mask] = (values - value_min) / (value_max - value_min)
        return out

    @staticmethod
    def _pairwise_cosine(left, right):
        left = F.normalize(left, dim=-1, eps=1e-6)
        right = F.normalize(right, dim=-1, eps=1e-6)
        return left @ right.transpose(-1, -2)

    @staticmethod
    def _pairwise_l2(left, right):
        return torch.cdist(left, right, p=2)

    @staticmethod
    def _avg_pool2d_replicate(feat, kernel_size):
        pad = int(kernel_size) // 2
        if pad > 0:
            feat = F.pad(feat, (pad, pad, pad, pad), mode='replicate')
        return F.avg_pool2d(
            feat,
            kernel_size=int(kernel_size),
            stride=1,
            padding=0,
            count_include_pad=False,
        )

    @staticmethod
    def _avg_pool2d_zero(feat, kernel_size):
        pad = int(kernel_size) // 2
        return F.avg_pool2d(
            feat,
            kernel_size=int(kernel_size),
            stride=1,
            padding=pad,
            count_include_pad=False,
        )

    def _avg_pool2d_with_mode(self, feat, kernel_size, pad_mode):
        if pad_mode == 'replicate':
            return self._avg_pool2d_replicate(feat, kernel_size=kernel_size)
        if pad_mode == 'zero':
            return self._avg_pool2d_zero(feat, kernel_size=kernel_size)
        raise RuntimeError(f'unknown GA-SPG-Lite pad mode: {pad_mode!r}')

    @staticmethod
    def _token_coord_grid(batch_size, height, width, device, dtype):
        yy, xx = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing='ij',
        )
        return torch.stack([yy, xx], dim=-1).unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()

    @staticmethod
    def _image_border_distance_from_coords(token_coords, height, width):
        y_coord = token_coords[..., 0]
        x_coord = token_coords[..., 1]
        return torch.minimum(
            torch.minimum(y_coord, x_coord),
            torch.minimum(
                float(height - 1) - y_coord,
                float(width - 1) - x_coord,
            ),
        )

    def _compact_ga_spg_guidance_debug(self, guidance):
        if guidance is None:
            return None
        keys = (
            'distance_raw',
            'distance_score',
            'local_variance_risk',
            'boundary_risk',
            'smoothness_risk',
            'scalar_risk',
            'image_border_risk',
            'image_border_distance',
        )
        out = {
            'mode': guidance.get('mode', self.safe_tome_score_mode),
            'pad_mode': self.ga_spg_lite_pad_mode if self.ga_spg_lite_mode else 'n/a',
            'explicit_image_border_risk': bool(self.ga_spg_lite_mode and self.ga_spg_lite_explicit_border_risk),
        }
        for key in keys:
            tensor = guidance.get(key)
            if tensor is not None:
                out[key] = tensor.detach().cpu()
        return out

    def _ga_spg_guidance(self, emb, boundary_split):
        feat = emb.detach().permute(0, 3, 1, 2).contiguous()
        safe = boundary_split['safe_candidate_token'].bool()
        protect = boundary_split['protect_mask_token'].bool()
        b, _, h, w = feat.shape

        feat_sq = feat.pow(2)
        avg3_handle = self._phase5f_start_timing('avgpool_3x3_time')
        branch_3x3 = self._avg_pool2d_replicate(feat, kernel_size=3)
        local_variance = (
            self._avg_pool2d_replicate(feat_sq, kernel_size=3)
            - branch_3x3.pow(2)
        ).mean(dim=1, keepdim=True).clamp(min=0.0)
        self._phase5f_end_timing(avg3_handle)
        avg9_handle = self._phase5f_start_timing('avgpool_9x9_time')
        branch_9x9 = self._avg_pool2d_replicate(feat, kernel_size=9)
        context_variance = (
            self._avg_pool2d_replicate(feat_sq, kernel_size=9)
            - branch_9x9.pow(2)
        ).mean(dim=1, keepdim=True).clamp(min=0.0)
        self._phase5f_end_timing(avg9_handle)
        distance_raw, distance_score = self.similarity_scorer._distance_rule_score(protect, safe)
        local_variance_norm = self.similarity_scorer._normalize_safe_map(
            local_variance * safe.float(),
            safe,
            zero_threshold=1e-6,
        )
        context_variance_norm = self.similarity_scorer._normalize_safe_map(
            context_variance * safe.float(),
            safe,
            zero_threshold=1e-6,
        )
        boundary_risk_raw = torch.where(
            safe,
            1.0 / (distance_raw.clamp(min=0.0) + 1.0e-6),
            torch.zeros_like(distance_raw),
        )
        boundary_risk_norm = self.similarity_scorer._normalize_safe_map(
            boundary_risk_raw * safe.float(),
            safe,
            zero_threshold=1e-6,
        )
        yy, xx = torch.meshgrid(
            torch.arange(h, device=feat.device, dtype=feat.dtype),
            torch.arange(w, device=feat.device, dtype=feat.dtype),
            indexing='ij',
        )
        token_coords = torch.stack([yy, xx], dim=-1).unsqueeze(0).expand(b, -1, -1, -1).contiguous()

        distance_stats = self._safe_stats(distance_raw, safe)
        local_stats = self._safe_stats(local_variance_norm, safe)
        context_stats = self._safe_stats(context_variance_norm, safe)
        boundary_stats = self._safe_stats(boundary_risk_norm, safe)
        self.ga_spg_profile = {
            'enabled': True,
            'mode': self.safe_tome_score_mode if self.ga_spg_mode else 'audit_only',
            'guidance_source': self.similarity_guidance_source,
            'optimized': bool(self.ga_spg_lite_mode and self.ga_spg_lite_optimized),
            'lambda_risk': self.ga_spg_lambda_risk,
            'weights': dict(self.ga_spg_weights),
            'hard_risk_threshold': self.ga_spg_hard_risk_threshold,
            'hard_penalty': self.ga_spg_hard_penalty,
            'calls': self.ga_spg_profile['calls'] + 1,
            'safe_tokens': int(safe.sum().detach().cpu()),
            'protect_tokens': int(protect.sum().detach().cpu()),
            'distance_mean': distance_stats['mean'],
            'distance_min': distance_stats['min'],
            'distance_max': distance_stats['max'],
            'distance_std': distance_stats['std'],
            'local_variance_mean': local_stats['mean'],
            'local_variance_min': local_stats['min'],
            'local_variance_max': local_stats['max'],
            'local_variance_std': local_stats['std'],
            'context_variance_mean': context_stats['mean'],
            'context_variance_min': context_stats['min'],
            'context_variance_max': context_stats['max'],
            'context_variance_std': context_stats['std'],
            'boundary_risk_mean': boundary_stats['mean'],
            'boundary_risk_min': boundary_stats['min'],
            'boundary_risk_max': boundary_stats['max'],
            'boundary_risk_std': boundary_stats['std'],
            'smoothness_risk_mean': 0.0,
            'smoothness_risk_min': 0.0,
            'smoothness_risk_max': 0.0,
            'smoothness_risk_std': 0.0,
            'scalar_risk_mean': 0.0,
            'scalar_risk_min': 0.0,
            'scalar_risk_max': 0.0,
            'scalar_risk_std': 0.0,
        }

        return {
            'mode': self.safe_tome_score_mode,
            'distance_raw': distance_raw,
            'distance_score': distance_score,
            'raw_tokens': emb,
            'branch_3x3': branch_3x3.permute(0, 2, 3, 1).contiguous(),
            'branch_9x9': branch_9x9.permute(0, 2, 3, 1).contiguous(),
            'local_variance_risk': local_variance_norm,
            'context_variance_risk': context_variance_norm,
            'boundary_risk': boundary_risk_norm,
            'token_coords': token_coords,
            'distance_raw_flat': distance_raw.view(b, -1),
            'token_coords_flat': token_coords.view(b, -1, 2),
        }

    def _ga_spg_lite_guidance(self, emb, boundary_split):
        risk_compute_handle = self._phase5f_start_timing('token_risk_compute_time')
        feat = emb.detach().permute(0, 3, 1, 2).contiguous()
        safe = boundary_split['safe_candidate_token'].bool()
        protect = boundary_split['protect_mask_token'].bool()
        b, _, h, w = feat.shape
        feat_sq = feat.pow(2)

        avg3_handle = self._phase5f_start_timing('avgpool_3x3_time')
        branch_3x3 = self._avg_pool2d_with_mode(
            feat,
            kernel_size=3,
            pad_mode=self.ga_spg_lite_pad_mode,
        )
        local_variance = (
            self._avg_pool2d_with_mode(
                feat_sq,
                kernel_size=3,
                pad_mode=self.ga_spg_lite_pad_mode,
            )
            - branch_3x3.pow(2)
        ).mean(dim=1, keepdim=True).clamp(min=0.0)
        feat_norm = F.normalize(feat, dim=1, eps=1.0e-6)
        branch_norm = F.normalize(branch_3x3, dim=1, eps=1.0e-6)
        smoothness_risk_raw = (1.0 - (feat_norm * branch_norm).sum(dim=1, keepdim=True)).clamp(min=0.0)
        self._phase5f_end_timing(avg3_handle)

        distance_handle = self._phase5f_start_timing('distance_map_time')
        distance_raw, distance_score = self.similarity_scorer._distance_rule_score(protect, safe)
        self._phase5f_end_timing(distance_handle)
        local_variance_norm = self.similarity_scorer._normalize_safe_map(
            local_variance * safe.float(),
            safe,
            zero_threshold=1e-6,
        )
        smoothness_risk_norm = self.similarity_scorer._normalize_safe_map(
            smoothness_risk_raw * safe.float(),
            safe,
            zero_threshold=1e-6,
        )
        boundary_risk_raw = torch.where(
            safe,
            1.0 / (distance_raw.clamp(min=0.0) + 1.0e-6),
            torch.zeros_like(distance_raw),
        )
        boundary_risk_norm = self.similarity_scorer._normalize_safe_map(
            boundary_risk_raw * safe.float(),
            safe,
            zero_threshold=1e-6,
        )
        token_coords = self._token_coord_grid(
            batch_size=b,
            height=h,
            width=w,
            device=feat.device,
            dtype=feat.dtype,
        )
        image_border_distance = self._image_border_distance_from_coords(token_coords, h, w).unsqueeze(1)
        if self.ga_spg_lite_explicit_border_risk:
            image_border_risk_raw = torch.where(
                safe,
                1.0 / (image_border_distance.clamp(min=0.0) + 1.0e-6),
                torch.zeros_like(distance_raw),
            )
            image_border_risk_norm = self.similarity_scorer._normalize_safe_map(
                image_border_risk_raw * safe.float(),
                safe,
                zero_threshold=1e-6,
            )
        else:
            image_border_risk_norm = torch.zeros_like(distance_raw)
        scalar_risk = (
            self.ga_spg_lite_weights['texture_risk'] * local_variance_norm
            + self.ga_spg_lite_weights['boundary_risk'] * boundary_risk_norm
            + self.ga_spg_lite_weights['smoothness_risk'] * smoothness_risk_norm
        )
        if self.ga_spg_lite_explicit_border_risk:
            scalar_risk = scalar_risk + self.ga_spg_lite_image_border_weight * image_border_risk_norm
        scalar_risk = self.similarity_scorer._normalize_safe_map(
            scalar_risk * safe.float(),
            safe,
            zero_threshold=1e-6,
        )
        self._phase5f_end_timing(risk_compute_handle)

        if self.phase5k_lean_profile_enabled and not self.safe_tome_debug_enabled and not self.ga_spg_audit_enabled:
            empty_stats = {'mean': 0.0, 'min': 0.0, 'max': 0.0, 'std': 0.0}
            distance_stats = empty_stats
            local_stats = empty_stats
            boundary_stats = empty_stats
            image_border_stats = empty_stats
            smoothness_stats = empty_stats
            scalar_stats = empty_stats
            safe_tokens = int(safe.numel() - protect.sum().detach().cpu()) if self.ga_spg_audit_enabled else 0
            protect_tokens = 0
        else:
            distance_stats = self._safe_stats(distance_raw, safe)
            local_stats = self._safe_stats(local_variance_norm, safe)
            boundary_stats = self._safe_stats(boundary_risk_norm, safe)
            image_border_stats = self._safe_stats(image_border_risk_norm, safe)
            smoothness_stats = self._safe_stats(smoothness_risk_norm, safe)
            scalar_stats = self._safe_stats(scalar_risk, safe)
            safe_tokens = int(safe.sum().detach().cpu())
            protect_tokens = int(protect.sum().detach().cpu())
        self.ga_spg_profile = {
            'enabled': True,
            'mode': self.safe_tome_score_mode,
            'guidance_source': self.similarity_guidance_source,
            'optimized': bool(self.ga_spg_lite_optimized),
            'runtime_cache_enabled': bool(self.ga_spg_lite_optimized and self.ga_spg_lite_runtime_cache_enabled),
            'pad_mode': self.ga_spg_lite_pad_mode,
            'explicit_image_border_risk': bool(self.ga_spg_lite_explicit_border_risk),
            'lambda_risk': self.ga_spg_lite_lambda_risk,
            'weights': dict(self.ga_spg_lite_weights),
            'hard_risk_threshold': self.ga_spg_hard_risk_threshold,
            'hard_penalty': self.ga_spg_hard_penalty,
            'calls': self.ga_spg_profile['calls'] + 1,
            'safe_tokens': int(safe_tokens),
            'protect_tokens': int(protect_tokens),
            'distance_mean': distance_stats['mean'],
            'distance_min': distance_stats['min'],
            'distance_max': distance_stats['max'],
            'distance_std': distance_stats['std'],
            'local_variance_mean': local_stats['mean'],
            'local_variance_min': local_stats['min'],
            'local_variance_max': local_stats['max'],
            'local_variance_std': local_stats['std'],
            'context_variance_mean': 0.0,
            'context_variance_min': 0.0,
            'context_variance_max': 0.0,
            'context_variance_std': 0.0,
            'boundary_risk_mean': boundary_stats['mean'],
            'boundary_risk_min': boundary_stats['min'],
            'boundary_risk_max': boundary_stats['max'],
            'boundary_risk_std': boundary_stats['std'],
            'image_border_risk_mean': image_border_stats['mean'],
            'image_border_risk_min': image_border_stats['min'],
            'image_border_risk_max': image_border_stats['max'],
            'image_border_risk_std': image_border_stats['std'],
            'smoothness_risk_mean': smoothness_stats['mean'],
            'smoothness_risk_min': smoothness_stats['min'],
            'smoothness_risk_max': smoothness_stats['max'],
            'smoothness_risk_std': smoothness_stats['std'],
            'scalar_risk_mean': scalar_stats['mean'],
            'scalar_risk_min': scalar_stats['min'],
            'scalar_risk_max': scalar_stats['max'],
            'scalar_risk_std': scalar_stats['std'],
        }
        return {
            'mode': self.safe_tome_score_mode,
            'distance_raw': distance_raw,
            'distance_score': distance_score,
            'raw_tokens': emb,
            'branch_3x3': branch_3x3.permute(0, 2, 3, 1).contiguous(),
            'local_variance_risk': local_variance_norm,
            'boundary_risk': boundary_risk_norm,
            'smoothness_risk': smoothness_risk_norm,
            'scalar_risk': scalar_risk,
            'image_border_risk': image_border_risk_norm,
            'image_border_distance': image_border_distance,
            'token_coords': token_coords,
            'distance_raw_flat': distance_raw.view(b, -1),
            'scalar_risk_flat': scalar_risk.view(b, -1),
            'local_variance_risk_flat': local_variance_norm.view(b, -1),
            'smoothness_risk_flat': smoothness_risk_norm.view(b, -1),
            'token_coords_flat': token_coords.view(b, -1, 2),
        }

    def _similarity_guidance(self, emb, boundary_split):
        if not self.similarity_scorer_enabled or boundary_split is None:
            return None

        timing_handle = self._phase4a_start_timing('Time_Scoring')
        guidance = self.similarity_scorer(
            emb,
            protect_mask_token=boundary_split['protect_mask_token'],
            safe_candidate_token=boundary_split['safe_candidate_token'],
        )
        self._phase4a_end_timing(timing_handle)
        safe_mask = boundary_split['safe_candidate_token'].bool()
        score_stats = self._safe_stats(guidance['redundancy_score_norm'], safe_mask)
        distance_stats = self._safe_stats(guidance['distance_score'], safe_mask)
        cos_stats = self._safe_stats(guidance['mean_cos_sim'], safe_mask)
        var_stats = self._safe_stats(guidance['local_variance'], safe_mask)
        var_norm_stats = self._safe_stats(guidance['local_variance_norm'], safe_mask)

        profile = {
            'enabled': True,
            'learnable': self.similarity_scorer.learnable,
            'guidance_source': self.similarity_guidance_source,
            'guidance_detached': self.similarity_guidance_source != 'projected',
            'kernel_size': self.similarity_scorer.kernel_size,
            'alpha': self.similarity_scorer.alpha,
            'beta': self.similarity_scorer.beta,
            'topk_ratio': self.similarity_scorer.topk_ratio,
            'calls': self.similarity_profile['calls'] + 1,
            'token_shape': [int(emb.shape[1]), int(emb.shape[2])],
            'safe_tokens': int(safe_mask.sum().detach().cpu()),
            'protect_tokens': int(boundary_split['protect_mask_token'].sum().detach().cpu()),
            'score_mean': score_stats['mean'],
            'score_min': score_stats['min'],
            'score_max': score_stats['max'],
            'score_std': score_stats['std'],
            'distance_mean': distance_stats['mean'],
            'distance_min': distance_stats['min'],
            'distance_max': distance_stats['max'],
            'distance_std': distance_stats['std'],
            'mean_cos_sim': cos_stats['mean'],
            'mean_cos_sim_min': cos_stats['min'],
            'mean_cos_sim_max': cos_stats['max'],
            'mean_cos_sim_std': cos_stats['std'],
            'local_variance_mean': var_stats['mean'],
            'local_variance_min': var_stats['min'],
            'local_variance_max': var_stats['max'],
            'local_variance_std': var_stats['std'],
            'local_variance_norm_mean': var_norm_stats['mean'],
            'local_variance_norm_min': var_norm_stats['min'],
            'local_variance_norm_max': var_norm_stats['max'],
            'local_variance_norm_std': var_norm_stats['std'],
            'topk_overlap_ratio': float(guidance['topk_overlap_ratio'].mean().detach().cpu()),
        }
        self.similarity_profile = profile
        return guidance

    def _compute_safe_tome_guidance(self, emb, boundary_split):
        if boundary_split is None:
            return None
        if self.ga_spg_guidance_enabled:
            timing_handle = self._phase4a_start_timing('Time_Scoring')
            if self.ga_spg_lite_mode:
                guidance = self._ga_spg_lite_guidance(emb, boundary_split)
            else:
                guidance = self._ga_spg_guidance(emb, boundary_split)
            self._phase4a_end_timing(timing_handle)
            return guidance
        if self.similarity_scorer_enabled:
            return self._similarity_guidance(emb, boundary_split)
        return None

    def _phase4a_should_time(self):
        return self.phase4a_timing_enabled and self.device.type == 'cuda'

    def _phase4a_reset_timing(self):
        if not self._phase4a_should_time():
            self._phase4a_event_buckets = None
            self._phase4a_num_sampling_steps = 0
            return
        self._phase4a_event_buckets = {bucket: [] for bucket in self.phase4a_timing_buckets}
        self._phase4a_num_sampling_steps = 0

    def _phase4a_start_timing(self, bucket):
        if not self._phase4a_should_time() or self._phase4a_event_buckets is None:
            return None
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        return bucket, start, end

    def _phase4a_end_timing(self, handle):
        if handle is None or self._phase4a_event_buckets is None:
            return
        bucket, start, end = handle
        end.record()
        self._phase4a_event_buckets[bucket].append((start, end))

    def _phase4a_finalize_timing(self, token_profile=None):
        if not self._phase4a_should_time() or self._phase4a_event_buckets is None:
            return None
        torch.cuda.synchronize(self.device)
        timings_ms = {}
        event_counts = {}
        for bucket, events in self._phase4a_event_buckets.items():
            timings_ms[bucket] = float(sum(start.elapsed_time(end) for start, end in events))
            event_counts[bucket] = int(len(events))
        profile = {
            'enabled': True,
            'calls': self.phase4a_timing_profile['calls'] + 1,
            'num_sampling_steps': int(self._phase4a_num_sampling_steps),
            'timings_ms': timings_ms,
            'event_counts': event_counts,
            'token_profile': token_profile or {
                'original_tokens': 0,
                'eligible_tokens': 0,
                'removed_tokens': 0,
                'merged_tokens': 0,
                'restored_tokens': 0,
            },
        }
        self.phase4a_timing_profile = profile
        self._phase4a_event_buckets = None
        self._phase4a_num_sampling_steps = 0
        return profile

    def _phase5f_should_time(self):
        return self.phase5f_timing_enabled and self.device.type == 'cuda'

    def _phase5f_reset_timing(self):
        if not self._phase5f_should_time():
            self._phase5f_event_buckets = None
            return
        self._phase5f_event_buckets = {bucket: [] for bucket in self.phase5f_timing_buckets}

    def _phase5f_start_timing(self, bucket):
        if not self._phase5f_should_time() or self._phase5f_event_buckets is None:
            return None
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        return bucket, start, end

    def _phase5f_end_timing(self, handle):
        if handle is None or self._phase5f_event_buckets is None:
            return
        bucket, start, end = handle
        end.record()
        self._phase5f_event_buckets[bucket].append((start, end))

    def _phase5f_finalize_timing(self):
        if not self._phase5f_should_time() or self._phase5f_event_buckets is None:
            return None
        torch.cuda.synchronize(self.device)
        timings_ms = {}
        event_counts = {}
        for bucket, events in self._phase5f_event_buckets.items():
            timings_ms[bucket] = float(sum(start.elapsed_time(end) for start, end in events))
            event_counts[bucket] = int(len(events))
        profile = {
            'enabled': True,
            'calls': self.phase5f_timing_profile['calls'] + 1,
            'timings_ms': timings_ms,
            'event_counts': event_counts,
        }
        self.phase5f_timing_profile = profile
        self._phase5f_event_buckets = None
        return profile

    def _safe_tome_selected_count(self, safe_count):
        if self.safe_tome_r <= 0:
            return 0
        return int(min(int(safe_count), max(self.safe_tome_r * 2, 2)))

    @staticmethod
    def _flat_stats(values):
        if values.numel() == 0:
            return {
                'mean': 0.0,
                'min': 0.0,
                'max': 0.0,
                'std': 0.0,
            }
        return {
            'mean': float(values.mean().detach().cpu()),
            'min': float(values.min().detach().cpu()),
            'max': float(values.max().detach().cpu()),
            'std': float(values.std(unbiased=False).detach().cpu()),
        }

    def _build_basic_pair_debug(self, eligible_seq, selection_plan, boundary_split, batch_idx):
        base_metric = self._pairwise_cosine(eligible_seq[::2], eligible_seq[1::2])
        src_distance = boundary_split['distance_raw_flat'][batch_idx].index_select(0, selection_plan['even_idx']).float()
        dst_distance = boundary_split['distance_raw_flat'][batch_idx].index_select(0, selection_plan['odd_idx']).float()
        pair_boundary_distance_min = torch.minimum(src_distance.unsqueeze(1), dst_distance.unsqueeze(0))
        coords = boundary_split['token_coords_flat'][batch_idx]
        src_coords = coords.index_select(0, selection_plan['even_idx']).float()
        dst_coords = coords.index_select(0, selection_plan['odd_idx']).float()
        pair_spatial_distance = self._pairwise_l2(src_coords, dst_coords)
        return {
            'base_metric': base_metric,
            'risk_penalty': torch.zeros_like(base_metric),
            'adjusted_metric': base_metric,
            'pair_cos_raw': base_metric,
            'pair_cos_3x3': base_metric,
            'pair_cos_9x9': base_metric,
            'pair_feature_l2_raw': self._pairwise_l2(eligible_seq[::2], eligible_seq[1::2]),
            'pair_feature_l2_3x3': torch.zeros_like(base_metric),
            'pair_feature_l2_9x9': torch.zeros_like(base_metric),
            'pair_local_variance_risk': torch.zeros_like(base_metric),
            'pair_context_variance_risk': torch.zeros_like(base_metric),
            'pair_boundary_distance_min': pair_boundary_distance_min,
            'pair_spatial_distance': pair_spatial_distance,
            'pair_boundary_distance_src': src_distance,
            'pair_boundary_distance_dst': dst_distance,
            'pair_context_risk_src': torch.zeros_like(src_distance),
            'pair_context_risk_dst': torch.zeros_like(dst_distance),
            'pair_local_risk_src': torch.zeros_like(src_distance),
            'pair_local_risk_dst': torch.zeros_like(dst_distance),
            'invalid_pair_mask': torch.zeros_like(base_metric, dtype=torch.bool),
            'high_risk_mask': torch.zeros_like(base_metric, dtype=torch.bool),
            'feasible_pair_count': int(base_metric.numel()),
            'invalid_pair_count': 0,
            'high_risk_pair_count': 0,
            'under_compression': False,
            'fallback_used': False,
            'fallback_reason': '',
        }

    def _build_ga_spg_pair_metric(self, eligible_seq, selection_plan, guidance, boundary_split, batch_idx):
        raw_tokens = guidance['raw_tokens'][batch_idx].view(-1, guidance['raw_tokens'].shape[-1])
        branch_3x3 = guidance['branch_3x3'][batch_idx].view(-1, guidance['branch_3x3'].shape[-1])
        branch_9x9 = guidance['branch_9x9'][batch_idx].view(-1, guidance['branch_9x9'].shape[-1])
        local_risk = guidance['local_variance_risk'][batch_idx].view(-1)
        context_risk = guidance['context_variance_risk'][batch_idx].view(-1)
        distance_raw = guidance['distance_raw'][batch_idx].view(-1)
        coords = guidance['token_coords'][batch_idx].view(-1, 2)

        src_idx = selection_plan['even_idx']
        dst_idx = selection_plan['odd_idx']
        src_proj = eligible_seq[::2]
        dst_proj = eligible_seq[1::2]
        src_raw = raw_tokens.index_select(0, src_idx)
        dst_raw = raw_tokens.index_select(0, dst_idx)
        src_3x3 = branch_3x3.index_select(0, src_idx)
        dst_3x3 = branch_3x3.index_select(0, dst_idx)
        src_9x9 = branch_9x9.index_select(0, src_idx)
        dst_9x9 = branch_9x9.index_select(0, dst_idx)
        src_context = context_risk.index_select(0, src_idx)
        dst_context = context_risk.index_select(0, dst_idx)
        src_local = local_risk.index_select(0, src_idx)
        dst_local = local_risk.index_select(0, dst_idx)
        src_distance = distance_raw.index_select(0, src_idx).float()
        dst_distance = distance_raw.index_select(0, dst_idx).float()
        src_coords = coords.index_select(0, src_idx).float()
        dst_coords = coords.index_select(0, dst_idx).float()

        base_handle = self._phase5f_start_timing('pair_metric_base_time')
        base_metric = self._pairwise_cosine(src_proj, dst_proj)
        pair_cos_raw = self._pairwise_cosine(src_raw, dst_raw)
        pair_cos_3x3 = self._pairwise_cosine(src_3x3, dst_3x3)
        pair_cos_9x9 = self._pairwise_cosine(src_9x9, dst_9x9)
        pair_feature_l2_raw = self._pairwise_l2(src_raw, dst_raw)
        pair_feature_l2_3x3 = self._pairwise_l2(src_3x3, dst_3x3)
        pair_feature_l2_9x9 = self._pairwise_l2(src_9x9, dst_9x9)
        pair_boundary_distance_min = torch.minimum(src_distance.unsqueeze(1), dst_distance.unsqueeze(0))
        pair_spatial_distance = self._pairwise_l2(src_coords, dst_coords)
        self._phase5f_end_timing(base_handle)

        risk_handle = self._phase5f_start_timing('risk_penalty_matrix_time')
        pair_local_raw = 0.5 * (src_local.unsqueeze(1) + dst_local.unsqueeze(0))
        pair_context_raw = 0.5 * (src_context.unsqueeze(1) + dst_context.unsqueeze(0))
        valid_pair_mask = torch.ones_like(base_metric, dtype=torch.bool)

        cos_3x3_risk = self._normalize_valid_matrix(1.0 - pair_cos_3x3, valid_pair_mask, zero_threshold=1e-6)
        cos_9x9_risk = self._normalize_valid_matrix(1.0 - pair_cos_9x9, valid_pair_mask, zero_threshold=1e-6)
        pair_local_risk = self._normalize_valid_matrix(pair_local_raw, valid_pair_mask, zero_threshold=1e-6)
        pair_context_risk = self._normalize_valid_matrix(pair_context_raw, valid_pair_mask, zero_threshold=1e-6)
        boundary_risk_raw = 1.0 / (pair_boundary_distance_min.clamp(min=0.0) + 1.0e-6)
        boundary_risk = self._normalize_valid_matrix(boundary_risk_raw, valid_pair_mask, zero_threshold=1e-6)
        risk_penalty = (
            self.ga_spg_weights['cos_3x3'] * cos_3x3_risk
            + self.ga_spg_weights['cos_9x9'] * cos_9x9_risk
            + self.ga_spg_weights['context_variance'] * pair_context_risk
            + self.ga_spg_weights['boundary_risk'] * boundary_risk
        )
        risk_penalty = self._normalize_valid_matrix(risk_penalty, valid_pair_mask, zero_threshold=1e-6)
        self._phase5f_end_timing(risk_handle)

        adjust_handle = self._phase5f_start_timing('adjusted_metric_time')
        invalid_pair_mask = torch.zeros_like(valid_pair_mask)
        high_risk_mask = risk_penalty >= self.ga_spg_hard_risk_threshold
        if self.safe_tome_score_mode == 'ga_spg_hardveto':
            feasible_mask = valid_pair_mask & (~high_risk_mask)
            invalid_pair_mask = ~feasible_mask
            adjusted_metric = base_metric.clone()
            safe_min = torch.finfo(adjusted_metric.dtype).min / 2.0
            adjusted_metric.masked_fill_(invalid_pair_mask, safe_min)
            feasible_pair_count = int(feasible_mask.any(dim=-1).sum().detach().cpu())
            under_compression = feasible_pair_count < min(int(self.safe_tome_r), int(base_metric.shape[-1]))
        else:
            adjusted_metric = base_metric - self.ga_spg_lambda_risk * risk_penalty
            feasible_pair_count = int(valid_pair_mask.any(dim=-1).sum().detach().cpu())
            under_compression = False
        self._phase5f_end_timing(adjust_handle)

        return {
            'base_metric': base_metric,
            'risk_penalty': risk_penalty,
            'adjusted_metric': adjusted_metric,
            'pair_cos_raw': pair_cos_raw,
            'pair_cos_3x3': pair_cos_3x3,
            'pair_cos_9x9': pair_cos_9x9,
            'pair_feature_l2_raw': pair_feature_l2_raw,
            'pair_feature_l2_3x3': pair_feature_l2_3x3,
            'pair_feature_l2_9x9': pair_feature_l2_9x9,
            'pair_local_variance_risk': pair_local_risk,
            'pair_context_variance_risk': pair_context_risk,
            'pair_boundary_distance_min': pair_boundary_distance_min,
            'pair_spatial_distance': pair_spatial_distance,
            'pair_boundary_distance_src': src_distance,
            'pair_boundary_distance_dst': dst_distance,
            'pair_context_risk_src': src_context,
            'pair_context_risk_dst': dst_context,
            'pair_local_risk_src': src_local,
            'pair_local_risk_dst': dst_local,
            'invalid_pair_mask': invalid_pair_mask,
            'high_risk_mask': high_risk_mask,
            'feasible_pair_count': feasible_pair_count,
            'invalid_pair_count': int(invalid_pair_mask.sum().detach().cpu()),
            'high_risk_pair_count': int(high_risk_mask.sum().detach().cpu()),
            'under_compression': under_compression,
            'fallback_used': False,
            'fallback_reason': '',
        }

    def _build_ga_spg_lite_adjusted_metric_fast(self, eligible_seq, runtime_plan):
        src_proj = eligible_seq[::2]
        dst_proj = eligible_seq[1::2]

        base_handle = self._phase5f_start_timing('pair_metric_base_time')
        base_metric = self._pairwise_cosine(src_proj, dst_proj)
        self._phase5f_end_timing(base_handle)

        risk_handle = self._phase5f_start_timing('risk_penalty_matrix_time')
        self._phase5f_end_timing(risk_handle)

        adjust_handle = self._phase5f_start_timing('adjusted_metric_time')
        base_metric.sub_(runtime_plan['src_penalty'].to(device=base_metric.device, dtype=base_metric.dtype).unsqueeze(1))
        base_metric.sub_(runtime_plan['dst_penalty'].to(device=base_metric.device, dtype=base_metric.dtype).unsqueeze(0))
        self._phase5f_end_timing(adjust_handle)
        return base_metric

    def _build_ga_spg_lite_pair_metric(
        self,
        eligible_seq,
        selection_plan,
        guidance,
        boundary_split,
        batch_idx,
        runtime_plan=None,
        use_fast_adjustment=False,
    ):
        if runtime_plan is not None:
            src_scalar = runtime_plan['src_scalar']
            dst_scalar = runtime_plan['dst_scalar']
            src_local = runtime_plan['src_local']
            dst_local = runtime_plan['dst_local']
            src_smooth = runtime_plan['src_smooth']
            dst_smooth = runtime_plan['dst_smooth']
            src_distance = runtime_plan['src_distance']
            dst_distance = runtime_plan['dst_distance']
            src_coords = runtime_plan['src_coords']
            dst_coords = runtime_plan['dst_coords']
        else:
            scalar_risk = guidance['scalar_risk'][batch_idx].view(-1)
            local_risk = guidance['local_variance_risk'][batch_idx].view(-1)
            smoothness_risk = guidance['smoothness_risk'][batch_idx].view(-1)
            distance_raw = guidance['distance_raw'][batch_idx].view(-1)
            coords = guidance['token_coords'][batch_idx].view(-1, 2)

            src_idx = selection_plan['even_idx']
            dst_idx = selection_plan['odd_idx']
            src_scalar = scalar_risk.index_select(0, src_idx)
            dst_scalar = scalar_risk.index_select(0, dst_idx)
            src_local = local_risk.index_select(0, src_idx)
            dst_local = local_risk.index_select(0, dst_idx)
            src_smooth = smoothness_risk.index_select(0, src_idx)
            dst_smooth = smoothness_risk.index_select(0, dst_idx)
            src_distance = distance_raw.index_select(0, src_idx).float()
            dst_distance = distance_raw.index_select(0, dst_idx).float()
            src_coords = coords.index_select(0, src_idx).float()
            dst_coords = coords.index_select(0, dst_idx).float()

        src_proj = eligible_seq[::2]
        dst_proj = eligible_seq[1::2]
        base_handle = self._phase5f_start_timing('pair_metric_base_time')
        base_metric = self._pairwise_cosine(src_proj, dst_proj)
        pair_boundary_distance_min = torch.minimum(src_distance.unsqueeze(1), dst_distance.unsqueeze(0))
        pair_spatial_distance = self._pairwise_l2(src_coords, dst_coords)
        self._phase5f_end_timing(base_handle)

        risk_handle = self._phase5f_start_timing('risk_penalty_matrix_time')
        risk_penalty = src_scalar.unsqueeze(1) + dst_scalar.unsqueeze(0)
        self._phase5f_end_timing(risk_handle)

        adjust_handle = self._phase5f_start_timing('adjusted_metric_time')
        adjusted_metric = base_metric.clone()
        if runtime_plan is not None and use_fast_adjustment:
            adjusted_metric.sub_(
                runtime_plan['src_penalty'].to(
                    device=adjusted_metric.device,
                    dtype=adjusted_metric.dtype,
                ).unsqueeze(1)
            )
            adjusted_metric.sub_(
                runtime_plan['dst_penalty'].to(
                    device=adjusted_metric.device,
                    dtype=adjusted_metric.dtype,
                ).unsqueeze(0)
            )
        else:
            adjusted_metric.sub_(self.ga_spg_lite_lambda_risk * risk_penalty)
        self._phase5f_end_timing(adjust_handle)

        zero_matrix = torch.zeros_like(base_metric)
        pair_local_risk = 0.5 * (src_local.unsqueeze(1) + dst_local.unsqueeze(0))
        pair_smoothness_risk = 0.5 * (src_smooth.unsqueeze(1) + dst_smooth.unsqueeze(0))
        return {
            'base_metric': base_metric,
            'risk_penalty': risk_penalty,
            'adjusted_metric': adjusted_metric,
            'pair_cos_raw': base_metric,
            'pair_cos_3x3': zero_matrix,
            'pair_cos_9x9': zero_matrix,
            'pair_feature_l2_raw': zero_matrix,
            'pair_feature_l2_3x3': zero_matrix,
            'pair_feature_l2_9x9': zero_matrix,
            'pair_local_variance_risk': pair_local_risk,
            'pair_context_variance_risk': pair_smoothness_risk,
            'pair_boundary_distance_min': pair_boundary_distance_min,
            'pair_spatial_distance': pair_spatial_distance,
            'pair_boundary_distance_src': src_distance,
            'pair_boundary_distance_dst': dst_distance,
            'pair_context_risk_src': src_smooth,
            'pair_context_risk_dst': dst_smooth,
            'pair_local_risk_src': src_local,
            'pair_local_risk_dst': dst_local,
            'invalid_pair_mask': torch.zeros_like(base_metric, dtype=torch.bool),
            'high_risk_mask': torch.zeros_like(base_metric, dtype=torch.bool),
            'feasible_pair_count': int(base_metric.shape[-2]),
            'invalid_pair_count': 0,
            'high_risk_pair_count': 0,
            'under_compression': False,
            'fallback_used': False,
            'fallback_reason': '',
        }

    def _build_safe_tome_state(self, emb, boundary_split, similarity_guidance):
        if not self.safe_tome_enabled:
            return None
        if boundary_split is None:
            raise RuntimeError('PUT_SAFE_TOME_R requires PUT_BOUNDARY_SPLIT=1.')
        if self.safe_tome_score_mode == 'similarity' and similarity_guidance is None:
            raise RuntimeError('PUT_SAFE_TOME_R requires PUT_SIMILARITY_SCORER=1.')
        if self.ga_spg_guidance_enabled and similarity_guidance is None:
            raise RuntimeError('GA-SPG requires static detached guidance features.')

        b, h, w, c = emb.shape
        device = emb.device
        num_tokens = h * w
        emb_seq = emb.reshape(b, num_tokens, c)
        safe_mask = boundary_split['safe_candidate_token'].view(b, num_tokens).bool()
        protect_mask = boundary_split['protect_mask_token'].view(b, num_tokens).bool()
        runtime = self._build_safe_tome_runtime_cache(boundary_split, similarity_guidance)
        selection = runtime['selection']
        distance_raw_flat = runtime['distance_raw_flat']
        token_coords_flat = runtime['token_coords_flat']
        boundary_split = dict(boundary_split)
        boundary_split['distance_raw_flat'] = distance_raw_flat
        boundary_split['token_coords_flat'] = token_coords_flat

        plans = []
        compressed_sequences = []
        total_removed = 0
        total_pair_count = 0
        total_invalid_pair_count = 0
        total_high_risk_pair_count = 0
        under_compression = False
        fallback_used = False
        fallback_reasons = []
        restore_alignment_ok = True
        max_len = 0
        selected_risk_values = []
        selected_base_metric_values = []
        selected_adjusted_values = []
        lean_profile = bool(
            self.phase5k_lean_profile_enabled
            and not self.safe_tome_debug_enabled
            and not self.ga_spg_audit_enabled
        )

        for batch_idx in range(b):
            emb_seq_i = emb_seq[batch_idx]
            runtime_plan = runtime['plans'][batch_idx]
            selection_plan = runtime_plan['selection_plan']
            eligible_idx = selection_plan['eligible_idx']
            frozen_idx = selection_plan['frozen_idx']
            frozen_seq = emb_seq_i.index_select(0, frozen_idx)

            eligible_seq = emb_seq_i.index_select(0, eligible_idx) if eligible_idx.numel() > 0 else emb_seq_i.new_zeros((0, c))
            actual_r = 0
            eligible_unmerge = None
            assignment_flat = None
            source_idx = None
            destination_idx = None
            pair_debug = None
            selected_pair_record = None
            candidate_pair_stats = {
                'invalid_pair_count': 0,
                'high_risk_pair_count': 0,
                'under_compression': False,
                'feasible_pair_count': 0,
                'fallback_used': False,
                'fallback_reason': '',
            }

            if eligible_idx.numel() >= 2:
                merge_meta = None
                if self.ga_spg_mode:
                    score_handle = self._phase4a_start_timing('Time_Scoring')
                    pair_metric = None
                    if self.ga_spg_lite_mode and self.ga_spg_lite_optimized and not self.safe_tome_debug_enabled:
                        pair_metric = self._build_ga_spg_lite_adjusted_metric_fast(
                            eligible_seq=eligible_seq,
                            runtime_plan=runtime_plan,
                        )
                    elif self.ga_spg_lite_mode:
                        pair_debug = self._build_ga_spg_lite_pair_metric(
                            eligible_seq=eligible_seq,
                            selection_plan=selection_plan,
                            guidance=similarity_guidance,
                            boundary_split=boundary_split,
                            batch_idx=batch_idx,
                            runtime_plan=runtime_plan if self.ga_spg_lite_optimized else None,
                            use_fast_adjustment=bool(self.ga_spg_lite_optimized),
                        )
                        pair_metric = pair_debug['adjusted_metric']
                    else:
                        pair_debug = self._build_ga_spg_pair_metric(
                            eligible_seq=eligible_seq,
                            selection_plan=selection_plan,
                            guidance=similarity_guidance,
                            boundary_split=boundary_split,
                            batch_idx=batch_idx,
                        )
                        pair_metric = pair_debug['adjusted_metric']
                    self._phase4a_end_timing(score_handle)
                    pairing_handle = self._phase4a_start_timing('Time_TopK_and_Pairing')
                    pairing_handle_phase5f = self._phase5f_start_timing('topk_matching_time')
                    if self.safe_tome_debug_enabled:
                        merge, eligible_unmerge, actual_r, merge_meta = _tome_bipartite_soft_matching_scores(
                            pair_metric.unsqueeze(0),
                            eligible_idx.numel(),
                            self.safe_tome_r,
                            return_metadata=True,
                        )
                    else:
                        merge, eligible_unmerge, actual_r = _tome_bipartite_soft_matching_scores(
                            pair_metric.unsqueeze(0),
                            eligible_idx.numel(),
                            self.safe_tome_r,
                            return_metadata=False,
                        )
                    self._phase4a_end_timing(pairing_handle)
                    self._phase5f_end_timing(pairing_handle_phase5f)
                else:
                    if self.safe_tome_debug_enabled and similarity_guidance is not None and self.ga_spg_guidance_enabled:
                        pair_debug = self._build_ga_spg_pair_metric(
                            eligible_seq=eligible_seq,
                            selection_plan=selection_plan,
                            guidance=similarity_guidance,
                            boundary_split=boundary_split,
                            batch_idx=batch_idx,
                        )
                    elif self.safe_tome_debug_enabled:
                        pair_debug = self._build_basic_pair_debug(
                            eligible_seq=eligible_seq,
                            selection_plan=selection_plan,
                            boundary_split=boundary_split,
                            batch_idx=batch_idx,
                        )
                    pairing_handle = self._phase4a_start_timing('Time_TopK_and_Pairing')
                    pairing_handle_phase5f = self._phase5f_start_timing('topk_matching_time')
                    if self.safe_tome_debug_enabled:
                        merge, eligible_unmerge, actual_r, merge_meta = _tome_bipartite_soft_matching(
                            eligible_seq.unsqueeze(0),
                            self.safe_tome_r,
                            return_metadata=True,
                        )
                    else:
                        merge, eligible_unmerge, actual_r = _tome_bipartite_soft_matching(
                            eligible_seq.unsqueeze(0),
                            self.safe_tome_r,
                            return_metadata=False,
                        )
                    self._phase4a_end_timing(pairing_handle)
                    self._phase5f_end_timing(pairing_handle_phase5f)

                if pair_debug is not None:
                    total_invalid_pair_count += int(pair_debug.get('invalid_pair_count', 0))
                    total_high_risk_pair_count += int(pair_debug.get('high_risk_pair_count', 0))
                    under_compression = under_compression or bool(pair_debug.get('under_compression', False))
                    fallback_used = fallback_used or bool(pair_debug.get('fallback_used', False))
                    fallback_reason = str(pair_debug.get('fallback_reason', '')).strip()
                    if fallback_reason:
                        fallback_reasons.append(fallback_reason)
                    candidate_pair_stats = {
                        'invalid_pair_count': int(pair_debug.get('invalid_pair_count', 0)),
                        'high_risk_pair_count': int(pair_debug.get('high_risk_pair_count', 0)),
                        'under_compression': bool(pair_debug.get('under_compression', False)),
                        'feasible_pair_count': int(pair_debug.get('feasible_pair_count', 0)),
                        'fallback_used': bool(pair_debug.get('fallback_used', False)),
                        'fallback_reason': fallback_reason,
                    }

                if int(actual_r) < int(self.safe_tome_r):
                    under_compression = True
                    if not candidate_pair_stats.get('fallback_reason'):
                        candidate_pair_stats['fallback_reason'] = 'insufficient_budget_for_target_r'

                if actual_r > 0:
                    merge_handle = self._phase4a_start_timing('Time_Merge')
                    merge_handle_phase5f = self._phase5f_start_timing('merge_time')
                    merged_seq = merge(eligible_seq.unsqueeze(0), mode='mean').squeeze(0)
                    compressed_seq = torch.cat([frozen_seq, merged_seq], dim=0)
                    if self.safe_tome_debug_enabled:
                        merged_len = merged_seq.shape[0]
                        group_ids = torch.arange(
                            merged_len,
                            device=device,
                            dtype=eligible_seq.dtype,
                        ).view(1, merged_len, 1)
                        restored_group_ids = eligible_unmerge(group_ids).view(-1).round().long()
                        assignment_flat = torch.empty(num_tokens, dtype=torch.long, device=device)
                        assignment_flat[frozen_idx] = torch.arange(
                            selection_plan['frozen_len'],
                            device=device,
                            dtype=torch.long,
                        )
                        assignment_flat[eligible_idx] = restored_group_ids + selection_plan['frozen_len']
                        source_idx = selection_plan['even_idx'].index_select(0, merge_meta['src_idx'][0, :, 0])
                        destination_idx = selection_plan['odd_idx'].gather(0, merge_meta['dst_idx'][0, :, 0])
                        if pair_debug is None:
                            pair_debug = self._build_basic_pair_debug(
                                eligible_seq=eligible_seq,
                                selection_plan=selection_plan,
                                boundary_split=boundary_split,
                                batch_idx=batch_idx,
                            )
                        src_sel = merge_meta['src_idx'][0, :, 0]
                        dst_sel = merge_meta['dst_idx'][0, :, 0]
                        selected_base_metric = pair_debug['base_metric'][src_sel, dst_sel]
                        selected_risk_penalty = pair_debug['risk_penalty'][src_sel, dst_sel]
                        selected_adjusted_metric = pair_debug['adjusted_metric'][src_sel, dst_sel]
                        selected_pair_record = {
                            'pair_src_idx': source_idx.detach().cpu().tolist(),
                            'pair_dst_idx': destination_idx.detach().cpu().tolist(),
                            'base_metric': selected_base_metric.detach().cpu().tolist(),
                            'risk_penalty': selected_risk_penalty.detach().cpu().tolist(),
                            'adjusted_metric': selected_adjusted_metric.detach().cpu().tolist(),
                            'pair_cos_raw': pair_debug['pair_cos_raw'][src_sel, dst_sel].detach().cpu().tolist(),
                            'pair_cos_3x3': pair_debug['pair_cos_3x3'][src_sel, dst_sel].detach().cpu().tolist(),
                            'pair_cos_9x9': pair_debug['pair_cos_9x9'][src_sel, dst_sel].detach().cpu().tolist(),
                            'pair_feature_l2_raw': pair_debug['pair_feature_l2_raw'][src_sel, dst_sel].detach().cpu().tolist(),
                            'pair_feature_l2_3x3': pair_debug['pair_feature_l2_3x3'][src_sel, dst_sel].detach().cpu().tolist(),
                            'pair_feature_l2_9x9': pair_debug['pair_feature_l2_9x9'][src_sel, dst_sel].detach().cpu().tolist(),
                            'pair_local_variance_risk': pair_debug['pair_local_variance_risk'][src_sel, dst_sel].detach().cpu().tolist(),
                            'pair_context_variance_risk': pair_debug['pair_context_variance_risk'][src_sel, dst_sel].detach().cpu().tolist(),
                            'pair_boundary_distance_min': pair_debug['pair_boundary_distance_min'][src_sel, dst_sel].detach().cpu().tolist(),
                            'pair_spatial_distance': pair_debug['pair_spatial_distance'][src_sel, dst_sel].detach().cpu().tolist(),
                            'selected_source_boundary_distance': pair_debug['pair_boundary_distance_src'][src_sel].detach().cpu().tolist(),
                            'selected_destination_boundary_distance': pair_debug['pair_boundary_distance_dst'][dst_sel].detach().cpu().tolist(),
                            'selected_source_context_risk': pair_debug['pair_context_risk_src'][src_sel].detach().cpu().tolist(),
                            'selected_destination_context_risk': pair_debug['pair_context_risk_dst'][dst_sel].detach().cpu().tolist(),
                            'selected_source_local_risk': pair_debug['pair_local_risk_src'][src_sel].detach().cpu().tolist(),
                            'selected_destination_local_risk': pair_debug['pair_local_risk_dst'][dst_sel].detach().cpu().tolist(),
                        }
                        selected_base_metric_values.append(selected_base_metric.detach())
                        selected_risk_values.append(selected_risk_penalty.detach())
                        selected_adjusted_values.append(selected_adjusted_metric.detach())
                    self._phase4a_end_timing(merge_handle)
                    self._phase5f_end_timing(merge_handle_phase5f)
                else:
                    compressed_seq = emb_seq_i
            else:
                compressed_seq = emb_seq_i
                if eligible_idx.numel() > 0 and eligible_idx.numel() < 2:
                    candidate_pair_stats['fallback_reason'] = 'insufficient_eligible_tokens'
                    under_compression = True
                elif int(self.safe_tome_r) > 0:
                    under_compression = True
                    candidate_pair_stats['fallback_reason'] = 'insufficient_budget_for_target_r'

            compressed_len = int(compressed_seq.shape[0])
            alignment_ok = True
            if assignment_flat is not None:
                restore_groups = int(torch.unique(assignment_flat).numel())
                alignment_ok = restore_groups == compressed_len
            restore_alignment_ok = restore_alignment_ok and alignment_ok

            plan = {
                'frozen_idx': frozen_idx,
                'eligible_idx': eligible_idx,
                'compressed_len': compressed_len,
                'frozen_len': selection_plan['frozen_len'],
                'actual_r': int(actual_r),
                'eligible_unmerge': eligible_unmerge,
                'alignment_ok': alignment_ok,
            }
            if self.safe_tome_debug_enabled:
                source_mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)
                destination_mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)
                if source_idx is not None and source_idx.numel() > 0:
                    source_mask[source_idx] = True
                if destination_idx is not None and destination_idx.numel() > 0:
                    destination_mask[destination_idx] = True
                plan.update(
                    {
                        'assignment_flat': assignment_flat,
                        'eligible_mask_flat': selection_plan['eligible_mask_flat'],
                        'source_mask_flat': source_mask,
                        'destination_mask_flat': destination_mask,
                        'source_indices': source_idx.detach().cpu().tolist() if source_idx is not None else [],
                        'destination_indices': destination_idx.detach().cpu().tolist() if destination_idx is not None else [],
                        'selected_pair_record': selected_pair_record,
                        'candidate_pair_stats': candidate_pair_stats,
                    }
                )
            plans.append(plan)
            compressed_sequences.append(compressed_seq)
            max_len = max(max_len, compressed_len)
            total_removed += int(actual_r)
            total_pair_count += int(actual_r)

        compressed_batch = emb.new_zeros((b, max_len, c))
        compressed_valid = torch.zeros((b, max_len), dtype=torch.bool, device=device)
        for batch_idx, seq in enumerate(compressed_sequences):
            seq_len = seq.shape[0]
            compressed_batch[batch_idx, :seq_len] = seq
            compressed_valid[batch_idx, :seq_len] = True

        protect_overlap_tokens = int(selection['protect_overlap_tokens'])
        selected_score_stats = selection['selected_score_stats']
        if lean_profile:
            empty_stats = {'mean': 0.0, 'min': 0.0, 'max': 0.0, 'std': 0.0}
            selected_risk_stats = empty_stats
            selected_base_metric_stats = empty_stats
            selected_adjusted_stats = empty_stats
            compressed_token_count = int(sum(int(seq.shape[0]) for seq in compressed_sequences))
            protect_token_count = int(selection.get('total_protect', 0))
            safe_token_count = int(selection.get('total_safe', 0))
        else:
            selected_risk_stats = self._flat_stats(
                torch.cat(selected_risk_values, dim=0)
                if len(selected_risk_values) > 0
                else emb.new_zeros((0,))
            )
            selected_base_metric_stats = self._flat_stats(
                torch.cat(selected_base_metric_values, dim=0)
                if len(selected_base_metric_values) > 0
                else emb.new_zeros((0,))
            )
            selected_adjusted_stats = self._flat_stats(
                torch.cat(selected_adjusted_values, dim=0)
                if len(selected_adjusted_values) > 0
                else emb.new_zeros((0,))
            )
            compressed_token_count = int(compressed_valid.sum().detach().cpu())
            protect_token_count = int(protect_mask.sum().detach().cpu())
            safe_token_count = int(safe_mask.sum().detach().cpu())
        profile = {
            'enabled': True,
            'requested_r': self.safe_tome_r,
            'score_mode': self.safe_tome_score_mode,
            'optimized': bool(self.ga_spg_lite_mode and self.ga_spg_lite_optimized),
            'runtime_cache_enabled': bool(
                self.ga_spg_lite_mode
                and self.ga_spg_lite_optimized
                and self.ga_spg_lite_runtime_cache_enabled
            ),
            'pad_mode': self.ga_spg_lite_pad_mode if self.ga_spg_lite_mode else 'n/a',
            'explicit_image_border_risk': bool(
                self.ga_spg_lite_mode and self.ga_spg_lite_explicit_border_risk
            ),
            'calls': self.safe_tome_profile['calls'] + 1,
            'token_shape': [int(h), int(w)],
            'original_tokens': int(b * num_tokens),
            'compressed_tokens': int(compressed_token_count),
            'protect_tokens': int(protect_token_count),
            'safe_tokens': int(safe_token_count),
            'eligible_tokens': int(selection['total_eligible']),
            'merged_tokens': int(compressed_token_count),
            'removed_tokens': int(total_removed),
            'restored_tokens': int(b * num_tokens),
            'pair_count': int(total_pair_count),
            'selected_score_mean': selected_score_stats['mean'],
            'selected_score_min': selected_score_stats['min'],
            'selected_score_max': selected_score_stats['max'],
            'selected_score_std': selected_score_stats['std'],
            'actual_compression_ratio': float(total_removed / max(int(b * num_tokens), 1)),
            'protect_overlap_tokens': int(protect_overlap_tokens),
            'invalid_pair_count': int(total_invalid_pair_count),
            'high_risk_pair_count': int(total_high_risk_pair_count),
            'mean_risk_penalty_selected': selected_risk_stats['mean'],
            'mean_base_metric_selected': selected_base_metric_stats['mean'],
            'mean_adjusted_metric_selected': selected_adjusted_stats['mean'],
            'under_compression': bool(under_compression),
            'fallback_used': bool(fallback_used),
            'fallback_reason': ';'.join(sorted(set(reason for reason in fallback_reasons if reason))),
            'restore_alignment_ok': bool(restore_alignment_ok),
        }
        debug = None
        if self.safe_tome_debug_enabled:
            debug = {
                'eligible_mask': torch.stack([p['eligible_mask_flat'].view(1, h, w) for p in plans], dim=0),
                'source_mask': torch.stack([p['source_mask_flat'].view(1, h, w) for p in plans], dim=0),
                'destination_mask': torch.stack([p['destination_mask_flat'].view(1, h, w) for p in plans], dim=0),
                'assignment_map': torch.stack([p['assignment_flat'].view(1, h, w) for p in plans], dim=0),
                'source_indices': [p['source_indices'] for p in plans],
                'destination_indices': [p['destination_indices'] for p in plans],
                'restore_alignment_ok': [bool(p['alignment_ok']) for p in plans],
                'compressed_tokens': [int(p['compressed_len']) for p in plans],
                'removed_tokens': [int(p['actual_r']) for p in plans],
                'eligible_tokens': [int(p['eligible_idx'].numel()) for p in plans],
                'selected_pair_records': [p.get('selected_pair_record') for p in plans],
                'candidate_pair_stats': [p.get('candidate_pair_stats') for p in plans],
            }

        state = {
            'plans': plans,
            'original_shape': (b, h, w, c),
            'num_tokens': int(num_tokens),
            'compressed_embedding': compressed_batch.unsqueeze(dim=1),
            'compressed_mask': compressed_valid.unsqueeze(dim=1),
            'profile': profile,
            'debug': debug,
        }
        self.safe_tome_profile = profile
        return state

    def _restore_safe_tome_embedding(self, emb, state):
        if state is None:
            return emb

        restore_handle = self._phase4a_start_timing('Time_Restore')
        restore_handle_phase5f = self._phase5f_start_timing('restore_time')
        emb_seq = emb.squeeze(dim=1)
        b, h, w, c = state['original_shape']
        restored = emb_seq.new_empty((b, state['num_tokens'], c))
        for batch_idx, plan in enumerate(state['plans']):
            seq = emb_seq[batch_idx, :plan['compressed_len']]
            if plan['actual_r'] > 0:
                frozen_seq = seq[:plan['frozen_len']]
                merged_seq = seq[plan['frozen_len']:]
                restored_eligible = plan['eligible_unmerge'](merged_seq.unsqueeze(0)).squeeze(0)
                restored[batch_idx, plan['frozen_idx']] = frozen_seq
                restored[batch_idx, plan['eligible_idx']] = restored_eligible
            else:
                restored[batch_idx] = seq[:h * w]
        self._phase4a_end_timing(restore_handle)
        self._phase5f_end_timing(restore_handle_phase5f)
        return restored.view(*state['original_shape'])

    def _safe_tome_selection_signature(self, boundary_split, similarity_guidance):
        safe = boundary_split['safe_candidate_token']
        protect = boundary_split['protect_mask_token']
        signature = [
            int(self.safe_tome_r),
            bool(self.safe_tome_debug_enabled),
            str(self.safe_tome_score_mode),
            int(safe.data_ptr()),
            tuple(safe.shape),
            int(protect.data_ptr()),
            tuple(protect.shape),
        ]
        if self.safe_tome_score_mode == 'similarity':
            score = similarity_guidance['redundancy_score_norm']
            signature.extend([int(score.data_ptr()), tuple(score.shape)])
        elif similarity_guidance is not None and 'distance_score' in similarity_guidance:
            score = similarity_guidance['distance_score']
            signature.extend([int(score.data_ptr()), tuple(score.shape)])
        else:
            signature.extend([0, ()])
        return tuple(signature)

    def _safe_tome_selection_score(self, boundary_split, similarity_guidance):
        if self.safe_tome_score_mode == 'similarity':
            if similarity_guidance is None:
                raise RuntimeError('PUT_SAFE_TOME_SCORE_MODE=similarity requires similarity guidance.')
            return similarity_guidance['redundancy_score_norm']
        if similarity_guidance is not None and 'distance_score' in similarity_guidance:
            return similarity_guidance['distance_score']
        distance_handle = self._phase5f_start_timing('distance_map_time')
        _, distance_score = self.similarity_scorer._distance_rule_score(
            boundary_split['protect_mask_token'],
            boundary_split['safe_candidate_token'],
        )
        self._phase5f_end_timing(distance_handle)
        return distance_score

    def _safe_tome_runtime_signature(self, boundary_split, similarity_guidance):
        signature = [
            self._safe_tome_selection_signature(boundary_split, similarity_guidance),
            bool(self.ga_spg_lite_mode),
            bool(self.ga_spg_lite_optimized),
            bool(self.ga_spg_lite_runtime_cache_enabled),
            self.ga_spg_lite_pad_mode if self.ga_spg_lite_mode else 'n/a',
            bool(self.ga_spg_lite_explicit_border_risk),
        ]
        if self.ga_spg_lite_mode and similarity_guidance is not None:
            for key in ('scalar_risk', 'local_variance_risk', 'smoothness_risk', 'distance_raw'):
                tensor = similarity_guidance.get(key)
                if tensor is None:
                    signature.extend([key, 0, ()])
                else:
                    signature.extend([key, int(tensor.data_ptr()), tuple(tensor.shape), str(tensor.dtype)])
        return tuple(signature)

    def _build_safe_tome_runtime_cache(self, boundary_split, similarity_guidance):
        signature = self._safe_tome_runtime_signature(boundary_split, similarity_guidance)
        use_runtime_cache = bool(
            self.ga_spg_lite_mode
            and self.ga_spg_lite_optimized
            and self.ga_spg_lite_runtime_cache_enabled
        )
        if use_runtime_cache and self._safe_tome_runtime_cache_signature == signature and self._safe_tome_runtime_cache is not None:
            hit_handle = self._phase5f_start_timing('runtime_cache_hit_time')
            self._phase5f_end_timing(hit_handle)
            return self._safe_tome_runtime_cache

        build_handle = self._phase5f_start_timing('runtime_cache_build_time')
        selection = self._get_safe_tome_selection(boundary_split, similarity_guidance)
        b = boundary_split['safe_candidate_token'].shape[0]
        num_tokens = int(boundary_split['safe_candidate_token'].shape[-2] * boundary_split['safe_candidate_token'].shape[-1])
        device = boundary_split['safe_candidate_token'].device
        dtype = boundary_split['token_visible_ratio'].dtype

        if similarity_guidance is not None and 'distance_raw_flat' in similarity_guidance:
            distance_raw_flat = similarity_guidance['distance_raw_flat']
        elif similarity_guidance is not None and 'distance_raw' in similarity_guidance:
            distance_raw_flat = similarity_guidance['distance_raw'].view(b, num_tokens)
        else:
            distance_raw_flat, _ = self.similarity_scorer._distance_rule_score(
                boundary_split['protect_mask_token'],
                boundary_split['safe_candidate_token'],
            )
            distance_raw_flat = distance_raw_flat.view(b, num_tokens)

        if similarity_guidance is not None and 'token_coords_flat' in similarity_guidance:
            token_coords_flat = similarity_guidance['token_coords_flat']
        elif 'token_coords_flat' in boundary_split:
            token_coords_flat = boundary_split['token_coords_flat']
        else:
            h = int(boundary_split['safe_candidate_token'].shape[-2])
            w = int(boundary_split['safe_candidate_token'].shape[-1])
            yy, xx = torch.meshgrid(
                torch.arange(h, device=device, dtype=dtype),
                torch.arange(w, device=device, dtype=dtype),
                indexing='ij',
            )
            token_coords_flat = torch.stack([yy, xx], dim=-1).view(1, num_tokens, 2).expand(b, -1, -1).contiguous()

        runtime = {
            'selection': selection,
            'distance_raw_flat': distance_raw_flat,
            'token_coords_flat': token_coords_flat,
            'plans': [],
        }

        if self.ga_spg_lite_mode and similarity_guidance is not None and self.ga_spg_lite_optimized:
            scalar_risk_flat = similarity_guidance.get('scalar_risk_flat')
            if scalar_risk_flat is None:
                scalar_risk_flat = similarity_guidance['scalar_risk'].view(b, num_tokens)
            local_risk_flat = similarity_guidance.get('local_variance_risk_flat')
            if local_risk_flat is None and self.safe_tome_debug_enabled:
                local_risk_flat = similarity_guidance['local_variance_risk'].view(b, num_tokens)
            smoothness_risk_flat = similarity_guidance.get('smoothness_risk_flat')
            if smoothness_risk_flat is None and self.safe_tome_debug_enabled:
                smoothness_risk_flat = similarity_guidance['smoothness_risk'].view(b, num_tokens)

            lambda_risk = float(self.ga_spg_lite_lambda_risk)
            for batch_idx in range(b):
                selection_plan = selection['plans'][batch_idx]
                plan = {
                    'selection_plan': selection_plan,
                }
                even_idx = selection_plan['even_idx']
                odd_idx = selection_plan['odd_idx']
                src_scalar = scalar_risk_flat[batch_idx].index_select(0, even_idx).contiguous()
                dst_scalar = scalar_risk_flat[batch_idx].index_select(0, odd_idx).contiguous()
                plan['src_scalar'] = src_scalar
                plan['dst_scalar'] = dst_scalar
                plan['src_penalty'] = (src_scalar * lambda_risk).contiguous()
                plan['dst_penalty'] = (dst_scalar * lambda_risk).contiguous()
                if self.safe_tome_debug_enabled:
                    current_distance = distance_raw_flat[batch_idx]
                    current_coords = token_coords_flat[batch_idx]
                    current_local = local_risk_flat[batch_idx]
                    current_smooth = smoothness_risk_flat[batch_idx]
                    plan.update(
                        {
                            'src_local': current_local.index_select(0, even_idx).contiguous(),
                            'dst_local': current_local.index_select(0, odd_idx).contiguous(),
                            'src_smooth': current_smooth.index_select(0, even_idx).contiguous(),
                            'dst_smooth': current_smooth.index_select(0, odd_idx).contiguous(),
                            'src_distance': current_distance.index_select(0, even_idx).float().contiguous(),
                            'dst_distance': current_distance.index_select(0, odd_idx).float().contiguous(),
                            'src_coords': current_coords.index_select(0, even_idx).float().contiguous(),
                            'dst_coords': current_coords.index_select(0, odd_idx).float().contiguous(),
                        }
                    )
                runtime['plans'].append(plan)
        else:
            for batch_idx in range(b):
                runtime['plans'].append({'selection_plan': selection['plans'][batch_idx]})

        if use_runtime_cache:
            self._safe_tome_runtime_cache_signature = signature
            self._safe_tome_runtime_cache = runtime
        else:
            self._safe_tome_runtime_cache_signature = None
            self._safe_tome_runtime_cache = None
        self._phase5f_end_timing(build_handle)
        return runtime

    def _phase5k_tensor_checksum(self, tensor):
        if tensor is None:
            return 0.0
        return float(tensor.detach().float().sum().detach().cpu())

    def _phase5k_record_step_audit(
        self,
        step_idx,
        input_mask,
        current_mask,
        boundary_split,
        similarity_guidance,
        safe_tome_state,
    ):
        if not self.phase5k_audit_enabled:
            return
        route = safe_tome_state.get('profile', {}) if safe_tome_state is not None else {}
        record = {
            'step_idx': int(step_idx),
            'input_mask_checksum': self._phase5k_tensor_checksum(input_mask),
            'current_mask_checksum': self._phase5k_tensor_checksum(current_mask),
            'boundary_ring_checksum': self._phase5k_tensor_checksum(
                boundary_split.get('boundary_token') if boundary_split is not None else None
            ),
            'safe_mask_checksum': self._phase5k_tensor_checksum(
                boundary_split.get('safe_candidate_token') if boundary_split is not None else None
            ),
            'protect_mask_checksum': self._phase5k_tensor_checksum(
                boundary_split.get('protect_mask_token') if boundary_split is not None else None
            ),
            'distance_raw_checksum': self._phase5k_tensor_checksum(
                similarity_guidance.get('distance_raw') if similarity_guidance is not None else None
            ),
            'risk_i_checksum': self._phase5k_tensor_checksum(
                similarity_guidance.get('scalar_risk') if similarity_guidance is not None else None
            ),
            'actual_removed_tokens': int(route.get('removed_tokens', 0)),
            'protect_overlap_tokens': int(route.get('protect_overlap_tokens', 0)),
            'restore_alignment_ok': bool(route.get('restore_alignment_ok', True)),
        }
        self.phase5k_audit_profile['records'].append(record)

    def _get_safe_tome_selection(self, boundary_split, similarity_guidance):
        signature = self._safe_tome_selection_signature(boundary_split, similarity_guidance)
        if self._safe_tome_selection_cache_signature == signature and self._safe_tome_selection_cache is not None:
            return self._safe_tome_selection_cache

        timing_handle = self._phase5f_start_timing('pure_distance_pool_time')

        b = boundary_split['safe_candidate_token'].shape[0]
        num_tokens = int(boundary_split['safe_candidate_token'].shape[-2] * boundary_split['safe_candidate_token'].shape[-1])
        safe_mask = boundary_split['safe_candidate_token'].view(b, num_tokens).bool()
        protect_mask = boundary_split['protect_mask_token'].view(b, num_tokens).bool()
        score = self._safe_tome_selection_score(boundary_split, similarity_guidance).view(b, num_tokens)
        all_idx = torch.arange(num_tokens, device=safe_mask.device, dtype=torch.long)

        plans = []
        selected_score_values = []
        total_eligible = 0
        total_safe = 0
        total_protect = 0
        protect_overlap_tokens = 0
        collect_selection_stats = not (
            self.phase5k_lean_profile_enabled
            and not self.safe_tome_debug_enabled
            and not self.ga_spg_audit_enabled
        )

        for batch_idx in range(b):
            safe_idx = safe_mask[batch_idx].nonzero(as_tuple=False).flatten()
            safe_count = int(safe_idx.numel())
            total_safe += safe_count
            total_protect += int(num_tokens - safe_count)
            eligible_target = self._safe_tome_selected_count(safe_count)

            if eligible_target > 0:
                candidate_scores = score[batch_idx, safe_idx]
                _, topk = torch.topk(candidate_scores, k=eligible_target, dim=0, largest=True)
                eligible_idx = torch.sort(safe_idx[topk]).values
                if collect_selection_stats:
                    selected_score_values.append(score[batch_idx, eligible_idx])
            else:
                eligible_idx = all_idx[:0]

            frozen_mask = torch.ones(num_tokens, dtype=torch.bool, device=safe_mask.device)
            if eligible_idx.numel() > 0:
                frozen_mask[eligible_idx] = False
                if collect_selection_stats:
                    protect_overlap_tokens += int(protect_mask[batch_idx, eligible_idx].sum().detach().cpu())
            plans.append(
                {
                    'eligible_idx': eligible_idx,
                    'frozen_idx': all_idx[frozen_mask],
                    'frozen_len': int(num_tokens - eligible_idx.numel()),
                    'even_idx': eligible_idx[::2],
                    'odd_idx': eligible_idx[1::2],
                    'eligible_mask_flat': frozen_mask.logical_not() if self.safe_tome_debug_enabled else None,
                }
            )
            total_eligible += int(eligible_idx.numel())

        selection = {
            'plans': plans,
            'total_eligible': int(total_eligible),
            'total_safe': int(total_safe),
            'total_protect': int(total_protect),
            'protect_overlap_tokens': int(protect_overlap_tokens),
            'selected_score_stats': (
                self._flat_stats(
                    torch.cat(selected_score_values, dim=0)
                    if len(selected_score_values) > 0
                    else score.new_zeros((0,))
                )
                if collect_selection_stats
                else {'mean': 0.0, 'min': 0.0, 'max': 0.0, 'std': 0.0}
            ),
        }
        self._safe_tome_selection_cache_signature = signature
        self._safe_tome_selection_cache = selection
        self._phase5f_end_timing(timing_handle)
        return selection

    @torch.no_grad()
    def generate_content(
        self,
        batch,
        filter_ratio = 0.5,
        filter_type = 'count',
        temperature = 1.0,
        replicate=1,
        mask_low_to_high=False,
        num_token_per_iter=1,
        calculate_acc_and_prob=True,
        accumulate_time=None,
        raster_order=False,
        **kwargs,
    ):
        self.eval()
        if replicate != 1:
            for k in batch.keys():
                if batch[k] is not None and torch.is_tensor(batch[k]):
                    batch[k] = torch.cat([batch[k] for _ in range(replicate)], dim=0)
        return self.sample(
            batch=batch,
            filter_ratio=filter_ratio,
            filter_type=filter_type,
            temperature=temperature,
            return_gt=False,
            return_mask_gt=False,
            return_reconstruction=False,
            mask_low_to_high=mask_low_to_high,
            num_token_per_iter=num_token_per_iter,
            calculate_acc_and_prob=calculate_acc_and_prob,
            accumulate_time=accumulate_time,
            raster_order=raster_order,

        )           


    @torch.no_grad()
    def sample(
        self,
        *,
        batch,
        filter_ratio = 0.8,
        filter_type='count',
        temperature = 1.0,
        return_gt=True,
        return_mask_gt=True,
        return_reconstruction=True,
        calculate_acc_and_prob=True, # calculate token accuracy
        mask_low_to_high=False,
        num_token_per_iter=None,
        accumulate_time=None, # for get the time consumption
        raster_order=False,
        **kwargs,
    ): 
        self.eval()
        self._phase4a_reset_timing()
        self._phase5f_reset_timing()
        self._safe_tome_selection_cache_signature = None
        self._safe_tome_selection_cache = None
        self._safe_tome_runtime_cache_signature = None
        self._safe_tome_runtime_cache = None

        for k in batch.keys():
            if torch.is_tensor(batch[k]):# isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(self.device)

        if mask_low_to_high:
            low_res = self.content_codec.token_shape
            ori_res = batch['mask'].shape[-2:]
            assert low_res is not None 
            # import pdb; pdb.set_trace()
            mask_ = F.interpolate(batch['mask'].float(), size=low_res, mode='nearest')
            mask_ = F.interpolate(mask_, size=ori_res, mode='nearest').bool()     
        else:
            mask_ = batch['mask']

            # batch['mask'] = mask_.clone()
        if accumulate_time is None:
            accumulate_time = {
                'encoder': 0,
                'prepare': 0,
                'transformer': 0,
                'decoder': 0,
                'count': 0,
            }

        total_inference_handle = self._phase5f_start_timing('total_inference_time')
        tic = time.time()
        need_guidance = self.similarity_scorer_enabled or self.ga_spg_guidance_enabled
        need_quantized_guidance = need_guidance and self._similarity_guidance_requires_quantized_feature()
        data_mask = self.content_codec.get_features(batch['image'], 
                                                    mask=mask_, 
                                                    return_quantize_feature=(self.input_feature_type == 'quantized') or need_quantized_guidance,
                                                    return_token=True, mask_pixel_value=self.mask_pixel_value) # dict
        accumulate_time['encoder'] = (accumulate_time['encoder'] * accumulate_time['count'] + time.time() - tic) / (accumulate_time['count']+1)

        tic = time.time()
        if self.input_feature_type == 'origin':
            feat_mask = data_mask['feature'] # B x C x H x W
        elif self.input_feature_type == 'quantized':
            feat_mask = data_mask['feature_quantize'] # B x C x H x W
        else:
            raise NotImplementedError('inpute feature type {} not implemented!'.format(self.input_feature_type))
        b, _, h, w = feat_mask.shape

        token_type, unmask_ratio = get_token_type(mask_, type='pixel_shuffle', token_shape=[h,w]) # B x 1 x H x W
        feat_mask = self.emb_proj(feat_mask.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) # B x C x H x W
        
        content_shape = (h // self.content_patch_token_shape[0], w // self.content_patch_token_shape[1])
        if self.pos_emb is not None:
            pos_emb = self.pos_emb.permute(0, 2, 1).view(1, -1, content_shape[0], content_shape[1]) # B x D x H/cps x W/cps
            pos_emb = pixel_shuffle(pos_emb, out_size=(h, w), chunked=True) # B x C x H x W
        else:
            pos_emb = torch.zeros_like(feat_mask)
        
        if self.mask_emb is not None:
            feat_mask = feat_mask * unmask_ratio + self.mask_emb * (1-unmask_ratio)

        feat_mask = feat_mask + pos_emb 
        
        # save features before feeding into transformer block
        if False:
            # import pdb; pdb.set_trace()
            feat_save_path = os.path.join('RESULT/debug/feature_pos_mask/{}_feature.pt'.format(batch['relative_path'][0].replace('.png','')))
            token_type_save_path = os.path.join(os.path.dirname(feat_save_path), '{}_unmask_ratio.pt'.format(batch['relative_path'][0].replace('.png','')))
            os.makedirs(os.path.dirname(feat_save_path), exist_ok=True)
            torch.save(feat_mask.to('cpu'), feat_save_path)
            torch.save(unmask_ratio, token_type_save_path)

            # import sys
            # sys.exit(1)

        
        
        content_feat = feat_mask # B x C x H x W
        content_token = data_mask['token'] # B x H x W
        content_mask = (token_type == 1).squeeze(dim=1) # B x H x W
        boundary_split = self._boundary_split(mask_, content_shape)
        similarity_guidance = None
        if need_guidance and boundary_split is not None:
            emb_for_score = self._build_similarity_guidance_feature(
                data_mask=data_mask,
                projected_feature=content_feat,
                content_shape=content_shape,
            )
            similarity_guidance = self._compute_safe_tome_guidance(emb_for_score, boundary_split)

        accumulate_time['prepare'] = (accumulate_time['prepare'] * accumulate_time['count'] + time.time() - tic) / (accumulate_time['count']+1)


        # begin to sample
        # import pdb; pdb.set_trace()
        step = 0
        tic = time.time()
        forward_time = 0
        sample_time = time.time()
        save_each_step_image = False

        if save_each_step_image:

            cache_decoder_requires_image = self.content_codec.decoder.requires_image
            cache_decoder_uo_layer_with_image = self.content_codec.decoder.up_layer_with_image


            patch_size = batch['image'].shape[-1] // w

            content_token.masked_fill_(~content_mask, 285) # set all invalid token to a fixed token for better comparison
            completed_iter = self.content_codec.decode(content_token, mask_im=batch['image'] * batch['mask'], mask=batch['mask'], token_shape=[h,w]) # B x C x H x W
            completed_iter = completed_iter[0].permute(1,2,0).to('cpu').numpy().astype(np.uint8)
            completed_iter = Image.fromarray(completed_iter)
            # save 
            save_path = os.path.join('RESULT/debug', batch['relative_path'][0], 'completed_{}.png'.format(str(step).zfill(len(str(h*w)))))
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            completed_iter.save(save_path)
            print('saved to {}'.format(save_path))


        if calculate_acc_and_prob:
            content_token_target = self.content_codec.get_features(batch['image'], 
                                                    return_quantize_feature=False,
                                                    return_token=True)['token'] # B x H x W
            acc_all = []
            prob_all = []


        if num_token_per_iter is None:
            num_token_per_iter = max(self.content_patch_seq_len, int(h*w//500))

        num_masked_tokens = (~content_mask).flatten(1).sum(-1).max() # in a batch, we get the max number of the masked tokens for iteratively sampling
        # import pdb; pdb.set_trace()
        if isinstance(num_token_per_iter, (int,)):
            if num_token_per_iter > 0:
                total_steps = (num_masked_tokens + num_token_per_iter - 1) // num_token_per_iter
                step_nums = [num_token_per_iter for _ in range(total_steps)]
                diff = int(sum(step_nums) - num_masked_tokens)
                step_nums[-1] -= diff 
            else:
                total_steps = 1
                step_nums = [-1]
        elif isinstance(num_token_per_iter, str) and num_token_per_iter.split('_')[0] in ['cosine', 'linear', 'average',  'cosine-1', 'linear-1']:
            total_steps = int(num_token_per_iter.split('_')[-1])
            if num_token_per_iter.split('_')[0] in ['cosine', 'cosine-1']:
                step_nums = list(range(total_steps))
                step_nums = [math.cos(math.pi * sn/total_steps) + 1 for sn in step_nums]
            elif num_token_per_iter.split('_')[0] in ['linear', 'linear-1']:
                step_nums = list(range(total_steps, 0, -1))
            elif num_token_per_iter.split('_')[0] == 'average':
                step_nums = [num_masked_tokens // total_steps for _ in range(total_steps)]
            else:
                raise NotImplementedError('{}'.format(num_token_per_iter))
            # import pdb; pdb.set_trace()
            factor = float(num_masked_tokens) / sum(step_nums)
            step_nums = [round(sn * factor) for sn in step_nums]
            diff = int(sum(step_nums) - num_masked_tokens)
            idx = 0
            while diff != 0:
                idx = (idx + 1) % len(step_nums)
                factor = 1 if diff < 0 else -1
                diff = diff + 1 if diff < 0 else diff -1
                step_nums[idx] = step_nums[idx] + factor
            
            if num_token_per_iter.split('_')[0] in ['cosine-1', 'linear-1']:
                step_nums = step_nums[::-1]
        else:
            raise NotImplementedError('{}'.format(num_token_per_iter))


        while step < total_steps:
        # while content_mask.sum() < content_mask.numel():
            
            sn = step_nums[step]
            self._phase4a_num_sampling_steps += 1

            step += 1
            step_ = step 
            # import pdb; pdb.set_trace()
            if step % 100 == 0:
                print('Step: {}/{}'.format(step, total_steps), content_mask.sum(), content_mask.numel())

            # prepare data for transformer blocks to get the logits
            emb = pixel_unshuffle(content_feat, out_size=content_shape, chunked=True) # B x D x H/cps x W/cps
            emb = emb.permute(0, 2, 3, 1) # B x H/cps x W/cps x D
            if self.drop is not None:
                emb = self.drop(emb)
            if self.attn_content_with_mask:
                attn_mask = pixel_unshuffle(content_mask.unsqueeze(dim=1), out_size=content_shape, chunked=True) # B x cps*cps x H/cps x W/cps
                attn_mask = (attn_mask.sum(dim=1, keepdim=False) == attn_mask.shape[1]).to(content_mask.dtype) # B x H/cps x W/cps
            else:
                attn_mask = None
            
            tic_forward = time.time()

            tome_unmerge = None
            tome_origin_shape = None
            safe_tome_state = None
            if self.safe_tome_enabled:
                if attn_mask is not None:
                    raise RuntimeError('PUT_SAFE_TOME_R currently supports attn_content_with_mask=False only.')
                safe_tome_state = self._build_safe_tome_state(emb, boundary_split, similarity_guidance)
                self._phase5k_record_step_audit(
                    step_idx=step_,
                    input_mask=mask_,
                    current_mask=content_mask,
                    boundary_split=boundary_split,
                    similarity_guidance=similarity_guidance,
                    safe_tome_state=safe_tome_state,
                )
                emb = safe_tome_state['compressed_embedding']
                attn_mask = safe_tome_state['compressed_mask']
            elif self.global_tome_r > 0:
                if attn_mask is not None:
                    raise RuntimeError('PUT_GLOBAL_TOME_R currently supports attn_content_with_mask=False only.')
                tome_origin_shape = emb.shape
                b_tome, h_tome, w_tome, c_tome = emb.shape
                emb_seq = emb.reshape(b_tome, h_tome * w_tome, c_tome)
                pairing_handle = self._phase4a_start_timing('Time_TopK_and_Pairing')
                merge, tome_unmerge, actual_r = _tome_bipartite_soft_matching(emb_seq, self.global_tome_r)
                self._phase4a_end_timing(pairing_handle)
                if actual_r > 0:
                    merge_handle = self._phase4a_start_timing('Time_Merge')
                    emb_seq, _ = _tome_merge_wavg(merge, emb_seq, size=None)
                    self._phase4a_end_timing(merge_handle)
                    self.global_tome_profile['calls'] += 1
                    self.global_tome_profile['original_tokens'] = int(h_tome * w_tome)
                    self.global_tome_profile['merged_tokens'] = int(emb_seq.shape[1])
                    self.global_tome_profile['removed_tokens'] = int(actual_r)
                    self.global_tome_profile['restored_tokens'] = int(h_tome * w_tome)
                    self.global_tome_profile['pair_count'] = int(actual_r)
                    emb = emb_seq.unsqueeze(dim=1) # B x 1 x N' x D
                else:
                    tome_unmerge = None

            transformer_blocks_handle = self._phase5f_start_timing('transformer_blocks_time')
            for block_idx in range(len(self.blocks)):   
                emb, att_w = self.blocks[block_idx](emb, mask=attn_mask) # B x H x W x D, B x H x W x H x W
            self._phase5f_end_timing(transformer_blocks_handle)

            if safe_tome_state is not None:
                emb = self._restore_safe_tome_embedding(emb, safe_tome_state)
            elif tome_unmerge is not None:
                restore_handle = self._phase4a_start_timing('Time_Restore')
                restore_handle_phase5f = self._phase5f_start_timing('restore_time')
                emb = tome_unmerge(emb.squeeze(dim=1)).reshape(tome_origin_shape)
                self._phase4a_end_timing(restore_handle)
                self._phase5f_end_timing(restore_handle_phase5f)

            emb = self.norm(emb) # B x H/cps x W/cps x D
            emb = emb.permute(0, 3, 1, 2) # B x D x H/cps x W/cps
            emb = pixel_shuffle(emb, out_size=(h, w), chunked=True) # B x C x H x W
            emb = emb.permute(0, 2, 3, 1) # B x H x W x C
            logits = self.to_logits(emb) # B x  H x W x Cls

            forward_time += time.time() - tic_forward

            # for each position, only keep the topk probabilities
            logits_filter = logits_top_k(logits, filter_ratio=filter_ratio, minimum=1, filter_type=filter_type) # B x H x W x Cls
            probs = F.softmax(logits_filter * temperature, dim=-1) # B x H x W x Cls
            sample = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(*probs.shape[:3]) # B x H x W
            
            if sn == -1 or sn >= h*w:
                content_token = content_token * content_mask + sample * (~content_mask)
                pos_mask = ~content_mask
                content_mask = torch.ones_like(content_mask)
            else:
                # select the sn positions for sampling
                if raster_order:
                    index_raster = torch.tensor(list(range(h*w))).view(1, h, w).int().to(content_mask.device) # B x H x W
                    index_raster = index_raster + content_mask.int() * (h*w+1)
                    index_raster = 0 - index_raster.view(-1, h*w)
                    _, pos = torch.topk(index_raster, dim=1, k=sn) # B x num, in range [0, HW)
                    pos_mask = torch.zeros_like(index_raster).float().scatter_(1, pos, 1.0).to(content_mask.dtype) # B x HW
                else:
                    logits_max, _ = logits_filter.max(dim=-1) # B x H x W
                    logits_max.masked_fill_(content_mask, float('-inf')) # set the logits for those unmasked tokens to -inf
                    logits_max = logits_max.view(-1, h*w) # B x HW
                    _, pos = torch.topk(logits_max, dim=1, k=sn) # B x num, in range [0, HW)
                    pos_mask = torch.zeros_like(logits_max).scatter_(1, pos, 1.0).to(content_mask.dtype) # B x HW
                pos_mask = pos_mask.view(-1, h, w) # B x H x W
                pos_mask.masked_fill_(content_mask, False) # B x H x W
                
                # import pdb; pdb.set_trace()

                # update token and mask
                content_token = content_token * (~pos_mask) + sample * pos_mask
                content_mask = content_mask + pos_mask
                
                # update featuer
                # import pdb; pdb.set_trace()
                sample_feat = self.content_codec.get_codebook_entry_with_token(sample)['feature'] # B x C' x H x W
                sample_feat = self.emb_proj(sample_feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) # B x C x H x W
                sample_feat = sample_feat + pos_emb
                content_feat = content_feat * (~pos_mask.unsqueeze(dim=1)) + sample_feat * pos_mask.unsqueeze(dim=1)

            if calculate_acc_and_prob:
                pre_token = torch.argmax(logits, dim=-1, keepdim=False) # B x H x W
                acc = (pre_token == content_token_target).to(logits) # B x H x W
                acc = acc[pos_mask].tolist()
                acc_all += acc

                prob = logits.softmax(dim=-1) # B x H x W x Cls
                target_one_hot = F.one_hot(content_token_target, num_classes=prob.shape[-1])
                prob, _ = torch.max(prob * target_one_hot, dim=-1, keepdim=False) # B x H x W
                prob = prob[pos_mask].tolist()
                prob_all += prob
            
            if save_each_step_image:
                # import pdb; pdb.set_trace()
                self.content_codec.decoder.requires_image = False 
                self.content_codec.decoder.up_layer_with_image = False
                completed_iter = self.content_codec.decode(content_token, combine_rec_and_gt=False, token_shape=[h,w]) # B x C x H x W
                completed_iter = completed_iter[0].permute(1,2,0).to('cpu').numpy().astype(np.uint8)
                completed_iter = Image.fromarray(completed_iter)
                # save 
                _, index = torch.topk(pos_mask[0].long().view(-1), k=pos_mask[0].sum()) # HW 
                index = index.to('cpu').tolist()

                # plot the patch
                # import pdb; pdb.set_trace()
                token_count = int(content_mask[0].sum())
                if token_count == h*w: # finished
                    save_path = os.path.join('RESULT/debug', batch['relative_path'][0], 'completed_{}_{}_debug.png'.format(str(step).zfill(len(str(h*w))), token_count))
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    completed_iter.save(save_path)   

                im_draw = ImageDraw.ImageDraw(completed_iter)
                for idx in index:
                    r = idx // w
                    c = idx % w
                    y1 = r * patch_size 
                    x1 = c * patch_size
                    im_draw.rectangle(((x1, y1),(x1+patch_size, y1+patch_size)), fill=None, outline='yellow', width=2)
                                    
                save_path = os.path.join('RESULT/debug', batch['relative_path'][0], 'completed_{}_{}_{}.png'.format(str(step).zfill(len(str(h*w))), token_count, str(index)))
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                completed_iter.save(save_path)
                print('saved to {}'.format(save_path))

        accumulate_time['transformer'] = (accumulate_time['transformer'] * accumulate_time['count'] + time.time() - tic) / (accumulate_time['count']+1)
        # print('Time consumption: forward {}s/iter, sample {}s/token, sample {}s/img'.format(forward_time/len(step_nums), (time.time()-sample_time)/len(step_nums), time.time()-sample_time))
        assert content_mask.sum() == content_mask.numel(), "Unfinised: {} tokens are not predicted!".format(content_mask.numel() - content_mask.sum())

        if save_each_step_image:
            self.content_codec.decoder.requires_image = cache_decoder_requires_image
            self.content_codec.decoder.up_layer_with_image = cache_decoder_uo_layer_with_image


        # decode
        tic = time.time()
        masked_im = batch['image'] * batch['mask']
        sampled_im = self.content_codec.decode(content_token, mask_im=masked_im, mask=batch['mask'], token_shape=[h,w])
        self._phase5f_end_timing(total_inference_handle)
        
        accumulate_time['decoder'] = (accumulate_time['decoder'] * accumulate_time['count'] + time.time() - tic) / (accumulate_time['count']+1)
        accumulate_time['total'] = accumulate_time['encoder'] + accumulate_time['prepare'] + accumulate_time['transformer'] + accumulate_time['decoder']
        accumulate_time['count'] = accumulate_time['count'] + 1

        output = {
            'completed': sampled_im,
            'mask': batch['mask'],
            'masked_gt': masked_im,
            'accumulate_time': accumulate_time
        }
        current_token_count = int(content_shape[0] * content_shape[1])
        phase4a_original_tokens = int(self.safe_tome_profile.get('original_tokens', 0))
        phase4a_merged_tokens = int(self.safe_tome_profile.get('merged_tokens', 0))
        phase4a_restored_tokens = int(self.safe_tome_profile.get('restored_tokens', 0))
        phase4a_eligible_tokens = int(self.safe_tome_profile.get('eligible_tokens', 0))
        phase4a_removed_tokens = int(self.safe_tome_profile.get('removed_tokens', 0))
        if self.global_tome_r > 0 and not self.safe_tome_enabled:
            phase4a_original_tokens = int(self.global_tome_profile.get('original_tokens', 0))
            phase4a_merged_tokens = int(self.global_tome_profile.get('merged_tokens', 0))
            phase4a_restored_tokens = int(self.global_tome_profile.get('restored_tokens', 0))
            phase4a_eligible_tokens = int(self.global_tome_profile.get('original_tokens', 0))
            phase4a_removed_tokens = int(self.global_tome_profile.get('removed_tokens', 0))
        if phase4a_original_tokens <= 0:
            phase4a_original_tokens = current_token_count
        if phase4a_merged_tokens <= 0:
            phase4a_merged_tokens = current_token_count
        if phase4a_restored_tokens <= 0:
            phase4a_restored_tokens = current_token_count
        phase4a_timing_profile = self._phase4a_finalize_timing(
            token_profile={
                'original_tokens': phase4a_original_tokens,
                'eligible_tokens': phase4a_eligible_tokens,
                'removed_tokens': phase4a_removed_tokens,
                'merged_tokens': phase4a_merged_tokens,
                'restored_tokens': phase4a_restored_tokens,
            },
        )
        phase5f_timing_profile = self._phase5f_finalize_timing()
        if self.global_tome_r > 0:
            output['global_tome_profile'] = self.global_tome_profile
        if boundary_split is not None:
            output['boundary_split'] = boundary_split
            output['boundary_split_profile'] = self.boundary_split_profile
        if similarity_guidance is not None:
            if self.similarity_scorer_enabled:
                output['similarity_profile'] = self.similarity_profile
            if self.ga_spg_guidance_enabled:
                output['ga_spg_profile'] = self.ga_spg_profile
                if self.ga_spg_lite_mode and (self.safe_tome_debug_enabled or self.ga_spg_audit_enabled):
                    output['ga_spg_guidance_debug'] = self._compact_ga_spg_guidance_debug(similarity_guidance)
        if self.safe_tome_enabled:
            output['safe_tome_profile'] = self.safe_tome_profile
            if self.phase5k_audit_enabled:
                output['phase5k_audit_profile'] = self.phase5k_audit_profile
            if 'safe_tome_state' in locals() and safe_tome_state is not None and safe_tome_state.get('debug') is not None:
                output['safe_tome_debug'] = safe_tome_state['debug']
        if phase4a_timing_profile is not None:
            output['phase4a_timing_profile'] = phase4a_timing_profile
        if phase5f_timing_profile is not None:
            output['phase5f_timing_profile'] = phase5f_timing_profile
        if return_gt:
            output['input'] = batch['image']
        if return_mask_gt:
            output['mask_input'] = masked_im
        if return_reconstruction:
            token = self.content_codec.get_tokens(batch['image'], mask=batch['mask'])
            output['reconstruction'] = self.content_codec.decode(token['token'], token_shape=[h,w])
        if calculate_acc_and_prob:
            output['acc'] = torch.FloatTensor(acc_all).to(masked_im).mean()
            output['prob'] = torch.FloatTensor(prob_all).to(masked_im).mean()

        if save_each_step_image:      
            completed = output['completed'][0].permute(1,2,0).to('cpu').numpy().astype(np.uint8)
            completed = Image.fromarray(completed)
            save_path = os.path.join('RESULT/debug', batch['relative_path'][0], 'completed_{}_{}.png'.format(str(step).zfill(len(str(h*w))), token_count))
            completed.save(save_path)

            mask = mask_[0][0].to('cpu').numpy().astype(np.uint8)
            mask = Image.fromarray(mask * 255)
            save_path = os.path.join('RESULT/debug', batch['relative_path'][0], 'mask_{}_{}.png'.format(str(step).zfill(len(str(h*w))), token_count))
            mask.save(save_path)


            # merge
            merge = output['completed'] * (1 - batch['mask'].float()) + batch['image'] * batch['mask'].float()
            merge = merge[0].permute(1,2,0).to('cpu').numpy().astype(np.uint8)
            merge = Image.fromarray(merge)
            save_path = os.path.join('RESULT/debug', batch['relative_path'][0], 'completed_merge_{}_{}.png'.format(str(step).zfill(len(str(h*w))), token_count))
            merge.save(save_path)
            
            self.train()
            return None

        self.train()
        return output
    
    def parameters(self, recurse=True, name=None):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # return super().parameters(recurse=True)
        if name is None or name == 'none':
            return super().parameters(recurse=recurse)
        else:
            # separate out all parameters to those that will and won't experience regularizing weight decay
            print("Transformer: get parameters by the overwrite method!")
            if self.init_type == 'beit':
                decay = set()
                no_decay = set()
                whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d) #TODO(torch.nn.Linear, )
                blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
                for mn, m in self.named_modules():
                    for pn, p in m.named_parameters():
                        if not p.requires_grad:
                            continue 

                        fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                        if pn.endswith('bias'):
                            # all biases will not be decayed
                            no_decay.add(fpn)
                        elif pn.endswith('_param'):
                            no_decay.add(fpn)
                        elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                            # weights of whitelist modules will be weight decayed
                            decay.add(fpn)
                        elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                            # weights of blacklist modules will NOT be weight decayed
                            no_decay.add(fpn)
                no_decay.add('pos_emb')
                if self.mask_emb is not None:
                    no_decay.add('mask_emb')
            elif self.init_type == 'mae':
                no_decay = set(['pos_emb'])
                if self.mask_emb is not None:
                    no_decay.add('mask_emb')
                decay = set()
                for mn, m in self.named_modules():
                    for pn, p in m.named_parameters():
                        if not p.requires_grad:
                            continue 
                        fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                        if pn.endswith('_param'):
                            no_decay.add(fpn)
                            continue
                        if fpn in no_decay:
                            continue
                        decay.add(fpn)
            else:
                raise NotImplementedError('init type: {} not implemented!'.format(self.init_type))

            # validate that we considered every parameter
            param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad} 
            inter_params = decay & no_decay
            union_params = decay | no_decay
            assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
            assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                        % (str(param_dict.keys() - union_params), )

            # create the pytorch optimizer object
            optim_groups = [
                {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.weight_decay},
                {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
            ]
            return optim_groups

    def prepare_data(self, image, mask):
        """
        Get the feature from image

        Args:
            image: B x 3 x H x W
            mask: B x 1 x H x W
        """
        data_mask = self.content_codec.get_features(
                image, 
                mask=mask, 
                return_quantize_feature=True,
                return_token=False,
                mask_pixel_value=self.mask_pixel_value)

        if self.input_feature_type == 'origin':
            feat_mask = data_mask['feature'] # B x C x H x W
            b, _, h, w = feat_mask.shape
            # random change origin feature with quantized feature
            token_type, unmask_ratio = get_token_type(mask, type='pixel_shuffle', token_shape=[h,w]) # B x 1 x H/cps x W/cps
            valid_token_mask = token_type == 1
            if self.random_quantize > 0:
                quantize_mask = torch.rand(*token_type.shape).to(token_type.device) < self.random_quantize # B x 1 x H x W, in range [0, 1)
                # only quantize those unmasked tokens
                quantize_mask = (valid_token_mask * quantize_mask).to(feat_mask.dtype) # 1 denotes to be quantized
                feat_mask = feat_mask * (1-quantize_mask) +  (data_mask['feature_quantize'] * quantize_mask) # B x C x H x W
        elif self.input_feature_type == 'quantized':
            feat_mask = data_mask['feature_quantize'] # B x C x H x W
            b, _, h, w = feat_mask.shape
            token_type, unmask_ratio = get_token_type(mask, type='pixel_shuffle', token_shape=[h,w]) # B x 1 x H x W
            valid_token_mask = token_type == 1
        else:
            raise NotImplementedError('input feature type {} is not impleted!'.format(self.input_feature_type))

        # import pdb; pdb.set_trace()
        feat_mask = self.emb_proj(feat_mask.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) # B x C x H x W
        if self.mask_emb is not None:
            # import pdb; pdb.set_trace()
            feat_mask = feat_mask * unmask_ratio + self.mask_emb * (1-unmask_ratio)
        


        






        # add position embedding
        content_shape = (h // self.content_patch_token_shape[0], w // self.content_patch_token_shape[1])
        if self.pos_emb is not None:
            pos_emb = self.pos_emb.permute(0, 2, 1).view(1, -1, content_shape[0], content_shape[1]) # B x D x H/cps x W/cps
            pos_emb = pixel_shuffle(pos_emb, out_size=(h, w), chunked=True) # B x C x H x W
            feat_mask = feat_mask + pos_emb 
        projected_feat_mask = feat_mask
        
        if self.content_patch_token_shape != (1, 1):
            feat_mask = pixel_unshuffle(feat_mask, out_size=content_shape, chunked=True) # B x D x H/cps x W/cps
            valid_token_mask = get_token_type(mask, type='pixel_shuffle', token_shape=content_shape)[0] == 1 # B x 1 x H/cps x W/cps
        
        # reshape the data
        feat_mask = feat_mask.permute(0, 2, 3, 1).contiguous() # B x H/cps x W/cps x D
        valid_token_mask = valid_token_mask.squeeze(dim=1).contiguous() # B x H/cps x W/cps
        token_type = token_type.squeeze(dim=1).contiguous() # B x H x W
        unmask_ratio = unmask_ratio.squeeze(dim=1).contiguous()  # B x H/cps x W/cps
        boundary_split = self._boundary_split(mask, feat_mask.shape[1:3])
        similarity_guidance = None
        if (self.similarity_scorer_enabled or self.ga_spg_guidance_enabled) and boundary_split is not None:
            similarity_guidance_feat = self._build_similarity_guidance_feature(
                data_mask=data_mask,
                projected_feature=projected_feat_mask,
                content_shape=content_shape,
            )
            similarity_guidance = self._compute_safe_tome_guidance(similarity_guidance_feat, boundary_split)

        # prepare target
        data_target = self.content_codec.get_features(image, return_token=True, return_distance=True)

        output = {
            'token_target': data_target['token'].contiguous(), # B x H x W
            'token_type': token_type.contiguous(), # B x H x W
            'unmask_ratio': unmask_ratio,  # B x H x W

            'token_mask': valid_token_mask.contiguous(), # B x H/cps x W/cps
            'embedding': feat_mask # B x H/cps x W/cps x D
        }
        if boundary_split is not None:
            output['boundary_split'] = boundary_split
            output['boundary_split_profile'] = self.boundary_split_profile
        if similarity_guidance is not None:
            output['similarity_guidance'] = similarity_guidance
            if self.similarity_scorer_enabled:
                output['similarity_profile'] = self.similarity_profile
            if self.ga_spg_guidance_enabled:
                output['ga_spg_profile'] = self.ga_spg_profile
        
        if isinstance(self.loss_func, LabelSmoothingLoss):
            output['token_distance'] = data_target['distance'][:,:,:,:self.num_cls].contiguous() # B x H x W x Cls
            
        return output


    def forward(
            self, 
            batch, 
            return_loss=False, 
            return_logits=True, 
            return_att_weight=False,
            **kwargs):
        self._safe_tome_selection_cache_signature = None
        self._safe_tome_selection_cache = None
        self._safe_tome_runtime_cache_signature = None
        self._safe_tome_runtime_cache = None

        # 1) get data from input data
        if batch.get('count_flops', False):
            data = batch
        else:
            data = self.prepare_data(batch['image'], mask=batch['mask'])
        emb = data['embedding']

        # 2) forward in transformer
        if self.drop is not None:
            emb = self.drop(emb)
        if self.attn_content_with_mask:
            attn_mask = data['token_mask']
        else:
            attn_mask = None
        safe_tome_state = None
        if self.safe_tome_enabled:
            if attn_mask is not None:
                raise RuntimeError('PUT_SAFE_TOME_R currently supports attn_content_with_mask=False only.')
            safe_tome_state = self._build_safe_tome_state(
                emb,
                data.get('boundary_split'),
                data.get('similarity_guidance'),
            )
            emb = safe_tome_state['compressed_embedding']
            attn_mask = safe_tome_state['compressed_mask']
        for block_idx in range(len(self.blocks)):   
            emb, att_weight = self.blocks[block_idx](emb, mask=attn_mask) # B x H/cps x W/cps x D, B x H/cps x W/cps x H/cps x W/cps
        if safe_tome_state is not None:
            emb = self._restore_safe_tome_embedding(emb, safe_tome_state)
        
        # 3) get logits
        emb = self.norm(emb)
        if self.content_patch_token_shape != (1, 1):
            content_shape = (data['token_target'].shape[-2], data['token_target'].shape[-1])
            emb = emb.permute(0, 3, 1, 2) # B x D x H/cps*W/cps
            emb = pixel_shuffle(emb, out_size=content_shape, chunked=True) # B x D/cps^2 x H x W
            emb = emb.permute(0, 2, 3, 1) # B x H x W x D/cps^2
            logits = self.to_logits(emb) # B x H x W x n
        else:
            logits = self.to_logits(emb) # B x H x W x n
        # import pdb; pdb.set_trace()

        # 4) get output, especially loss
        out = {}

        if return_logits:
            out['logits'] = logits
        if return_att_weight:
            out['attention_weight'] = att_weight
        if data.get('boundary_split_profile') is not None:
            out['boundary_split_profile'] = data['boundary_split_profile']
        if data.get('similarity_profile') is not None:
            out['similarity_profile'] = data['similarity_profile']
        if data.get('ga_spg_profile') is not None:
            out['ga_spg_profile'] = data['ga_spg_profile']
        if self.safe_tome_enabled:
            out['safe_tome_profile'] = self.safe_tome_profile

        if return_loss:
            token_target = data['token_target'] # B x H x W
            # print(token_target.max())
            token_type = data['token_type'] # B x H x W
            # import pdb; pdb.set_trace()

            if self.loss_func is None:
                loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), token_target.view(-1), ignore_index=self.content_ignore_token, reduction='none')
                loss = loss.view(token_type.shape)
                loss_out = {'loss': loss}
            elif isinstance(self.loss_func, PolyLoss):
                loss_out = self.loss_func(logits=logits.permute(0, 3, 1, 2), labels=token_target, mask=None, reduction='none')
            elif isinstance(self.loss_func, LabelSmoothingLoss):
                loss_out = self.loss_func(logits=logits.permute(0, 3, 1, 2), labels=data['token_distance'].permute(0,3,1,2), mask=None, reduction='none')
            else:
                raise NotImplementedError

            # get the predicted probabilities
            prob = F.softmax(logits, dim=-1)# B x H x W x n
            gt = F.one_hot(token_target, num_classes=prob.shape[-1]) # B x H x W x n
            gt_prob = (prob * gt).sum(-1) # B x H x W
            loss_out['pred_prob'] = gt_prob.detach()

            # get prediction accuracy
            pred = torch.argmax(logits, dim=-1) # B x H x W
            right_or_wrong = pred == token_target # B x H x W
            loss_out['pred_acc'] = right_or_wrong.detach()

            if self.loss_mask_type == 'binary':
                loss_mask_overall = token_type!=1
            elif self.loss_mask_type == 'mask_ratio':
                loss_mask_overall = 1 - data['unmask_ratio']
            else:
                raise NotImplementedError

            loss_mask = {
                'placeholder': loss_mask_overall, # B x H x W
                'partial': token_type == 2, # partially masked
                'fully': token_type == 0, # fully masked
            }

            for loss_k in loss_out:
                for mask_k in loss_mask:
                    if mask_k != 'placeholder':
                        out_k = mask_k + '_' + loss_k
                    else:
                        out_k = loss_k
                    out[out_k] = (loss_out[loss_k] * loss_mask[mask_k]).sum() / (loss_mask[mask_k].sum() + 1e-18)
        return out
