"""Interface-Preserving Token Coarsening for the bundled PUT host."""
from contextlib import contextmanager
import os

METHOD = 'sdpa_lean_nullmask_align_race_cache_tail_dirichlet'


def configure_environment():
    """Call before constructing PUT; select the evaluated spatial-only path."""
    for key in list(os.environ):
        if key.startswith('PUT_'):
            del os.environ[key]
    os.environ.update({
        'PUT_BOUNDARY_SPLIT': '1', 'PUT_BOUNDARY_RING_RADIUS': '1',
        'PUT_SAFE_TOME_R': '224', 'PUT_GLOBAL_TOME_R': '0',
        'PUT_SAFE_TOME_DEBUG': '0', 'PUT_PHASE4A_TIMING': '0',
        'PUT_PHASE5F_TIMING': '0', 'PUT_PHASE5K_LEAN_PROFILE': '1',
        'PUT_GA_SPG_AUDIT': '0', 'PUT_GA_SPG_LITE_OPTIMIZED': '1',
        'PUT_GA_SPG_LITE_RUNTIME_CACHE': '0', 'PUT_GA_SPG_LITE_PAD_MODE': 'zero',
        'PUT_SIMILARITY_SCORER': '0', 'PUT_SAFE_TOME_SCORE_MODE': 'dirichlet',
        'PUT_EXPERIMENT_DIRICHLET_MU': '0.10',
    })


@contextmanager
def inference_context(model, sample_id, seed):
    """Install IPTC for ONE image; route state must not survive image changes.

    The model must be the bundled PUT overlay, in eval mode on CUDA, FP32,
    batch size one. Not thread-safe: use separate models for concurrent calls.
    """
    from .routing import Intervention
    if model.training:
        raise ValueError('Call model.eval() before IPTC inference')
    if getattr(model, '_iptc_active', False):
        raise RuntimeError('Nested or concurrent IPTC contexts are unsupported')
    required = ('_build_safe_tome_state', '_build_experimental_pair_metric')
    if not all(hasattr(model, name) for name in required):
        raise TypeError('IPTC requires the bundled PUT model overlay')
    state = Intervention(model, METHOD, {'sample_id': str(sample_id)}, int(seed), False)
    model._iptc_active = True
    try:
        state.install()
        yield model
    finally:
        try:
            state.remove()
        finally:
            del model._iptc_active
