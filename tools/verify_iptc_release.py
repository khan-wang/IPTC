"""Standard-library validation; does not run model inference."""
import ast
import hashlib
import json
from pathlib import Path
import runpy
import os
import sys
import types
import importlib

root = Path(__file__).resolve().parents[1]
records = json.loads((root/'iptc/source_hashes.json').read_text())
for row in records:
    path = root/row['module']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['release_sha256'], path
    ast.parse(path.read_text(encoding='utf-8'))
for path in (root/'iptc').glob('*.py'):
    ast.parse(path.read_text(encoding='utf-8'))
    assert '/home/' not in path.read_text(encoding='utf-8')
namespace = runpy.run_path(str(root/'iptc/__init__.py'))
old = dict(os.environ)
try:
    os.environ['PUT_UNRELATED_TEST_FLAG'] = '1'
    namespace['configure_environment']()
    assert 'PUT_UNRELATED_TEST_FLAG' not in os.environ
    assert os.environ['PUT_EXPERIMENT_DIRICHLET_MU'] == '0.10'
    assert os.environ['PUT_BOUNDARY_RING_RADIUS'] == '1'
finally:
    os.environ.clear()
    os.environ.update(old)
# Import/lifecycle smoke without allocating tensors or needing a checkpoint.
torch = types.ModuleType('torch')
nn = types.ModuleType('torch.nn')
functional = types.ModuleType('torch.nn.functional')
class Generator:
    def __init__(self, **kwargs):
        pass
    def manual_seed(self, seed):
        return self
torch.Generator = Generator
torch.nn = nn
nn.functional = functional
sys.modules.update({'torch':torch, 'torch.nn':nn, 'torch.nn.functional':functional,
                    'numpy':types.ModuleType('numpy')})
sys.path.insert(0, str(root))
api = importlib.import_module('iptc')
def original(*args, **kwargs):
    return None
model = types.SimpleNamespace(training=False, blocks=[], safe_tome_r=224,
    safe_tome_score_mode='dirichlet', safe_tome_debug_enabled=False,
    ga_spg_audit_enabled=False, _ga_spg_guidance=original,
    _build_safe_tome_state=original, _build_experimental_pair_metric=original,
    _safe_tome_selected_count=original, _safe_tome_selection_score=original)
with api.inference_context(model, 'sample.png', 1):
    assert hasattr(model, '_live_tail_forward')
    assert model._build_safe_tome_state is not original
    try:
        with api.inference_context(model, 'nested.png', 1):
            pass
        raise AssertionError('nested context was accepted')
    except RuntimeError:
        pass
assert not hasattr(model, '_live_tail_forward')
assert not hasattr(model, '_iptc_active')
assert model._build_safe_tome_state is original
assert model._ga_spg_guidance is original
print('PASS: hashes, syntax, configuration, package imports and mocked context lifecycle')
