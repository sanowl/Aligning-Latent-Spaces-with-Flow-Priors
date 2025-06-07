import sys, os
import types
if "numpy" not in sys.modules:
    sys.modules["numpy"] = types.ModuleType("numpy")
if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.distributed = types.ModuleType("distributed")
    sys.modules["torch"] = torch_stub
    sys.modules["torch.distributed"] = torch_stub.distributed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import tempfile
import os
import builtins

import continuous_tokenizer.utils.misc as misc
import continuous_tokenizer.utils.optim as optim


class DummyParam:
    def __init__(self, ndim, requires_grad=True):
        self.ndim = ndim
        self.requires_grad = requires_grad


class DummyModel:
    def __init__(self, params):
        self._params = params

    def named_parameters(self):
        for name, param in self._params.items():
            yield name, param


def test_str2bool():
    assert misc.str2bool('yes') is True
    assert misc.str2bool('No') is False
    assert misc.str2bool(True) is True
    try:
        misc.str2bool('maybe')
    except Exception as e:
        assert isinstance(e, Exception)
    else:
        assert False, 'Expected exception for invalid boolean string'


def test_load_model_state_dict():
    state = {
        'module.layer.weight': 1,
        '_orig_mod.bias': 2,
        'other': 3,
    }
    out = misc.load_model_state_dict(state)
    assert out['layer.weight'] == 1
    assert out['module.layer.weight'] == 1
    assert out['bias'] == 2
    assert out['other'] == 3


def test_manage_checkpoints(tmp_path):
    # create some fake checkpoint files
    for i in range(1, 6):
        (tmp_path / f"{i}.pt").touch()
    (tmp_path / "best_ckpt.pt").touch()
    misc.manage_checkpoints(str(tmp_path), keep_last_n=2)
    remaining = sorted(p.name for p in tmp_path.glob('*.pt'))
    assert remaining == ['3.pt', '4.pt', '5.pt', 'best_ckpt.pt']


def test_param_groups_weight_decay():
    params = {
        'layer.weight': DummyParam(2),
        'layer.bias': DummyParam(1),
        'norm.weight': DummyParam(1),
        'conv.weight': DummyParam(2),
        'frozen.weight': DummyParam(2, requires_grad=False),
    }
    model = DummyModel(params)
    groups = optim.param_groups_weight_decay(
        model, weight_decay=0.1, no_weight_decay_list=('layer.weight',)
    )
    no_decay_params = groups[0]['params']
    decay_params = groups[1]['params']
    assert params['layer.weight'] in no_decay_params
    assert params['layer.bias'] in no_decay_params
    assert params['norm.weight'] in no_decay_params
    assert params['conv.weight'] in decay_params
    assert params['frozen.weight'] not in decay_params
    assert params['frozen.weight'] not in no_decay_params
