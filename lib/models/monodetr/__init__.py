# from .monodetr import build
from typing import Any, Dict
from . import monodetr
from . import monooqdd

_AVAILABLE_MODELS = {
    'monodetr': monodetr.build,
    'monooqdd': monooqdd.build,
}


def build_monodetr(model_cfg: Dict[str, Any], loss_cfg: Dict[str, Any]):
    model_type: str = model_cfg.pop('type', 'monodetr')
    assert model_type in _AVAILABLE_MODELS, f'Invalid model type {model_type}. Supported model types are {list(_AVAILABLE_MODELS.keys())}.'
    return _AVAILABLE_MODELS[model_type](model_cfg, loss_cfg)
