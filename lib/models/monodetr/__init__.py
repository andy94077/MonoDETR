# from .monodetr import build
from typing import Any, Dict
from . import monodetr
from . import monooqdd
from . import monodepth_attention

AVAILABLE_MODELS = {
    'MonoDepthAttention': monodepth_attention.build,
    'MonoDETR': monodetr.build,
    'MonoOQDD': monooqdd.build,
}


def build_monodetr(model_cfg: Dict[str, Any], loss_cfg: Dict[str, Any]):
    return monodetr.build(model_cfg, loss_cfg)
