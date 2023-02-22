from lib.models.monodetr import AVAILABLE_MODELS


def build_model(model_cfg, loss_cfg):
    model_type: str = model_cfg.pop('type', 'MonoDETR')
    assert model_type in AVAILABLE_MODELS, f'Invalid model type {model_type}. Supported model types are {list(AVAILABLE_MODELS.keys())}.'
    return AVAILABLE_MODELS[model_type](model_cfg, loss_cfg)
