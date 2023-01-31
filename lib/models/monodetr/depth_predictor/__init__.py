from .depth_predictor import DepthPredictor
from .depth_predictor_residual import DepthPredictorResidual


def build_depth_predictor(model_cfg):
    if model_cfg.get('with_depth_residual'):
        return DepthPredictorResidual(model_cfg)
    return DepthPredictor(model_cfg)
