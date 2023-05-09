from .depth_predictor import DepthPredictor
from .depth_predictor_residual import DepthPredictorResidual
from .depth_predictor_residual_roi import DepthPredictorResidualRoI
from .dla_depth_predictor_residual_roi import DLADepthPredictorResidualRoI

_AVAILABLE_DEPTH_PREDICTORS = {
    'DepthPredictor': DepthPredictor,
    'DepthPredictorResidual': DepthPredictorResidual,
    'DepthPredictorResidualRoI': DepthPredictorResidualRoI,
    'DLADepthPredictorResidualRoI': DLADepthPredictorResidualRoI,
}


def build_depth_predictor(model_cfg):
    if 'depth_predictor' in model_cfg:
        model_type = model_cfg['depth_predictor'].pop('type')
        if model_type not in _AVAILABLE_DEPTH_PREDICTORS:
            raise NotImplementedError(
                f'Depth predictor type {model_type} is not supported. Available types are: {_AVAILABLE_DEPTH_PREDICTORS.keys()}')
        return _AVAILABLE_DEPTH_PREDICTORS[model_type](model_cfg)
    if model_cfg.get('with_depth_residual'):
        return DepthPredictorResidual(model_cfg)
    return DepthPredictor(model_cfg)
