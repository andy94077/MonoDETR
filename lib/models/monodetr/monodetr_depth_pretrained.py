"""
MonoDETR: Depth-aware Transformer for Monocular 3D Object Detection
"""
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
from torch.types import Number
import torch.distributed as dist
from lib.models.monodetr.depth_predictor.ddn_loss.balancer import Balancer
from lib.models.monodetr.depth_predictor.ddn_loss.focalloss import focal_loss, regression_focal_loss

from utils import box_ops, depth_utils, misc
from utils.misc import (NestedTensor, get_world_size, is_dist_avail_and_initialized)

from .backbone import build_backbone
from .depth_predictor import build_depth_predictor
from .depth_predictor.ddn_loss import DDNLoss, DDNWithResidualLoss, DDNWithWeightedDepthLoss


class MonoDepthPretrained(nn.Module):
    """ This is the MonoDETR module that performs monocualr 3D object detection """

    def __init__(self, backbone, depth_predictor, hidden_dim, num_feature_levels, with_depth_residual=True):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            depthaware_transformer: depth-aware transformer architecture. See depth_aware_transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For KITTI, we recommend 50 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            with_depth_residual: predicts depth map residual
            two_stage: two-stage MonoDETR
        """
        super().__init__()

        self.depth_predictor = depth_predictor
        self.num_feature_levels = num_feature_levels

        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        self.backbone = backbone
        self.with_depth_residual = with_depth_residual

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def forward(self, images, calibs, img_sizes, targets) -> Dict[str, Union[torch.Tensor, List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]]:
        """The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        Returns:
            A dict of Tensors with key:
            * pred_logits: predicted class logits with shape [batch, num_boxes, num_classes]
            * pred_boxes: predicted normalized 3D bbox (3d_cx, 3d_cy, l, r, t, b) with shape
                [batch, num_boxes, 6]. Each element is in [0, 1].
            * pred_3d_dim: predicted 3D bbox dimension (x, y, z) - mean_size with shape
                [batch, num_boxes, 3].
            * pred_depth: predicted depth for each 3D bbox with shape [batch, num_boxes]
            * pred_angle: predicted angle class logits and offsets with shape
                [batch, num_boxes, 24]. 12 for classes and 12 for class offsets.
            * pred_depth_map_logits: predicted depth map logits with shape
                [batch, num_depth_bins, H, W].
        """
        features, pos = self.backbone(images, calibs)

        srcs = []
        masks = []
        for lvl, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[lvl](src))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for lvl in range(_len_srcs, self.num_feature_levels):
                if lvl == _len_srcs:
                    src = self.input_proj[lvl](features[-1].tensors)
                else:
                    src = self.input_proj[lvl](srcs[-1])
                m = torch.zeros(src.shape[0], src.shape[2], src.shape[3]).to(torch.bool).to(src.device)
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask), calibs, images.shape[-2:]).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        out: Dict[str, Union[torch.Tensor, List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]] = {}
        kwargs_dict = {
            'calibs': calibs,
            'targets': targets,
        }
        if self.with_depth_residual:
            pred_depth_map_logits, depth_pos_embed, weighted_depth, pred_depth_residual = self.depth_predictor(srcs, masks[1], pos[1], **kwargs_dict)
            out['pred_depth_residual'] = pred_depth_residual
        else:
            pred_depth_map_logits, depth_pos_embed, weighted_depth = self.depth_predictor(srcs, masks[1], pos[1], **kwargs_dict)
        out['pred_depth_map_logits'] = pred_depth_map_logits
        out['weighted_depth'] = weighted_depth

        return out


class SetCriterion(nn.Module):
    """ This class computes the loss for MonoDETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self,
                 num_classes: int,
                 weight_dict: Dict[str, float],
                 focal_alpha: float,
                 losses: List[str],
                 depth_min: float = 1e-3,
                 depth_max: float = 60,
                 num_depth_bins: int = 80,
                 use_gt_depth_map: Optional[bool] = False,
                 ):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.weight_dict = weight_dict
        self.loss_names = losses
        self.focal_alpha = focal_alpha
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.num_depth_bins = num_depth_bins
        self.use_gt_depth_map = use_gt_depth_map
        self.depth_bin_values = nn.parameter.Parameter(
            depth_utils.get_depth_bin_values(depth_min, depth_max, num_depth_bins),
            requires_grad=False)

        self.ddn_loss = DDNLoss(alpha=self.focal_alpha,
                                depth_min=self.depth_min,
                                depth_max=self.depth_max,
                                num_depth_bins=self.num_depth_bins)  # for depth map
        self.ddn_with_residual_loss = DDNWithResidualLoss(alpha=self.focal_alpha,
                                                          depth_min=self.depth_min,
                                                          depth_max=self.depth_max,
                                                          num_depth_bins=self.num_depth_bins)  # for depth map with residual
        self.ddn_with_weighted_depth_loss = DDNWithWeightedDepthLoss(alpha=self.focal_alpha,
                                                                     depth_min=self.depth_min,
                                                                     depth_max=self.depth_max,
                                                                     num_depth_bins=self.num_depth_bins)  # for depth map with residual
        self.balancer = Balancer(fg_weight=13., bg_weight=1., downsample_factor=1)

    def loss_depth_map(self,
                       outputs: Dict[str, torch.Tensor],
                       targets: List[Dict[str, torch.Tensor]],
                       indices: List[Tuple[torch.Tensor, torch.Tensor]],
                       num_boxes: int,
                       **kwargs) -> torch.Tensor:
        depth_map_logits = outputs['pred_depth_map_logits']

        num_gt_per_img = [len(t['boxes']) for t in targets]
        gt_boxes2d = torch.cat([t['boxes'] for t in targets], dim=0) * depth_map_logits.new_tensor([80, 24, 80, 24])
        gt_boxes2d = box_ops.box_cxcywh_to_xyxy(gt_boxes2d)
        gt_center_depth = torch.cat([t['depth'] for t in targets], dim=0).squeeze(dim=1)

        depth_map_loss = self.ddn_loss(
            depth_map_logits, gt_boxes2d, num_gt_per_img, gt_center_depth)
        return depth_map_loss

    def loss_depth_map_with_residual(self,
                                     outputs: Dict[str, torch.Tensor],
                                     targets: List[Dict[str, torch.Tensor]],
                                     indices: List[Tuple[torch.Tensor, torch.Tensor]],
                                     num_boxes: int,
                                     **kwargs) -> Dict[str, torch.Tensor]:
        depth_map_logits = outputs['pred_depth_map_logits']
        depth_residual = outputs['pred_depth_residual']

        num_gt_per_img = [len(t['boxes']) for t in targets]
        gt_boxes2d = torch.cat([t['boxes'] for t in targets], dim=0) * depth_map_logits.new_tensor([80, 24, 80, 24])
        gt_boxes2d = box_ops.box_cxcywh_to_xyxy(gt_boxes2d)
        gt_center_depth = torch.cat([t['depth'] for t in targets], dim=0).squeeze(dim=1)

        depth_map_loss, depth_residual_loss = self.ddn_with_residual_loss(
            depth_map_logits, depth_residual, gt_boxes2d, num_gt_per_img, gt_center_depth)
        return {
            'loss_depth_map': depth_map_loss,
            'loss_depth_residual': depth_residual_loss,
        }

    def loss_depth_map_residual(self,
                                outputs: Dict[str, torch.Tensor],
                                targets: List[Dict[str, torch.Tensor]],
                                indices: List[Tuple[torch.Tensor, torch.Tensor]],
                                num_boxes: int,
                                **kwargs) -> Dict[str, torch.Tensor]:
        # [batch, num_depth_bins, depth_map_H, depth_map_W]
        depth_map_logits = outputs['pred_depth_map_logits']
        # [batch, num_depth_bins, depth_map_H, depth_map_W]
        depth_residual = outputs['pred_depth_residual']

        # [batch, depth_map_H, depth_map_W]
        gt_depth_map_values = depth_utils.get_gt_depth_map_values(depth_map_logits, targets, self.depth_max)
        depth_target = depth_utils.bin_depths(gt_depth_map_values, depth_min=self.depth_min, depth_max=self.depth_max, num_bins=self.num_depth_bins, target=True)
        # [batch, depth_map_H, depth_map_W]
        gt_weighted_depth = self.depth_bin_values[depth_target]
        gt_depth_residual = gt_depth_map_values - gt_weighted_depth

        num_gt_per_img = [len(t['boxes']) for t in targets]
        gt_boxes2d = torch.cat([t['boxes'] for t in targets], dim=0) * depth_map_logits.new_tensor([80, 24, 80, 24])
        gt_boxes2d = box_ops.box_cxcywh_to_xyxy(gt_boxes2d)

        depth_residual_value = depth_residual.gather(dim=1, index=depth_target.unsqueeze(1)).squeeze()
        depth_residual_loss = F.l1_loss(depth_residual_value, gt_depth_residual, reduction='none')
        depth_residual_loss = self.ddn_loss.balancer(loss=depth_residual_loss, gt_boxes2d=gt_boxes2d, num_gt_per_img=num_gt_per_img)
        return {'loss_depth_residual': depth_residual_loss}

    def loss_weighted_depth(self,
                            outputs: Dict[str, torch.Tensor],
                            targets: List[Dict[str, torch.Tensor]],
                            indices: List[Tuple[torch.Tensor, torch.Tensor]],
                            num_boxes: int,
                            **kwargs) -> Dict[str, torch.Tensor]:
        # [batch, num_depth_bins, depth_map_H, depth_map_W]
        depth_map_logits = outputs['pred_depth_map_logits']
        # [batch, depth_map_H, depth_map_W]
        weighted_depth = outputs['weighted_depth']
        weighted_depth = torch.broadcast_to(weighted_depth.unsqueeze(1), depth_map_logits.shape)

        num_gt_per_img = [len(t['boxes']) for t in targets]
        gt_boxes2d = torch.cat([t['boxes'] for t in targets], dim=0) * depth_map_logits.new_tensor([80, 24, 80, 24])
        gt_boxes2d = box_ops.box_cxcywh_to_xyxy(gt_boxes2d)
        if self.use_gt_depth_map:
            gt_depth_map_values = torch.stack([t['depth_map'] for t in targets])
            gt_depth_indices = depth_utils.bin_depths(gt_depth_map_values, depth_min=self.depth_min, depth_max=self.depth_max, num_bins=self.num_depth_bins, target=True)
            depth_map_loss = focal_loss(depth_map_logits, gt_depth_indices, alpha=self.focal_alpha, reduction='none')
            depth_map_loss = self.balancer(loss=depth_map_loss, gt_boxes2d=gt_boxes2d, num_gt_per_img=num_gt_per_img)
            weighted_depth_loss = regression_focal_loss(depth_map_logits, weighted_depth, gt_depth_indices, gt_depth_map_values, self.focal_alpha, reduction='none')
            weighted_depth_loss = self.balancer(loss=weighted_depth_loss, gt_boxes2d=gt_boxes2d, num_gt_per_img=num_gt_per_img)
        else:
            gt_center_depth = torch.cat([t['depth'] for t in targets], dim=0).squeeze(dim=1)
            depth_map_loss, weighted_depth_loss = self.ddn_with_weighted_depth_loss(
                depth_map_logits, weighted_depth, gt_boxes2d, num_gt_per_img, gt_center_depth)
        return {
            'loss_depth_map': depth_map_loss,
            'loss_weighted_depth': weighted_depth_loss,
        }

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs) -> Dict[str, torch.Tensor]:

        loss_map = {
            'loss_depth_map': self.loss_depth_map,
            'loss_depth_map_residual': self.loss_depth_map_residual,
            'loss_depth_map_with_residual': self.loss_depth_map_with_residual,
            'loss_weighted_depth': self.loss_weighted_depth,
        }

        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        loss_dict = loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
        if isinstance(loss_dict, torch.Tensor):
            loss_dict = {loss: loss_dict}
        if not isinstance(loss_dict, dict):
            raise TypeError(f'Invalid loss return type {type(loss_dict)}. Expected "Dict[str, torch.Tensor]" | "torch.Tensor".')
        return loss_dict

    def forward(self, outputs, targets) -> Tuple[Dict[str, torch.Tensor], Dict[str, Number]]:
        """This performs the loss computation.

        Args:
            outputs: dict of tensors, see the output specification of the model for the format
            targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc

        Returns:
            losses: A dict of weighted loss tensors.
            unweighted_losses_log_dict: A dict of unweighted loss numbers for logging purposes only.
        """
        device = next(iter(outputs.values())).device
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = torch.tensor([len(t["labels"]) for t in targets], dtype=torch.float, device=device).sum()
        if is_dist_avail_and_initialized():
            dist.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        unweighted_losses_log_dict = {}
        losses = {}
        # Compute all the requested losses
        for loss_name in self.loss_names:
            loss_dict = self.get_loss(loss_name, outputs, targets, indices=None, num_boxes=num_boxes)
            for key, loss_val in loss_dict.items():
                losses[key] = loss_val * self.weight_dict[key]
                unweighted_losses_log_dict[key] = loss_val

        unweighted_losses_log_dict = misc.reduce_dict(unweighted_losses_log_dict)
        unweighted_losses_log_dict = {loss_name: loss.item() for loss_name, loss in unweighted_losses_log_dict.items()}

        return losses, unweighted_losses_log_dict


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(model_cfg, loss_cfg):
    # backbone
    backbone = build_backbone(model_cfg)

    # depth prediction module
    depth_predictor = build_depth_predictor(model_cfg)

    model = MonoDepthPretrained(
        backbone,
        depth_predictor,
        hidden_dim=model_cfg['hidden_dim'],
        num_feature_levels=model_cfg['num_feature_levels'],
        with_depth_residual=model_cfg.get('with_depth_residual'),
    )

    # loss
    weight_dict = loss_cfg['weights']

    criterion = SetCriterion(
        model_cfg['num_classes'],
        weight_dict=weight_dict,
        focal_alpha=loss_cfg['focal_alpha'],
        losses=loss_cfg['losses'],
        depth_min=model_cfg['depth_min'],
        depth_max=model_cfg['depth_max'],
        num_depth_bins=model_cfg['num_depth_bins'],
        use_gt_depth_map=loss_cfg.get('use_gt_depth_map'))

    return model, criterion
