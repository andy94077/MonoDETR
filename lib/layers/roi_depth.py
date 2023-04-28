import torch
from torchvision.ops import roi_align
from torch import nn

from utils import box_ops


class RoIDepth(nn.Module):
    def __init__(self, grid_H: int = 5, grid_W: int = 7):
        super().__init__()
        self.grid_H = grid_H
        self.grid_W = grid_W

    def forward(self,
                depth_feat: torch.Tensor,
                coords2d: torch.Tensor,
                num_gt_per_img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            depth_feat: A tensor with shape [batch, num_depth_bins, H, W].
            coords2d: A tensor of 2D bboxes (cx, cy, w, h) with shape [num_boxes, 4]. Each element is in [0, 1].
            num_gt_per_img: A tensor representing the number of ground truths for each image with shape [batch,].
        Returns:
            RoI aligned depth map for each 2D bbox with shape [num_boxes, num_depth_bins, self.grid_H, self.grid_W].
        """
        batch, _, H, W = depth_feat.shape

        # [num_boxes, 4]
        bboxes = box_ops.box_cxcywh_to_xyxy(coords2d)
        bboxes = bboxes.clamp(0, 1) * bboxes.new_tensor([W, H, W, H])

        # [num_boxes, 1]
        batch_idx = torch.arange(batch, device=bboxes.device).repeat_interleave(num_gt_per_img).view(-1, 1)
        # [num_boxes, 5]
        bboxes_with_batch_idx = torch.cat([batch_idx, bboxes], dim=-1)

        # [num_boxes, num_depth_bins, self.grid_H, self.grid_W]
        roi_aligned_depth_map = roi_align(depth_feat, bboxes_with_batch_idx.detach(), (self.grid_H, self.grid_W), aligned=True)
        return roi_aligned_depth_map


_AVAILABLE_ROI_DEPTH_LAYERS = {
    'RoIDepth': RoIDepth,
}


def build_roi_depth_layer(cfg):
    assert 'roi_depth_layer' in cfg
    roi_depth_layer_type: str = cfg['roi_depth_layer'].pop('type', 'RoIDepth')
    assert roi_depth_layer_type in _AVAILABLE_ROI_DEPTH_LAYERS, (
        f'Invalid bbox_coder type {roi_depth_layer_type}. Supported bbox_coder types are {list(_AVAILABLE_ROI_DEPTH_LAYERS.keys())}.')
    return _AVAILABLE_ROI_DEPTH_LAYERS[roi_depth_layer_type](**cfg['roi_depth_layer'])
