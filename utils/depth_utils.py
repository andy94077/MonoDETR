import math
from typing import Dict, List
import torch
import torch.nn.functional as F

from utils import box_ops


def bin_depths(depth_map, mode="LID", depth_min: float = 1e-3, depth_max: float = 60, num_bins: int = 80, target: bool = False):
    """
    Converts depth map into bin indices
    Args:
        depth_map [torch.Tensor(H, W)]: Depth Map
        mode [string]: Discretiziation mode (See https://arxiv.org/pdf/2005.13423.pdf for more details)
            UD: Uniform discretiziation
            LID: Linear increasing discretiziation
            SID: Spacing increasing discretiziation
        depth_min [float]: Minimum depth value
        depth_max [float]: Maximum depth value
        num_bins [int]: Number of depth bins
        target [bool]: Whether the depth bins indices will be used for a target tensor in loss comparison
    Returns:
        indices [torch.Tensor(H, W)]: Depth bin indices
    """
    if mode == "UD":
        bin_size = (depth_max - depth_min) / num_bins
        indices = ((depth_map - depth_min) / bin_size)
    elif mode == "LID":
        bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
        indices = -0.5 + 0.5 * torch.sqrt(1 + 8 * (depth_map - depth_min) / bin_size)
    elif mode == "SID":
        indices = num_bins * (torch.log(1 + depth_map) - math.log(1 + depth_min)) / \
            (math.log(1 + depth_max) - math.log(1 + depth_min))
    else:
        raise NotImplementedError

    if target:
        # Remove indicies outside of bounds
        mask = (indices < 0) | (indices > num_bins) | (~torch.isfinite(indices))
        indices[mask] = num_bins

        # Convert to integer
        indices = indices.type(torch.int64)

    return indices


def get_depth_bin_values(depth_min: float, depth_max: float, num_depth_bins: int) -> torch.Tensor:
    bin_size = 2 * (depth_max - depth_min) / (num_depth_bins * (1 + num_depth_bins))
    bin_indice = torch.linspace(0, num_depth_bins - 1, num_depth_bins)
    bin_value = (bin_indice + 0.5).pow(2) * bin_size / 2 - bin_size / 8 + depth_min
    depth_bin_values = torch.cat([bin_value, bin_value.new_tensor([depth_max])], dim=0)
    return depth_bin_values


def get_gt_depth_map_values(depth_logits: torch.Tensor, targets: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
    num_gt_per_img = [len(t['boxes']) for t in targets]
    gt_boxes2d = torch.cat([t['boxes'] for t in targets], dim=0) * depth_logits.new_tensor([80, 24, 80, 24])
    gt_boxes2d = box_ops.box_cxcywh_to_xyxy(gt_boxes2d)
    gt_center_depth = torch.cat([t['depth'] for t in targets], dim=0).squeeze(dim=1)

    B, _, H, W = depth_logits.shape
    depth_maps = torch.zeros((B, H, W), device=depth_logits.device, dtype=depth_logits.dtype)

    # Set box corners
    gt_boxes2d[:, :2] = torch.floor(gt_boxes2d[:, :2])
    gt_boxes2d[:, 2:] = torch.ceil(gt_boxes2d[:, 2:])
    gt_boxes2d = gt_boxes2d.long()

    # Set all values within each box to True
    gt_boxes2d = gt_boxes2d.split(num_gt_per_img, dim=0)
    gt_center_depth = gt_center_depth.split(num_gt_per_img, dim=0)
    B = len(gt_boxes2d)
    for b in range(B):
        center_depth_per_batch = gt_center_depth[b]
        center_depth_per_batch, sorted_idx = torch.sort(center_depth_per_batch, dim=0, descending=True)
        gt_boxes_per_batch = gt_boxes2d[b][sorted_idx]
        for n in range(gt_boxes_per_batch.shape[0]):
            u1, v1, u2, v2 = gt_boxes_per_batch[n]
            depth_maps[b, v1:v2, u1:u2] = center_depth_per_batch[n]

    return depth_maps
