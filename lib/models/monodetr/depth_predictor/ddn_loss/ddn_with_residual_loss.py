from typing import List
import torch
import torch.nn as nn
import math

from .balancer import Balancer
from .focalloss import FocalLoss, RegressionFocalLoss

# based on:
# https://github.com/TRAILab/CaDDN/blob/master/pcdet/models/backbones_3d/ffe/ddn_loss/ddn_loss.py


class DDNWithResidualLoss(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2.0,
                 fg_weight=13,
                 bg_weight=1,
                 downsample_factor=1,
                 depth_min: float = 1e-3,
                 depth_max: float = 60,
                 num_depth_bins: int = 80):
        """
        Initializes DDNLoss module
        Args:
            weight [float]: Loss function weight
            alpha [float]: Alpha value for Focal Loss
            gamma [float]: Gamma value for Focal Loss
            disc_cfg [dict]: Depth discretiziation configuration
            fg_weight [float]: Foreground loss weight
            bg_weight [float]: Background loss weight
            downsample_factor [int]: Depth map downsample factor
        """
        super().__init__()
        self.device = torch.cuda.current_device()
        self.balancer = Balancer(
            downsample_factor=downsample_factor,
            fg_weight=fg_weight,
            bg_weight=bg_weight)

        # Set loss function
        self.alpha = alpha
        self.gamma = gamma
        self.depth_map_criterion = FocalLoss(alpha=self.alpha, gamma=self.gamma, reduction='none')
        self.depth_residual_criterion = RegressionFocalLoss(alpha=self.alpha, gamma=self.gamma, norm='l1', reduction='none')

        self.depth_min = depth_min
        self.depth_max = depth_max
        self.num_depth_bins = num_depth_bins
        bin_size = 2 * (depth_max - depth_min) / (num_depth_bins * (1 + num_depth_bins))
        bin_indice = torch.linspace(0, num_depth_bins - 1, num_depth_bins)
        bin_value = (bin_indice + 0.5).pow(2) * bin_size / 2 - bin_size / 8 + depth_min
        self.depth_bin_values = nn.parameter.Parameter(
            torch.cat([bin_value, torch.tensor([depth_max])], dim=0), requires_grad=False)

    def build_target_depth_from_3dcenter(self, depth_logits: torch.Tensor, gt_boxes2d: torch.Tensor, gt_center_depth: torch.Tensor, num_gt_per_img: List[int]):
        B, _, H, W = depth_logits.shape
        depth_maps = depth_logits.new_full((B, H, W), self.depth_max)

        # Set box corners
        gt_boxes2d[:, :2] = torch.floor(gt_boxes2d[:, :2])
        gt_boxes2d[:, 2:] = torch.ceil(gt_boxes2d[:, 2:])
        gt_boxes2d = gt_boxes2d.long()

        # Set all values within each box to True
        gt_boxes2d_list = gt_boxes2d.split(num_gt_per_img, dim=0)
        gt_center_depth_list = gt_center_depth.split(num_gt_per_img, dim=0)
        B = len(gt_boxes2d_list)
        for b in range(B):
            center_depth_per_batch = gt_center_depth_list[b]
            center_depth_per_batch, sorted_idx = torch.sort(center_depth_per_batch, dim=0, descending=True)
            gt_boxes_per_batch = gt_boxes2d_list[b][sorted_idx]
            for n in range(gt_boxes_per_batch.shape[0]):
                u1, v1, u2, v2 = gt_boxes_per_batch[n]
                depth_maps[b, v1:v2, u1:u2] = center_depth_per_batch[n]

        return depth_maps

    def bin_depths(self, depth_map, mode="LID", depth_min: float = 1e-3, depth_max: float = 60, num_bins: int = 80, target=False):
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

    def forward(self, depth_logits, depth_residuals, gt_boxes2d, num_gt_per_img, gt_center_depth):
        """Gets depth_map loss
        Args:
            depth_logits: torch.Tensor(B, D+1, H, W)]: Predicted depth logits
            gt_boxes2d [torch.Tensor (B, N, 4)]: 2D box labels for foreground/background balancing
            num_gt_per_img:
            gt_center_depth:
        Returns:
            [depth_map_loss, depth_residual_loss].
            * depth_map_loss [torch.Tensor(1)]: Depth classification network loss
            * depth_residual_loss [torch.Tensor(1)]: Depth residual loss
        """

        # Bin depth map to create target
        depth_map_values = self.build_target_depth_from_3dcenter(depth_logits, gt_boxes2d, gt_center_depth, num_gt_per_img)
        depth_target = self.bin_depths(depth_map_values, depth_min=self.depth_min, depth_max=self.depth_max, num_bins=self.num_depth_bins, target=True)
        # [batch, depth_map_H, depth_map_W]
        weighted_depth = self.depth_bin_values[depth_target]
        assert depth_map_values.shape == weighted_depth.shape
        depth_residual_target = depth_map_values - weighted_depth

        # Compute loss
        depth_map_loss = self.depth_map_criterion(depth_logits, depth_target)
        depth_residual_loss = self.depth_residual_criterion(depth_logits, depth_residuals, depth_target, depth_residual_target)

        # Compute foreground/background balancing
        depth_map_loss = self.balancer(loss=depth_map_loss, gt_boxes2d=gt_boxes2d, num_gt_per_img=num_gt_per_img)
        depth_residual_loss = self.balancer(loss=depth_residual_loss, gt_boxes2d=gt_boxes2d, num_gt_per_img=num_gt_per_img)

        return depth_map_loss, depth_residual_loss
