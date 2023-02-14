# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Various positional encodings for the transformer.
"""
import math
from typing import List, Literal, Tuple
import torch
from torch import nn

from utils.misc import inverse_sigmoid, NestedTensor


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor, calibs: torch.Tensor, img_size: torch.Tensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)

    def forward(self, tensor_list: NestedTensor, calibs: torch.Tensor, img_size: torch.Tensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device) / w * 49
        j = torch.arange(h, device=x.device) / h * 49
        x_emb = self.get_embed(i, self.col_embed)
        y_emb = self.get_embed(j, self.row_embed)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos

    def get_embed(self, coord, embed):
        floor_coord = coord.floor()
        delta = (coord - floor_coord).unsqueeze(-1)
        floor_coord = floor_coord.long()
        ceil_coord = (floor_coord + 1).clamp(max=49)
        return embed(floor_coord) * (1 - delta) + embed(ceil_coord) * delta


class PETR3DPositionEmbedding(nn.Module):
    def __init__(self,
                 num_pos_feats: int,
                 mode: Literal['LID', 'UD'] = 'LID',
                 depth_min: float = 1e-3,
                 depth_max: float = 60,
                 num_depth_bins: int = 80,
                 coord_range: List[float] = [-60, -3, -1, 60, 5, 60],
                 ):
        """
        Args:
            num_pos_feats: number of embedding dimensions.
            mode: depth class mode. Supports 'LID', 'UD'.
            depth_min: minimum depth value.
            depth_max: maximum depth value.
            num_depth_bins: number of depth classes.
            coord_range: range of the camera coordinate. (x_min, y_min, z_min, x_max, y_max, z_max).
        """
        super().__init__()
        self.num_pos_feats = num_pos_feats
        if mode not in ['LID', 'UD']:
            raise NotImplementedError(f'Expected modes are ["LID", "UD"]. Got {mode} instead.')
        self.mode = mode
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.num_depth_bins = num_depth_bins
        assert len(coord_range) == 6, '`coord_range` should be (x_min, y_min, z_min, x_max, y_max, z_max).'
        self.coord_range = coord_range

        self.position_encoder = nn.Sequential(
            nn.Conv2d(self.num_depth_bins * 3, self.num_pos_feats * 4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.num_pos_feats * 4, self.num_pos_feats, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, tensor_list: NestedTensor, calibs: torch.Tensor, img_size: Tuple[int, int]) -> torch.Tensor:
        """Returns 3D coordinate position embedding.

        Args:
            tensor_list: NestedTensor containing image features with shape [B, C, H, W].
            calibs: camera to image calibration tensor with shape [B, 3, 4].
            img_size: input image size (H, W).

        Returns:
            3D coordinate position embedding with shape [B, C, H, W].
        """
        eps = 1e-5
        x = tensor_list.tensors
        B, _, H, W = x.shape
        scale_factor_h, scale_factor_w = H / img_size[0], W / img_size[1]
        coords_h = torch.arange(H, device=x.device).float()
        coords_w = torch.arange(W, device=x.device).float()

        index = torch.arange(self.num_depth_bins, device=x.device).float()
        if self.mode == 'LID':
            bin_size = (self.depth_max - self.depth_min) / (self.num_depth_bins * (self.num_depth_bins + 1))
            coords_d = self.depth_min + bin_size * index * (index + 1)
        elif self.mode == 'UD':
            bin_size = (self.depth_max - self.depth_min) / self.num_depth_bins
            coords_d = self.depth_min + bin_size * index
        else:
            raise NotImplementedError(f'Expected modes are ["LID", "UD"]. Got {self.mode} instead.')

        # [W, H, D, 3]. Each element is [u, v, d].
        coords = torch.stack(
            torch.meshgrid(coords_w, coords_h, coords_d, indexing='ij'),
            dim=-1)
        # [W, H, D, 4]. Each element is [u, v, d, 1].
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        # [W, H, D, 4]. Each element is [u * d, v * d, d, 1].
        coords[..., :2] *= torch.maximum(coords[..., 2:3], coords.new_tensor([eps]))

        scale_matrix = calibs.new_tensor([[scale_factor_w, 0, 0],
                                          [0, scale_factor_h, 0],
                                          [0, 0, 1]])
        calibs = scale_matrix @ calibs
        # [B, 4, 4]
        calibs_4x4 = torch.cat([calibs, calibs.new_tensor([0, 0, 0, 1]).repeat(B, 1, 1)], dim=1)
        # [B, 3, 4]
        img2camera = torch.linalg.inv(calibs_4x4)[:, :3]

        # [B, W, H, D, 3]
        coords3d: torch.Tensor = torch.einsum('bij,whdj->bwhdi', img2camera, coords)
        coords3d[..., 0] = (coords3d[..., 0] - self.coord_range[0]) / (self.coord_range[3] - self.coord_range[0])
        coords3d[..., 1] = (coords3d[..., 1] - self.coord_range[1]) / (self.coord_range[4] - self.coord_range[1])
        coords3d[..., 2] = (coords3d[..., 2] - self.coord_range[2]) / (self.coord_range[5] - self.coord_range[2])

        # [B, D, 3, H, W] -> [B, D * 3, H, W]
        coords3d = coords3d.permute(0, 3, 4, 2, 1).flatten(1, 2)
        coords3d = inverse_sigmoid(coords3d)
        coords_position_embeding = self.position_encoder(coords3d)

        return coords_position_embeding.view(B, self.num_pos_feats, H, W)


def build_position_encoding(cfg):
    N_steps = cfg['hidden_dim'] // 2
    if cfg['position_embedding'] in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif cfg['position_embedding'] in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    elif cfg['position_embedding'] in ('3d_learned',):
        position_embedding = PETR3DPositionEmbedding(cfg['hidden_dim'],
                                                     mode=cfg['mode'],
                                                     depth_min=cfg['depth_min'],
                                                     depth_max=cfg['depth_max'],
                                                     num_depth_bins=cfg['num_depth_bins'],
                                                     )
    else:
        raise NotImplementedError(f"not supported {cfg['position_embedding']}")

    return position_embedding
