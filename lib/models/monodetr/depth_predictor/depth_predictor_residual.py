import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import TransformerEncoder, TransformerEncoderLayer
from utils import depth_utils


class DepthPredictorResidual(nn.Module):

    def __init__(self, model_cfg):
        """
        Initialize depth predictor and depth encoder
        Args:
            model_cfg [EasyDict]: Depth classification network config
        """
        super().__init__()
        self.num_depth_bins = int(model_cfg["num_depth_bins"])
        self.depth_min = float(model_cfg["depth_min"])
        self.depth_max = float(model_cfg["depth_max"])

        self.depth_bin_values = nn.parameter.Parameter(
            depth_utils.get_depth_bin_values(self.depth_min, self.depth_max, self.num_depth_bins),
            requires_grad=False)

        # Create modules
        d_model = model_cfg["hidden_dim"]
        self.downsample = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.GroupNorm(32, d_model))
        self.proj = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(1, 1)),
            nn.GroupNorm(32, d_model))
        self.upsample = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(1, 1)),
            nn.GroupNorm(32, d_model))

        self.depth_head = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(32, num_channels=d_model),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(32, num_channels=d_model),
            nn.ReLU())

        self.depth_classifier = nn.Conv2d(d_model, self.num_depth_bins + 1, kernel_size=(1, 1))
        self.depth_residual = nn.Conv2d(d_model, self.num_depth_bins + 1, kernel_size=(1, 1))

        depth_encoder_layer = TransformerEncoderLayer(
            d_model, nhead=8, dim_feedforward=256, dropout=0.1)

        self.depth_encoder = TransformerEncoder(depth_encoder_layer, 1)

        self.depth_pos_embed = nn.Embedding(int(self.depth_max) + 1, 256)

    def forward(self, feature, mask, pos, targets):

        assert len(feature) == 4

        # foreground depth map
        src_16 = self.proj(feature[1])
        src_32 = self.upsample(F.interpolate(feature[2], size=src_16.shape[-2:]))
        src_8 = self.downsample(feature[0])
        src = (src_8 + src_16 + src_32) / 3

        src = self.depth_head(src)
        depth_logits: torch.Tensor = self.depth_classifier(src)
        # [batch, num_depth_bins + 1, depth_map_H, depth_map_W]
        depth_residual: torch.Tensor = self.depth_residual(src)

        # weighted_depth = depth_utils.get_gt_depth_map_values(depth_logits, targets, self.depth_max)
        # num_bins = 80
        # depth_indices = depth_utils.bin_depths(weighted_depth, num_bins=num_bins, target=True)
        # # [batch, depth_map_H, depth_map_W, num_depth_bins + 1], dtype: torch.float
        # depth_logits = F.one_hot(depth_indices, num_classes=num_bins + 1).float() * 10
        # # [batch, num_depth_bins, depth_map_H, depth_map_W], dtype: torch.float
        # depth_logits = depth_logits.permute(0, 3, 1, 2)

        # # gt depth residual
        # # Bin depth map to create target
        # depth_map_values = depth_utils.get_gt_depth_map_values(depth_logits, targets, self.depth_max)
        # depth_target = depth_utils.bin_depths(depth_map_values, depth_min=self.depth_min, depth_max=self.depth_max, num_bins=self.num_depth_bins, target=True)
        # # [batch, depth_map_H, depth_map_W]
        # weighted_depth_target = self.depth_bin_values[depth_target]
        # assert depth_map_values.shape == weighted_depth_target.shape
        # depth_residual_target = depth_map_values - weighted_depth_target
        # depth_probs = F.softmax(depth_logits, dim=1)
        # # [batch, depth_map_H, depth_map_W]
        # weighted_depth = torch.einsum('bchw,c->bhw', depth_probs, self.depth_bin_values)
        # weighted_depth += depth_residual_target
        # depth_residual = depth_residual_target.unsqueeze(1).repeat(1, depth_logits.shape[1], 1, 1)

        depth_probs = F.softmax(depth_logits, dim=1)
        # [batch, depth_map_H, depth_map_W]
        weighted_depth = torch.einsum('bchw,c->bhw', depth_probs, self.depth_bin_values)
        weighted_depth += depth_residual.gather(dim=1, index=depth_logits.argmax(1, keepdim=True)).squeeze()

        # depth embeddings with depth positional encodings
        B, C, H, W = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        pos = pos.flatten(2).permute(2, 0, 1)

        depth_embed = self.depth_encoder(src, mask, pos)
        depth_embed = depth_embed.permute(1, 2, 0).reshape(B, C, H, W)

        depth_pos_embed_ip = self.interpolate_depth_embed(weighted_depth)
        depth_embed = depth_embed + depth_pos_embed_ip

        return depth_logits, depth_embed, weighted_depth, depth_residual

    def interpolate_depth_embed(self, depth):
        depth = depth.clamp(min=0, max=self.depth_max)
        pos = self.interpolate_1d(depth, self.depth_pos_embed)
        pos = pos.permute(0, 3, 1, 2)
        return pos

    def interpolate_1d(self, coord, embed):
        floor_coord = coord.floor()
        delta = (coord - floor_coord).unsqueeze(-1)
        floor_coord = floor_coord.long()
        ceil_coord = (floor_coord + 1).clamp(max=embed.num_embeddings - 1)
        return embed(floor_coord) * (1 - delta) + embed(ceil_coord) * delta
