from typing import Literal, Tuple, Union
import torch
from torch import nn
from lib.layers.mlp import MLP


class CrossAttentionMLP(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 output_dim: int,
                 nheads: int = 8,
                 num_preprocess_layer: int = 0,
                 num_postprocess_layer: int = 1,
                 dropout: Union[bool, float] = False):
        super().__init__()
        if num_preprocess_layer > 0:
            self.preprocess_layer = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, num_layers=num_preprocess_layer, dropout=dropout)
        else:
            self.preprocess_layer = None
        self.cross_attn = nn.MultiheadAttention(hidden_dim, nheads, batch_first=True)

        assert num_postprocess_layer >= 1, 'number of postprocess layers must be greater equal than 1.'
        self.postprocess_layer = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_postprocess_layer, dropout=dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Returns attention values for each query in `x`.

        Args:
            query: A query tensor with shape [batch, num_queries, hidden_dim].
            key: A key tensor with shape [batch, num_queries, hidden_dim].
            value: A value tensor with shape [batch, num_queries, hidden_dim].

        Returns:
            A tensor with shape [batch, num_queries, output_dim].
        """
        if self.preprocess_layer:
            query = self.preprocess_layer(query)
        attn_value, attn_weights = self.cross_attn(query, key, value)
        output = self.postprocess_layer(attn_value)
        return output


class CrossAttentionMLPWithResidual(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 nheads: int = 8,
                 num_preprocess_layer: int = 0,
                 num_postprocess_layer: int = 1,
                 dropout: Union[bool, float] = False,
                 attn_module: Literal['attn', 'dot'] = 'attn'):
        super().__init__()
        if num_preprocess_layer > 0:
            self.preprocess_layer = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, num_layers=num_preprocess_layer, dropout=dropout)
        else:
            self.preprocess_layer = None
        assert attn_module in ['attn', 'dot']
        if attn_module == 'attn':
            self.cross_attn = nn.MultiheadAttention(hidden_dim, nheads, batch_first=True)
        elif attn_module == 'dot':
            self.cross_attn = DotAttention()

        assert num_postprocess_layer >= 1, 'number of postprocess layers must be greater equal than 1.'
        self.postprocess_layer = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=2, num_layers=num_postprocess_layer, dropout=dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Returns attention values for each query.

        Args:
            query: A query tensor with shape [batch, num_queries, hidden_dim].
            key: A key tensor with shape [batch, num_keys, hidden_dim].
            value: A value tensor with shape [batch, num_keys, hidden_dim].

        Returns:
            A tensor with shape [batch, num_queries, output_dim].
        """
        if self.preprocess_layer:
            query = self.preprocess_layer(query)
        attn_value, attn_weights = self.cross_attn(query, key, value)
        depth_bin_values = torch.arange(key.shape[1], dtype=key.dtype, device=key.device) + 0.5
        weighted_depth = torch.einsum('bqc,c->bq', attn_weights, depth_bin_values)
        depth_residual, depth_log_std = torch.split(self.postprocess_layer(attn_value), 1, dim=-1)
        depth = weighted_depth.unsqueeze(-1) + depth_residual
        return torch.cat([depth, depth_log_std], dim=-1)


class DotAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes attention values for each query.

        Args:
            query: A query tensor with shape [batch, num_queries, hidden_dim].
            key: A key tensor with shape [batch, num_keys, hidden_dim].
            value: A value tensor with shape [batch, num_keys, hidden_dim].

        Returns:
            A tuple with two tensors (attention_value, attention_weight).
            Each tensor has shape [batch, num_queries, output_dim].
        """
        dot_value: torch.Tensor = torch.einsum('bqc,bkc->bqk', query, key)
        weight = dot_value.softmax(-1)
        attn_value: torch.Tensor = torch.einsum('bqk,bkc->bqc', weight, value)
        return attn_value, weight


def build_depth_embedding(cfg):
    if 'depth_embedding' in cfg:
        module_type = cfg['depth_embedding'].pop('type', 'MLP')
        if module_type == 'MLP':
            return MLP(cfg['hidden_dim'], cfg['hidden_dim'], 2, 2)
        elif module_type == 'CrossAttentionMLP':
            return CrossAttentionMLP(hidden_dim=cfg['hidden_dim'],
                                     output_dim=2,
                                     **cfg['depth_embedding'])
        elif module_type == 'CrossAttentionMLPWithResidual':
            return CrossAttentionMLPWithResidual(hidden_dim=cfg['hidden_dim'],
                                                 **cfg['depth_embedding'])
        else:
            raise NotImplementedError(f'Invalid depth embedding type {module_type}.')
    return MLP(cfg['hidden_dim'], cfg['hidden_dim'], 2, 2)
