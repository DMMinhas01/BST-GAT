"""Bayesian Spatio-Temporal Graph Attention Network implementation."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn


class VariationalParameter(nn.Module):
    """A trainable parameter representing a diagonal Gaussian distribution."""

    def __init__(self, shape: Tuple[int, ...], init_std: float = 0.02) -> None:
        super().__init__()
        init_rho = torch.log(torch.expm1(torch.tensor(init_std)))
        self.mu = nn.Parameter(torch.zeros(*shape))
        self.rho = nn.Parameter(torch.full(shape, init_rho.item()))

    @property
    def std(self) -> Tensor:
        return torch.nn.functional.softplus(self.rho)

    def sample(self) -> Tensor:
        epsilon = torch.randn_like(self.mu)
        return self.mu + self.std * epsilon

    def kl_divergence(self) -> Tensor:
        var_post = self.std.pow(2)
        return 0.5 * torch.sum(self.mu.pow(2) + var_post - torch.log(var_post + 1e-8) - 1)


class BayesianLinear(nn.Module):
    """Variational Bayesian linear layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.weight = VariationalParameter((out_features, in_features), init_std)
        self.bias = VariationalParameter((out_features,), init_std) if bias else None

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        weight = self.weight.sample()
        output = torch.nn.functional.linear(input, weight)
        kl = self.weight.kl_divergence()
        if self.bias is not None:
            bias = self.bias.sample()
            output = output + bias
            kl = kl + self.bias.kl_divergence()
        return output, kl


class BayesianGraphAttentionLayer(nn.Module):
    """Graph Attention layer with variational parameters."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        alpha: float = 0.2,
    ) -> None:
        super().__init__()
        if out_features % num_heads != 0:
            raise ValueError("out_features must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads

        self.linear = BayesianLinear(in_features, out_features)
        self.att_src = VariationalParameter((num_heads, self.head_dim))
        self.att_dst = VariationalParameter((num_heads, self.head_dim))
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, adjacency: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Run the attention layer.

        Args:
            x: Tensor of shape ``(batch, nodes, in_features)``.
            adjacency: Normalized adjacency matrix ``(nodes, nodes)``.
            mask: Optional boolean mask ``(batch, nodes)`` marking valid nodes.
        Returns:
            output tensor and accumulated KL divergence.
        """

        Wh, kl_linear = self.linear(x)
        Wh = Wh.view(x.size(0), x.size(1), self.num_heads, self.head_dim)

        if mask is not None:
            mask = mask.unsqueeze(-1).unsqueeze(-1)
            Wh = Wh * mask

        att_src = torch.einsum("bnhd,hd->bnh", Wh, self.att_src.sample())
        att_dst = torch.einsum("bnhd,hd->bnh", Wh, self.att_dst.sample())
        scores = self.leakyrelu(att_src.unsqueeze(-1) + att_dst.unsqueeze(-2))

        adjacency = adjacency + torch.eye(adjacency.size(0), device=adjacency.device)
        adjacency_mask = (adjacency > 0).unsqueeze(0).unsqueeze(1)
        scores = scores.masked_fill(~adjacency_mask, float("-inf"))

        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention) * adjacency.unsqueeze(0).unsqueeze(1)
        attention = attention / (attention.sum(dim=-1, keepdim=True) + 1e-6)

        h_prime = torch.einsum("bhnm,bmhd->bnhd", attention, Wh)
        h_prime = h_prime.reshape(x.size(0), x.size(1), -1)

        kl_total = kl_linear + self.att_src.kl_divergence() + self.att_dst.kl_divergence()
        return h_prime, kl_total


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TemporalEncoder(nn.Module):
    """Transformer encoder for temporal dynamics."""

    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        ff_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.positional = PositionalEncoding(model_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_projection(x)
        x = self.positional(x)
        return self.transformer(x)


@dataclass
class BSTGATOutput:
    """Model output packaged with helper utilities."""

    mean: Tensor
    log_var: Tensor
    kl_divergence: Tensor

    def predictive_samples(self, num_samples: int) -> Tensor:
        std = torch.exp(0.5 * self.log_var)
        eps = torch.randn(num_samples, *self.mean.shape, device=self.mean.device)
        return self.mean.unsqueeze(0) + eps * std.unsqueeze(0)


class BayesianSpatioTemporalGAT(nn.Module):
    """Main BST-GAT model combining temporal, spatial and Bayesian layers."""

    def __init__(
        self,
        num_nodes: int,
        feature_dim: int,
        history: int,
        horizon: int,
        temporal_dim: int = 128,
        gat_dim: int = 128,
        gat_heads: int = 4,
        transformer_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.history = history
        self.horizon = horizon
        self.num_nodes = num_nodes

        self.temporal = TemporalEncoder(
            input_dim=feature_dim,
            model_dim=temporal_dim,
            ff_dim=temporal_dim * 4,
            num_layers=transformer_layers,
            num_heads=4,
            dropout=dropout,
        )
        self.temporal_norm = nn.LayerNorm(temporal_dim)
        self.gat = BayesianGraphAttentionLayer(
            temporal_dim,
            gat_dim,
            num_heads=gat_heads,
            dropout=dropout,
        )
        self.projection = BayesianLinear(gat_dim * num_nodes, horizon * num_nodes * 2)

    def forward(
        self,
        features: Tensor,
        adjacency: Tensor,
        mask: Optional[Tensor] = None,
    ) -> BSTGATOutput:
        """Forward pass.

        Args:
            features: Tensor with shape ``(batch, history, nodes, feature_dim)``.
            adjacency: Normalized adjacency matrix ``(nodes, nodes)``.
            mask: Optional boolean tensor ``(batch, nodes)`` marking valid nodes.
        """

        batch_size = features.size(0)
        x = (
            features.permute(0, 2, 1, 3)
            .reshape(batch_size * self.num_nodes, self.history, -1)
        )
        x = self.temporal(x)
        x = x[:, -1, :]
        x = self.temporal_norm(x)
        x = x.view(batch_size, self.num_nodes, -1)

        if mask is not None:
            spatial_mask = mask
        else:
            spatial_mask = None

        spatial, kl_gat = self.gat(x, adjacency, spatial_mask)
        spatial = torch.relu(spatial)
        spatial = spatial.reshape(batch_size, -1)

        output, kl_proj = self.projection(spatial)
        kl_total = kl_gat + kl_proj

        output = output.view(batch_size, self.horizon, self.num_nodes, 2)
        mean = output[..., 0]
        log_var = output[..., 1]
        return BSTGATOutput(mean=mean, log_var=log_var, kl_divergence=kl_total)


def gaussian_nll(pred: BSTGATOutput, target: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """Compute Gaussian negative log likelihood with optional mask."""
    var = torch.exp(pred.log_var) + 1e-6
    diff = target - pred.mean
    nll = 0.5 * (
        diff.pow(2) / var
        + pred.log_var
        + math.log(2 * math.pi)
    )
    if mask is not None:
        nll = nll * mask.unsqueeze(1)
    return nll.mean()


__all__ = [
    "BayesianSpatioTemporalGAT",
    "BSTGATOutput",
    "gaussian_nll",
]
