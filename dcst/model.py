"""Core DCST architecture.

The implementation contains only the three modules used by the final model:

* Frequency Token Representation (FTR)
* Foldable Query-Key Re-parameterization (QKR)
* Center-Token Top-k Fusion (CTF)
"""

from __future__ import annotations

import copy
import math
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn


def _dct_basis(channels: int, coefficients: int) -> torch.Tensor:
    if not 0 < coefficients <= channels:
        raise ValueError(
            f"DCT coefficients must be in [1, {channels}], got {coefficients}"
        )
    sample = torch.arange(channels, dtype=torch.float32)
    frequency = torch.arange(coefficients, dtype=torch.float32).unsqueeze(1)
    basis = math.sqrt(2.0 / channels) * torch.cos(
        math.pi / channels * (sample + 0.5) * frequency
    )
    basis[0].fill_(math.sqrt(1.0 / channels))
    return basis


class FrequencyTokenRepresentation(nn.Module):
    """Fuse PCA-domain tokens with low-frequency raw-spectrum tokens."""

    def __init__(
        self,
        pca_channels: int,
        raw_channels: int,
        token_dim: int,
        dct_coefficients: int,
        dct_scale: float,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.main_projection = nn.Conv2d(
            pca_channels,
            token_dim,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.normalization = nn.BatchNorm2d(token_dim)
        self.activation = nn.ReLU()
        self.dct_scale = float(dct_scale)
        self.register_buffer(
            "dct_basis",
            _dct_basis(raw_channels, dct_coefficients),
        )

        # Keep the additional zero-initialized path RNG-neutral.
        rng_state = torch.get_rng_state()
        self.frequency_projection = nn.Conv2d(
            dct_coefficients,
            token_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        nn.init.zeros_(self.frequency_projection.weight)
        torch.set_rng_state(rng_state)

    def forward(
        self,
        pca_patch: torch.Tensor,
        raw_patch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frequency = F.conv2d(
            raw_patch,
            self.dct_basis[:, :, None, None],
        )
        features = self.main_projection(pca_patch)
        features = features + self.dct_scale * self.frequency_projection(
            frequency
        )
        features = self.activation(self.normalization(features))
        tokens = features.flatten(2).transpose(1, 2)
        frequency_tokens = frequency.flatten(2).transpose(1, 2)
        return tokens, frequency_tokens


class FullRankPath(nn.Module):
    """A linear path whose matrix product can be folded analytically."""

    def __init__(self, in_features: int, out_features: int, depth: int) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("Full-rank path depth must be positive")
        layers = [nn.Linear(in_features, out_features, bias=False)]
        layers.extend(
            nn.Linear(out_features, out_features, bias=False)
            for _ in range(depth - 1)
        )
        self.layers = nn.ModuleList(layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

    @torch.no_grad()
    def effective_weight(self) -> torch.Tensor:
        weight = self.layers[0].weight
        for layer in self.layers[1:]:
            weight = layer.weight @ weight
        return weight


class LowRankPath(nn.Module):
    """A bottleneck path with an analytically foldable effective matrix."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        depth: int,
    ) -> None:
        super().__init__()
        if rank < 1 or depth < 1:
            raise ValueError("Low-rank path rank and depth must be positive")
        self.down = nn.Linear(in_features, rank, bias=False)
        self.middle = nn.ModuleList(
            nn.Linear(rank, rank, bias=False) for _ in range(depth - 1)
        )
        self.up = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_normal_(self.down.weight)
        for layer in self.middle:
            nn.init.eye_(layer.weight)
        nn.init.kaiming_normal_(self.up.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.down(inputs)
        for layer in self.middle:
            outputs = layer(outputs)
        return self.up(outputs)

    @torch.no_grad()
    def effective_weight(self) -> torch.Tensor:
        weight = self.down.weight
        for layer in self.middle:
            weight = layer.weight @ weight
        return self.up.weight @ weight


class FoldableQueryKeyProjection(nn.Module):
    """QKR projection with gated full-rank and low-rank residual paths."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        full_rank_depths: Iterable[int],
        low_rank_depths: Iterable[int],
        low_rank: int,
        full_rank_scale: float,
        low_rank_scale: float,
        gate_init: float = 0.0,
        deploy: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.full_rank_scale = float(full_rank_scale)
        self.low_rank_scale = float(low_rank_scale)
        self.deploy = bool(deploy)

        if self.deploy:
            self.projection = nn.Linear(
                self.in_features,
                self.out_features,
                bias=False,
            )
            return

        self.base_projection = nn.Linear(
            self.in_features,
            self.out_features,
            bias=False,
        )
        self.full_rank_paths = nn.ModuleList(
            FullRankPath(self.in_features, self.out_features, int(depth))
            for depth in full_rank_depths
        )
        self.low_rank_paths = nn.ModuleList(
            LowRankPath(
                self.in_features,
                self.out_features,
                int(low_rank),
                int(depth),
            )
            for depth in low_rank_depths
        )
        self.full_rank_gate = nn.Parameter(
            torch.tensor(float(gate_init), dtype=torch.float32)
        )
        self.low_rank_gate = nn.Parameter(
            torch.tensor(float(gate_init), dtype=torch.float32)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.projection(inputs)

        outputs = self.base_projection(inputs)
        if self.full_rank_paths:
            full_rank = sum(path(inputs) for path in self.full_rank_paths)
            outputs = outputs + (
                self.full_rank_gate * self.full_rank_scale * full_rank
            )
        if self.low_rank_paths:
            low_rank = sum(path(inputs) for path in self.low_rank_paths)
            outputs = outputs + (
                self.low_rank_gate * self.low_rank_scale * low_rank
            )
        return outputs

    @torch.no_grad()
    def effective_weight(self) -> torch.Tensor:
        if self.deploy:
            return self.projection.weight
        weight = self.base_projection.weight.clone()
        for path in self.full_rank_paths:
            weight.add_(
                self.full_rank_gate
                * self.full_rank_scale
                * path.effective_weight()
            )
        for path in self.low_rank_paths:
            weight.add_(
                self.low_rank_gate
                * self.low_rank_scale
                * path.effective_weight()
            )
        return weight

    @torch.no_grad()
    def fold(self) -> None:
        if self.deploy:
            return
        reference = self.base_projection.weight
        projection = nn.Linear(
            self.in_features,
            self.out_features,
            bias=False,
        ).to(device=reference.device, dtype=reference.dtype)
        projection.weight.copy_(self.effective_weight())
        self.projection = projection
        del self.base_projection
        del self.full_rank_paths
        del self.low_rank_paths
        del self.full_rank_gate
        del self.low_rank_gate
        self.deploy = True


class CenterTokenTopKFusion(nn.Module):
    """Replace the center-query row with DCT-guided Top-k fusion."""

    def __init__(self, heads: int, top_k: int) -> None:
        super().__init__()
        self.top_k = int(top_k)
        self.fusion_weight = nn.Parameter(torch.zeros(int(heads)))

    @staticmethod
    def center_similarity(
        frequency_tokens: torch.Tensor,
    ) -> torch.Tensor:
        normalized = F.normalize(frequency_tokens, dim=-1, eps=1e-6)
        center_index = normalized.shape[1] // 2
        similarity = torch.einsum(
            "bd,bnd->bn",
            normalized[:, center_index],
            normalized,
        )
        neighbor_mask = torch.ones(
            similarity.shape[1],
            dtype=torch.bool,
            device=similarity.device,
        )
        neighbor_mask[center_index] = False
        neighbors = similarity[:, neighbor_mask]
        mean = neighbors.mean(dim=1, keepdim=True)
        std = neighbors.std(
            dim=1,
            keepdim=True,
            unbiased=False,
        ).clamp_min(1e-6)
        similarity = (similarity - mean) / std
        similarity[:, center_index] = 0.0
        return similarity

    def forward(
        self,
        logits: torch.Tensor,
        values: torch.Tensor,
        frequency_tokens: torch.Tensor,
    ) -> torch.Tensor:
        token_count = logits.shape[-1]
        if not 0 < self.top_k < token_count:
            raise ValueError(
                f"top_k must be in [1, {token_count - 1}], got {self.top_k}"
            )

        full_weights = logits.softmax(dim=-1)
        outputs = torch.einsum("bhij,bhjd->bhid", full_weights, values)
        center_index = token_count // 2

        similarity = self.center_similarity(frequency_tokens)
        neighbor_scores = similarity.clone()
        neighbor_scores[:, center_index] = -torch.inf
        selected = neighbor_scores.topk(self.top_k, dim=-1).indices
        allowed = torch.zeros_like(similarity, dtype=torch.bool)
        allowed.scatter_(1, selected, True)
        allowed[:, center_index] = True

        center_logits = logits[:, :, center_index, :].masked_fill(
            ~allowed.unsqueeze(1),
            -torch.inf,
        )
        topk_weights = center_logits.softmax(dim=-1)
        topk_output = torch.einsum("bhj,bhjd->bhd", topk_weights, values)
        full_output = outputs[:, :, center_index, :]
        mixed_output = topk_output + self.fusion_weight.view(1, -1, 1) * (
            full_output - topk_output
        )
        outputs = outputs.clone()
        outputs[:, :, center_index, :] = mixed_output
        return outputs


class QKRAttention(nn.Module):
    """Multi-head attention with QKR projections and optional CTF."""

    def __init__(
        self,
        token_dim: int,
        heads: int,
        head_dim: int,
        qkr_config: dict,
        top_k: int | None,
        dropout: float,
        deploy: bool,
    ) -> None:
        super().__init__()
        inner_dim = int(heads) * int(head_dim)
        projection_args = {
            "in_features": token_dim,
            "out_features": inner_dim,
            "full_rank_depths": qkr_config["full_rank_depths"],
            "low_rank_depths": qkr_config["low_rank_depths"],
            "low_rank": qkr_config["rank"],
            "full_rank_scale": qkr_config["full_rank_scale"],
            "low_rank_scale": qkr_config["low_rank_scale"],
            "gate_init": qkr_config["gate_init"],
            "deploy": deploy,
        }
        self.query_projection = FoldableQueryKeyProjection(**projection_args)
        self.key_projection = FoldableQueryKeyProjection(**projection_args)
        self.value_projection = nn.Linear(token_dim, inner_dim, bias=False)
        self.output_projection = nn.Sequential(
            nn.Linear(inner_dim, token_dim),
            nn.Dropout(dropout),
        )
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        self.scale = self.head_dim ** -0.5
        self.ctf = (
            CenterTokenTopKFusion(self.heads, int(top_k))
            if top_k is not None
            else None
        )

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = tensor.shape
        return tensor.view(
            batch,
            tokens,
            self.heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)

    def forward(
        self,
        tokens: torch.Tensor,
        frequency_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        queries = self._split_heads(self.query_projection(tokens))
        keys = self._split_heads(self.key_projection(tokens))
        values = self._split_heads(self.value_projection(tokens))
        logits = torch.einsum(
            "bhid,bhjd->bhij",
            queries,
            keys,
        ) * self.scale
        if self.ctf is not None:
            if frequency_tokens is None:
                raise ValueError("CTF requires low-frequency DCT tokens")
            outputs = self.ctf(logits, values, frequency_tokens)
        else:
            weights = logits.softmax(dim=-1)
            outputs = torch.einsum("bhij,bhjd->bhid", weights, values)
        outputs = outputs.permute(0, 2, 1, 3).contiguous()
        outputs = outputs.view(outputs.shape[0], outputs.shape[1], -1)
        return self.output_projection(outputs)

    @torch.no_grad()
    def fold_qkr(self) -> None:
        self.query_projection.fold()
        self.key_projection.fold()


class FeedForward(nn.Module):
    def __init__(self, token_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, token_dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.layers(tokens)


class DCSTBlock(nn.Module):
    def __init__(
        self,
        token_dim: int,
        heads: int,
        head_dim: int,
        mlp_dim: int,
        qkr_config: dict,
        top_k: int | None,
        dropout: float,
        deploy: bool,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(token_dim)
        self.attention = QKRAttention(
            token_dim,
            heads,
            head_dim,
            qkr_config,
            top_k,
            dropout,
            deploy,
        )
        self.ffn_norm = nn.LayerNorm(token_dim)
        self.ffn = FeedForward(token_dim, mlp_dim, dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        frequency_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens = tokens + self.attention(
            self.attention_norm(tokens),
            frequency_tokens,
        )
        tokens = tokens + self.ffn(self.ffn_norm(tokens))
        return tokens


class DCST(nn.Module):
    """Dual-frequency center-token Transformer for HSI classification."""

    def __init__(self, config: dict, deploy: bool = False) -> None:
        super().__init__()
        data = config["data"]
        model = config["model"]
        ftr = model["ftr"]
        qkr = model["qkr"]
        ctf = model["ctf"]

        token_dim = int(model["token_dim"])
        depth = int(model["depth"])
        self.ftr = FrequencyTokenRepresentation(
            pca_channels=int(data["pca_components"]),
            raw_channels=int(data["raw_channels"]),
            token_dim=token_dim,
            dct_coefficients=int(ftr["dct_coefficients"]),
            dct_scale=float(ftr["dct_scale"]),
            kernel_size=int(ftr["kernel_size"]),
        )
        self.token_dropout = nn.Dropout(float(model["token_dropout"]))
        self.blocks = nn.ModuleList(
            DCSTBlock(
                token_dim=token_dim,
                heads=int(model["heads"]),
                head_dim=int(model["head_dim"]),
                mlp_dim=int(model["mlp_dim"]),
                qkr_config=qkr,
                top_k=int(ctf["top_k"]) if index == 0 else None,
                dropout=float(model["dropout"]),
                deploy=deploy,
            )
            for index in range(depth)
        )
        classifier_dim = int(model["classifier_hidden_dim"])
        self.classifier = nn.Sequential(
            nn.Linear(token_dim, classifier_dim),
            nn.BatchNorm1d(classifier_dim),
            nn.Dropout(float(model["classifier_dropout"])),
            nn.ReLU(),
            nn.Linear(classifier_dim, int(data["num_classes"])),
        )

    def forward(
        self,
        pca_patch: torch.Tensor,
        raw_patch: torch.Tensor,
    ) -> torch.Tensor:
        tokens, frequency_tokens = self.ftr(pca_patch, raw_patch)
        tokens = self.token_dropout(tokens)
        for index, block in enumerate(self.blocks):
            tokens = block(
                tokens,
                frequency_tokens if index == 0 else None,
            )
        center_token = tokens[:, tokens.shape[1] // 2, :]
        return self.classifier(center_token)

    @torch.no_grad()
    def fold_qkr(self) -> None:
        for block in self.blocks:
            block.attention.fold_qkr()


@torch.no_grad()
def build_deploy_model(train_model: DCST) -> DCST:
    deploy_model = copy.deepcopy(train_model)
    deploy_model.fold_qkr()
    deploy_model.eval()
    return deploy_model
