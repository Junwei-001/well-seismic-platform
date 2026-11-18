"""Inference architecture copied from locked P12/P13 research definitions.

Only the encoder type annotation changes; the native normalization, CNN query,
SAM2 key/value attention and decoder forward calculations are preserved.
Origin hashes are bound by source-locks/facies-p81-recovery-v1.json.
"""
import torch

ATTENTION_DIM = 128
ATTENTION_HEADS = 4
NATIVE_INPUT_SIZE = (128, 128)

def _sam2_normalize_native(inputs: torch.Tensor) -> torch.Tensor:
    """Normalize a native facies slice without spatial interpolation."""
    if inputs.ndim != 4 or inputs.shape[1] != 1:
        raise ValueError(f"SAM2 expects [B,1,H,W], got {tuple(inputs.shape)}")
    spatial = tuple(int(value) for value in inputs.shape[-2:])
    if spatial != NATIVE_INPUT_SIZE:
        raise ValueError(
            "P12 native SAM2 adaptation is locked to 128x128 development "
            f"slices, got {spatial}"
        )
    normalized = torch.clamp(inputs, -5.0, 5.0).add(5.0).div(10.0)
    normalized = normalized.repeat(1, 3, 1, 1)
    mean = normalized.new_tensor((0.485, 0.456, 0.406))[
        None, :, None, None
    ]
    std = normalized.new_tensor((0.229, 0.224, 0.225))[
        None, :, None, None
    ]
    return (normalized - mean) / std

def _coordinate_grid(
    batch_size: int,
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    coordinates = torch.stack((xx, yy), dim=0)[None]
    return coordinates.expand(batch_size, -1, -1, -1)

class CrossAttentionFusion(torch.nn.Module):
    """CNN queries attend to SAM2 key/value tokens at low resolution."""

    def __init__(
        self,
        *,
        cnn_channels: int = 512,
        sam_channels: int = 256,
        attention_dim: int = ATTENTION_DIM,
        attention_heads: int = ATTENTION_HEADS,
    ) -> None:
        super().__init__()
        if attention_dim % attention_heads != 0:
            raise ValueError("attention_dim must divide attention_heads")
        self.cnn_channels = cnn_channels
        self.sam_channels = sam_channels
        self.attention_dim = attention_dim
        self.query_projection = torch.nn.Sequential(
            torch.nn.Conv2d(cnn_channels, attention_dim, kernel_size=1),
            torch.nn.GroupNorm(8, attention_dim),
        )
        self.key_projection = torch.nn.Sequential(
            torch.nn.Conv2d(sam_channels, attention_dim, kernel_size=1),
            torch.nn.GroupNorm(8, attention_dim),
        )
        self.value_projection = torch.nn.Conv2d(
            sam_channels,
            attention_dim,
            kernel_size=1,
        )
        self.position_projection = torch.nn.Conv2d(
            2,
            attention_dim,
            kernel_size=1,
        )
        self.attention = torch.nn.MultiheadAttention(
            attention_dim,
            attention_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.output_projection = torch.nn.Conv2d(
            attention_dim,
            cnn_channels,
            kernel_size=1,
        )
        self.output_norm = torch.nn.GroupNorm(16, cnn_channels)
        self.fusion_scale_logit = torch.nn.Parameter(
            torch.tensor(-1.3862944)
        )
        torch.nn.init.normal_(
            self.output_projection.weight,
            mean=0.0,
            std=1e-3,
        )
        torch.nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        cnn_feature: torch.Tensor,
        sam_feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            cnn_feature.ndim != 4
            or sam_feature.ndim != 4
            or cnn_feature.shape[0] != sam_feature.shape[0]
        ):
            raise ValueError(
                "cross attention expects aligned four-dimensional "
                "CNN/SAM2 feature maps"
            )
        if cnn_feature.shape[1] != self.cnn_channels:
            raise ValueError(
                f"expected {self.cnn_channels} CNN channels, got "
                f"{cnn_feature.shape[1]}"
            )
        if sam_feature.shape[1] != self.sam_channels:
            raise ValueError(
                f"expected {self.sam_channels} SAM2 channels, got "
                f"{sam_feature.shape[1]}"
            )
        batch_size, _, query_height, query_width = cnn_feature.shape
        _, _, key_height, key_width = sam_feature.shape
        query_map = self.query_projection(cnn_feature)
        key_map = self.key_projection(sam_feature)
        value_map = self.value_projection(sam_feature)
        query_map = query_map + self.position_projection(
            _coordinate_grid(
                batch_size,
                query_height,
                query_width,
                device=query_map.device,
                dtype=query_map.dtype,
            )
        )
        key_map = key_map + self.position_projection(
            _coordinate_grid(
                batch_size,
                key_height,
                key_width,
                device=key_map.device,
                dtype=key_map.dtype,
            )
        )
        query_tokens = query_map.flatten(2).transpose(1, 2)
        key_tokens = key_map.flatten(2).transpose(1, 2)
        value_tokens = value_map.flatten(2).transpose(1, 2)
        attended, weights = self.attention(
            query_tokens,
            key_tokens,
            value_tokens,
            need_weights=True,
            average_attn_weights=True,
        )
        attended_map = attended.transpose(1, 2).reshape(
            batch_size,
            self.attention_dim,
            query_height,
            query_width,
        )
        delta = self.output_norm(
            self.output_projection(attended_map)
        )
        scale = torch.sigmoid(self.fusion_scale_logit)
        return cnn_feature + scale * delta, weights

    @property
    def fusion_scale(self) -> float:
        return float(
            torch.sigmoid(self.fusion_scale_logit).detach().cpu()
        )

class CrossAttentionSegmentationModel(torch.nn.Module):
    def __init__(
        self,
        small_model: torch.nn.Module,
        sam2_encoder: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.small_model = small_model
        self.sam2_encoder = sam2_encoder
        for parameter in self.small_model.parameters():
            parameter.requires_grad = False
        for parameter in self.small_model.decoder.parameters():
            parameter.requires_grad = True
        for parameter in self.small_model.segmentation_head.parameters():
            parameter.requires_grad = True
        self.fusion = CrossAttentionFusion(
            cnn_channels=int(
                self.small_model.encoder.out_channels[-1]
            )
        )

    def train(
        self,
        mode: bool = True,
    ) -> "CrossAttentionSegmentationModel":
        super().train(mode)
        self.small_model.encoder.eval()
        self.small_model.decoder.train(mode)
        self.small_model.segmentation_head.train(mode)
        self.sam2_encoder.train(mode)
        self.fusion.train(mode)
        return self

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            cnn_features = list(self.small_model.encoder(inputs))
        sam_features = self.sam2_encoder(inputs)
        fused, attention_weights = self.fusion(
            cnn_features[-1],
            sam_features[-1],
        )
        cnn_features[-1] = fused
        decoded = self.small_model.decoder(cnn_features)
        logits = self.small_model.segmentation_head(decoded)
        return logits, attention_weights
