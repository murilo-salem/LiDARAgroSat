from __future__ import annotations

import torch
from torch import nn

from lidaragrosat.models.blocks import FPNDecoder, FusionBlock, ResidualConvBlock, SimpleEncoder, TriFusionBlock


class DetectionHeads(nn.Module):
    def __init__(
        self,
        in_channels: int,
        use_height_head: bool,
        use_offsets: bool,
        use_distance_head: bool,
        use_peakness_head: bool,
        use_species_head: bool,
        species_num_classes: int,
    ) -> None:
        super().__init__()
        self.trunk = ResidualConvBlock(in_channels, in_channels)
        self.heatmap = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.offset = nn.Conv2d(in_channels, 2, kernel_size=1) if use_offsets else None
        self.height = nn.Conv2d(in_channels, 1, kernel_size=1) if use_height_head else None
        self.distance = nn.Conv2d(in_channels, 1, kernel_size=1) if use_distance_head else None
        self.peakness = nn.Conv2d(in_channels, 1, kernel_size=1) if use_peakness_head else None
        self.species = nn.Conv2d(in_channels, species_num_classes, kernel_size=1) if use_species_head and species_num_classes > 0 else None

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.trunk(x)
        outputs = {"heatmap": self.heatmap(features)}
        if self.offset is not None:
            outputs["offset"] = self.offset(features)
        if self.height is not None:
            outputs["height"] = self.height(features)
        if self.distance is not None:
            outputs["distance"] = self.distance(features)
        if self.peakness is not None:
            outputs["peakness"] = self.peakness(features)
        if self.species is not None:
            outputs["species"] = self.species(features)
        return outputs


class CenterBasedTreeDetector(nn.Module):
    def __init__(
        self,
        modality: str,
        fusion_strategy: str,
        rgb_in_channels: int,
        lidar_in_channels: int,
        chm_in_channels: int,
        encoder_channels: list[int],
        decoder_channels: list[int],
        normalization: str = "group",
        use_height_head: bool = True,
        use_offsets: bool = True,
        use_chm_branch: bool = False,
        use_distance_head: bool = False,
        use_peakness_head: bool = False,
        use_species_head: bool = False,
        species_num_classes: int = 0,
        use_deep_supervision: bool = False,
    ) -> None:
        super().__init__()
        self.modality = modality
        self.fusion_strategy = fusion_strategy
        self.use_deep_supervision = use_deep_supervision
        self.use_peakness_head = use_peakness_head
        self.use_chm_branch = use_chm_branch

        if modality in {"rgb", "lidar", "early"}:
            input_channels = rgb_in_channels if modality == "rgb" else lidar_in_channels
            if modality == "early":
                input_channels = rgb_in_channels + lidar_in_channels
            self.encoder = SimpleEncoder(input_channels, encoder_channels, norm=normalization)
            self.decoder = FPNDecoder(encoder_channels, decoder_channels, norm=normalization)
            head_in_channels = self.decoder.output_channels
        elif modality == "multimodal":
            self.rgb_encoder = SimpleEncoder(rgb_in_channels, encoder_channels, norm=normalization)
            self.lidar_encoder = SimpleEncoder(lidar_in_channels, encoder_channels, norm=normalization)
            self.chm_encoder = SimpleEncoder(chm_in_channels, encoder_channels, norm=normalization) if use_chm_branch else None
            self.decoder = FPNDecoder(encoder_channels, decoder_channels, norm=normalization)
            if fusion_strategy == "mid":
                if use_chm_branch:
                    self.fusion_blocks = nn.ModuleList(
                        [TriFusionBlock(channels, norm=normalization) for channels in encoder_channels]
                    )
                else:
                    self.fusion_blocks = nn.ModuleList(
                        [FusionBlock(channels, norm=normalization) for channels in encoder_channels]
                    )
                head_in_channels = self.decoder.output_channels
            elif fusion_strategy == "late":
                self.rgb_decoder = FPNDecoder(encoder_channels, decoder_channels, norm=normalization)
                self.lidar_decoder = FPNDecoder(encoder_channels, decoder_channels, norm=normalization)
                if use_chm_branch:
                    self.chm_decoder = FPNDecoder(encoder_channels, decoder_channels, norm=normalization)
                    self.late_fusion = ResidualConvBlock(3 * self.decoder.output_channels, self.decoder.output_channels)
                else:
                    self.late_fusion = ResidualConvBlock(2 * self.decoder.output_channels, self.decoder.output_channels)
                head_in_channels = self.decoder.output_channels
            else:
                raise ValueError(f"Unsupported fusion strategy: {fusion_strategy}")
        else:
            raise ValueError(f"Unsupported modality: {modality}")

        self.heads = DetectionHeads(
            in_channels=head_in_channels,
            use_height_head=use_height_head,
            use_offsets=use_offsets,
            use_distance_head=use_distance_head,
            use_peakness_head=use_peakness_head,
            use_species_head=use_species_head,
            species_num_classes=species_num_classes,
        )
        if self.use_deep_supervision:
            aux_channels = decoder_channels[:-1]
            self.aux_heatmap_heads = nn.ModuleList([nn.Conv2d(channels, 1, kernel_size=1) for channels in aux_channels])
        else:
            self.aux_heatmap_heads = nn.ModuleList()

    def forward(
        self,
        rgb: torch.Tensor | None = None,
        lidar: torch.Tensor | None = None,
        chm: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.modality == "multimodal" and self.use_chm_branch and chm is None:
            raise ValueError("CHM branch is enabled, but no CHM features were provided.")
        if self.modality == "rgb":
            decoded = self.decoder(self.encoder(rgb), return_pyramid=self.use_deep_supervision)
        elif self.modality == "lidar":
            decoded = self.decoder(self.encoder(lidar), return_pyramid=self.use_deep_supervision)
        elif self.modality == "early":
            decoded = self.decoder(self.encoder(torch.cat([rgb, lidar], dim=1)), return_pyramid=self.use_deep_supervision)
        elif self.modality == "multimodal" and self.fusion_strategy == "mid":
            rgb_features = self.rgb_encoder(rgb)
            lidar_features = self.lidar_encoder(lidar)
            if self.use_chm_branch:
                chm_features = self.chm_encoder(chm)
                fused = [
                    block(rgb_feature, lidar_feature, chm_feature)
                    for block, rgb_feature, lidar_feature, chm_feature in zip(self.fusion_blocks, rgb_features, lidar_features, chm_features)
                ]
            else:
                fused = [block(rgb_feature, lidar_feature) for block, rgb_feature, lidar_feature in zip(self.fusion_blocks, rgb_features, lidar_features)]
            decoded = self.decoder(fused, return_pyramid=self.use_deep_supervision)
        elif self.modality == "multimodal" and self.fusion_strategy == "late":
            rgb_features = self.rgb_encoder(rgb)
            lidar_features = self.lidar_encoder(lidar)
            rgb_decoded = self.rgb_decoder(rgb_features)
            lidar_decoded = self.lidar_decoder(lidar_features)
            if self.use_chm_branch:
                chm_features = self.chm_encoder(chm)
                chm_decoded = self.chm_decoder(chm_features)
                decoded = self.late_fusion(torch.cat([rgb_decoded, lidar_decoded, chm_decoded], dim=1))
            else:
                decoded = self.late_fusion(torch.cat([rgb_decoded, lidar_decoded], dim=1))
        else:
            raise RuntimeError("Invalid detector state.")

        pyramid = None
        if isinstance(decoded, tuple):
            decoded, pyramid = decoded
        outputs = self.heads(decoded)
        if self.use_deep_supervision and pyramid is not None and self.aux_heatmap_heads:
            aux_features = pyramid[:-1]
            outputs["aux_heatmaps"] = [
                head(feature) for head, feature in zip(self.aux_heatmap_heads, aux_features)
            ]
        return outputs


def build_detector_from_config(config) -> CenterBasedTreeDetector:
    lidar_in_channels = (
        len(config.data.structural_channels)
        if hasattr(config, "data") and getattr(config.data, "structural_channels", None)
        else config.model.lidar_in_channels
    )
    return CenterBasedTreeDetector(
        modality=config.model.modality,
        fusion_strategy=config.model.fusion_strategy,
        rgb_in_channels=config.model.rgb_in_channels,
        lidar_in_channels=lidar_in_channels,
        chm_in_channels=config.model.chm_in_channels,
        encoder_channels=config.model.encoder_channels,
        decoder_channels=config.model.decoder_channels,
        normalization=config.model.normalization,
        use_height_head=config.model.use_height_head,
        use_offsets=config.model.use_offsets,
        use_chm_branch=config.model.use_chm_branch,
        use_distance_head=config.model.use_distance_head,
        use_peakness_head=config.model.use_peakness_head,
        use_species_head=config.model.use_species_head,
        species_num_classes=config.model.species_num_classes,
        use_deep_supervision=config.model.use_deep_supervision,
    )
