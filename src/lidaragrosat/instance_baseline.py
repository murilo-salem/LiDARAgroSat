from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from torch import nn
from torch.utils.data import Dataset

from lidaragrosat.data.dataset import ProcessedPlotBundle
from lidaragrosat.instance_targets import build_instance_targets
from lidaragrosat.models.blocks import FPNDecoder, ResidualConvBlock, SimpleEncoder


def _build_input_stack(bundle: ProcessedPlotBundle) -> np.ndarray:
    arrays = [bundle.rgb.astype(np.float32), bundle.lidar.astype(np.float32), bundle.chm[None, ...].astype(np.float32)]
    if bundle.chm_features is not None:
        arrays.append(bundle.chm_features.astype(np.float32))
    return np.concatenate(arrays, axis=0)


def _extract_with_padding(array: np.ndarray, row: int, col: int, size: int) -> np.ndarray:
    row_end = row + size
    col_end = col + size
    pad_top = max(0, -row)
    pad_left = max(0, -col)
    pad_bottom = max(0, row_end - array.shape[-2])
    pad_right = max(0, col_end - array.shape[-1])
    row_start = max(0, row)
    col_start = max(0, col)
    row_stop = min(array.shape[-2], row_end)
    col_stop = min(array.shape[-1], col_end)
    if array.ndim == 3:
        crop = array[:, row_start:row_stop, col_start:col_stop]
        return np.pad(crop, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode="reflect").astype(np.float32)
    crop = array[row_start:row_stop, col_start:col_stop]
    return np.pad(crop, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="reflect").astype(np.float32)


@dataclass(slots=True)
class TileSample:
    plot_id: str
    row: int
    col: int


class InstanceTileDataset(Dataset):
    def __init__(
        self,
        bundles: dict[str, ProcessedPlotBundle],
        plot_ids: list[str],
        tile_size: int = 128,
        stride: int = 96,
        training: bool = True,
        seed: int = 13,
    ) -> None:
        self.bundles = bundles
        self.plot_ids = plot_ids
        self.tile_size = int(tile_size)
        self.training = bool(training)
        self.rng = np.random.default_rng(seed)
        self.samples: list[TileSample] = []
        self.inputs = {plot_id: _build_input_stack(bundles[plot_id]) for plot_id in plot_ids}
        self.targets = {plot_id: build_instance_targets(bundles[plot_id]) for plot_id in plot_ids}
        for plot_id in plot_ids:
            height, width = bundles[plot_id].raster_shape
            for row in range(0, max(height - tile_size + 1, 1), stride):
                for col in range(0, max(width - tile_size + 1, 1), stride):
                    self.samples.append(TileSample(plot_id=plot_id, row=row, col=col))
        if self.training and self.samples:
            positive = []
            negative = []
            for sample in self.samples:
                target = _extract_with_padding(
                    self.targets[sample.plot_id].seedness,
                    sample.row,
                    sample.col,
                    self.tile_size,
                )
                if float(target.max()) > 0.1:
                    positive.append(sample)
                else:
                    negative.append(sample)
            keep_neg = negative[: max(len(positive) // 2, min(64, len(negative)))]
            self.samples = positive + keep_neg

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        sample = self.samples[index]
        inputs = _extract_with_padding(self.inputs[sample.plot_id], sample.row, sample.col, self.tile_size)
        distance = _extract_with_padding(self.targets[sample.plot_id].distance_transform, sample.row, sample.col, self.tile_size)
        seedness = _extract_with_padding(self.targets[sample.plot_id].seedness, sample.row, sample.col, self.tile_size)
        if self.training:
            if self.rng.random() < 0.5:
                inputs = np.flip(inputs, axis=2).copy()
                distance = np.flip(distance, axis=1).copy()
                seedness = np.flip(seedness, axis=1).copy()
            if self.rng.random() < 0.5:
                inputs = np.flip(inputs, axis=1).copy()
                distance = np.flip(distance, axis=0).copy()
                seedness = np.flip(seedness, axis=0).copy()
        return {
            "image": torch.from_numpy(inputs.astype(np.float32)),
            "distance": torch.from_numpy(distance[None, ...].astype(np.float32)),
            "seedness": torch.from_numpy(seedness[None, ...].astype(np.float32)),
            "plot_id": sample.plot_id,
            "row": sample.row,
            "col": sample.col,
        }


class DistanceSeednessNet(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.encoder = SimpleEncoder(in_channels, [32, 64, 128], norm="group")
        self.decoder = FPNDecoder([32, 64, 128], [128, 64, 32], norm="group")
        self.trunk = ResidualConvBlock(32, 32, norm="group")
        self.distance_head = nn.Conv2d(32, 1, kernel_size=1)
        self.seedness_head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(x)
        decoded = self.decoder(features)
        trunk = self.trunk(decoded)
        return {
            "distance": torch.sigmoid(self.distance_head(trunk)),
            "seedness": self.seedness_head(trunk),
        }


def instance_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    distance_loss = torch.nn.functional.smooth_l1_loss(outputs["distance"], batch["distance"])
    seedness_loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs["seedness"], batch["seedness"])
    return distance_loss + seedness_loss


def infer_full_plot(
    model: nn.Module,
    bundle: ProcessedPlotBundle,
    tile_size: int = 128,
    stride: int = 96,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    inputs = _build_input_stack(bundle)
    height, width = bundle.raster_shape
    distance_sum = np.zeros((height, width), dtype=np.float32)
    seedness_sum = np.zeros((height, width), dtype=np.float32)
    counter = np.zeros((height, width), dtype=np.float32)
    rows = list(range(0, max(height - tile_size + 1, 1), stride))
    cols = list(range(0, max(width - tile_size + 1, 1), stride))
    if rows[-1] != max(height - tile_size, 0):
        rows.append(max(height - tile_size, 0))
    if cols[-1] != max(width - tile_size, 0):
        cols.append(max(width - tile_size, 0))
    with torch.no_grad():
        for row in rows:
            for col in cols:
                tile = _extract_with_padding(inputs, row, col, tile_size)
                tensor = torch.from_numpy(tile[None, ...]).to(device)
                outputs = model(tensor)
                distance = outputs["distance"].cpu().numpy()[0, 0]
                seedness = torch.sigmoid(outputs["seedness"]).cpu().numpy()[0, 0]
                row_end = min(row + tile_size, height)
                col_end = min(col + tile_size, width)
                distance_sum[row:row_end, col:col_end] += distance[: row_end - row, : col_end - col]
                seedness_sum[row:row_end, col:col_end] += seedness[: row_end - row, : col_end - col]
                counter[row:row_end, col:col_end] += 1.0
    counter = np.clip(counter, 1.0, None)
    return distance_sum / counter, seedness_sum / counter


def decode_instance_maps(
    bundle: ProcessedPlotBundle,
    distance_map: np.ndarray,
    seedness_map: np.ndarray,
    seed_threshold: float = 0.35,
    mask_threshold: float = 0.10,
    min_distance_m: float = 1.0,
    smooth_sigma_px: float = 1.0,
) -> pd.DataFrame:
    distance_smooth = gaussian_filter(distance_map.astype(np.float32), sigma=float(smooth_sigma_px), mode="nearest")
    seedness_smooth = gaussian_filter(seedness_map.astype(np.float32), sigma=float(smooth_sigma_px), mode="nearest")
    mask = (distance_smooth >= float(mask_threshold)) | (seedness_smooth >= float(seed_threshold * 0.5))
    min_distance_px = max(1, int(round(float(min_distance_m) / bundle.transform.pixel_size)))
    peaks = peak_local_max(
        seedness_smooth,
        min_distance=min_distance_px,
        threshold_abs=float(seed_threshold),
        labels=mask.astype(np.uint8),
        exclude_border=False,
    )
    markers = np.zeros_like(distance_smooth, dtype=np.int32)
    for marker_id, (row, col) in enumerate(peaks, start=1):
        markers[int(row), int(col)] = int(marker_id)
    if markers.max() == 0:
        return pd.DataFrame(columns=["x", "y", "height_pred", "confidence"])
    segments = watershed(-distance_smooth, markers=markers, mask=mask)
    records: list[dict] = []
    for segment_id in np.unique(segments):
        if int(segment_id) <= 0:
            continue
        seg_mask = segments == int(segment_id)
        if not np.any(seg_mask):
            continue
        score = float(seedness_smooth[seg_mask].max())
        row_indices, col_indices = np.where(seg_mask)
        chm_values = bundle.chm[seg_mask]
        if len(chm_values) == 0:
            continue
        best_index = int(np.nanargmax(chm_values))
        row = int(row_indices[best_index])
        col = int(col_indices[best_index])
        x, y = bundle.transform.rowcol_to_xy(row, col)
        height_pred = float(bundle.chm[row, col])
        records.append({"x": float(x), "y": float(y), "height_pred": height_pred, "confidence": score})
    return pd.DataFrame.from_records(records)
