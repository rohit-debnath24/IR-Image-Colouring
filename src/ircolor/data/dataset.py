"""Geospatial-aware dataset for IR->RGB super-resolution + colorization.

KEY INVARIANT: data stays 16-bit float until the very last step. We never round-trip
through 8-bit PNG, which would crush the high dynamic range of thermal bands. All
normalization is computed PER TILE from scene statistics, not from global thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class TilePair:
    """One co-registered training sample. Arrays are float32, CHW, 16-bit dynamic range preserved."""
    ir: np.ndarray          # (C_ir, H, W) low-res IR/thermal input
    rgb: np.ndarray         # (3, H, W) high-res RGB target (None for unpaired/CUT mode)
    crs: str                # source coordinate reference system (kept for georeferenced export)
    transform: tuple        # affine geotransform of the tile


class LandsatIRDataset(Dataset):
    """Reads pre-tiled .tif pairs produced by `ircolor.data.prepare`.

    Tiles live in `tiles_dir` as `{id}_ir.tif` and `{id}_rgb.tif`. See data/prepare.py
    for the co-registration + windowed-tiling step that creates them.
    """

    def __init__(self, tiles_dir: str | Path, normalize: str = "per_tile_zscore", paired: bool = True, scale: int = 4):
        self.tiles_dir = Path(tiles_dir)
        self.normalize = normalize
        self.paired = paired
        self.scale = scale
        self.ids = sorted(p.stem.removesuffix("_ir") for p in self.tiles_dir.glob("*_ir.tif"))

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> TilePair:  # pragma: no cover - IO heavy
        import rasterio
        
        tile_id = self.ids[idx]
        ir_path = self.tiles_dir / f"{tile_id}_ir.tif"
        rgb_path = self.tiles_dir / f"{tile_id}_rgb.tif"
        
        with rasterio.open(ir_path) as src_ir:
            ir_data = src_ir.read().astype(np.float32)
            crs = str(src_ir.crs)
            transform = tuple(src_ir.transform)
            
        rgb_data = None
        if self.paired and rgb_path.exists():
            with rasterio.open(rgb_path) as src_rgb:
                rgb_data = src_rgb.read().astype(np.float32)
                
        # Downsample IR to its low-resolution representation matching scale
        if self.scale > 1:
            ir_data = ir_data[:, ::self.scale, ::self.scale]
            
        # Adaptive per-tile normalization
        if self.normalize == "per_tile_zscore":
            ir_mean = ir_data.mean(axis=(1, 2), keepdims=True)
            ir_std = ir_data.std(axis=(1, 2), keepdims=True) + 1e-8
            ir_data = (ir_data - ir_mean) / ir_std
            
            if rgb_data is not None:
                rgb_mean = rgb_data.mean(axis=(1, 2), keepdims=True)
                rgb_std = rgb_data.std(axis=(1, 2), keepdims=True) + 1e-8
                rgb_data = (rgb_data - rgb_mean) / rgb_std
                
        elif self.normalize == "per_tile_minmax":
            ir_min = ir_data.min(axis=(1, 2), keepdims=True)
            ir_max = ir_data.max(axis=(1, 2), keepdims=True)
            ir_data = (ir_data - ir_min) / (ir_max - ir_min + 1e-8) * 2.0 - 1.0
            
            if rgb_data is not None:
                rgb_min = rgb_data.min(axis=(1, 2), keepdims=True)
                rgb_max = rgb_data.max(axis=(1, 2), keepdims=True)
                rgb_data = (rgb_data - rgb_min) / (rgb_max - rgb_min + 1e-8) * 2.0 - 1.0
                
        return TilePair(ir=ir_data, rgb=rgb_data, crs=crs, transform=transform)


