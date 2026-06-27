"""Stage 1 of the pipeline: geospatial ingest -> co-registered tiles.

Responsibilities (see CLAUDE.md "Data pipeline"):
  1. Read Landsat 8/9 L2 scenes (OLI RGB bands + TIRS/NIR bands) with rasterio.
  2. Reproject + resample IR bands onto the RGB grid so pixels are co-registered.
  3. Window into overlapping tiles (tile_size, overlap) preserving CRS + geotransform.
  4. Write {id}_ir.tif / {id}_rgb.tif as 16-bit float -- NO 8-bit conversion.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
from omegaconf import OmegaConf

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    
    config = OmegaConf.load(args.config)
    raw_dir = Path(config.data.raw_dir)
    tiles_dir = Path(config.data.tiles_dir)
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning raw directory: {raw_dir} for Landsat scenes...")
    # Find scene subfolders or tif files
    scenes = [d for d in raw_dir.iterdir() if d.is_dir() or d.suffix.lower() == ".tif"]
    
    if not scenes:
        print("No raw Landsat scenes found. Creating mock folders for pipeline validation.")
        print(f"Please place raw scenes in {raw_dir} and run again.")
        return

    tile_size = config.data.tile_size
    overlap = config.data.overlap
    stride = tile_size - overlap
    
    for scene in scenes:
        scene_name = scene.stem
        print(f"Processing scene: {scene_name}")
        
        # Determine files for RGB and IR bands. 
        # Supports separate files (e.g. *_B4.TIF) or stacked files.
        if scene.is_dir():
            band_files = list(scene.glob("*.tif")) + list(scene.glob("*.TIF"))
            if not band_files:
                continue
            
            # Map band names (e.g. B4 -> Red, B5 -> NIR)
            band_map = {}
            for bf in band_files:
                for b_idx in config.data.rgb_bands + config.data.ir_bands:
                    if f"_B{b_idx}." in bf.name or bf.name.endswith(f"_B{b_idx}.tif") or bf.name.endswith(f"_B{b_idx}.TIF"):
                        band_map[b_idx] = bf
            
            # Verify required bands
            required_bands = config.data.rgb_bands + config.data.ir_bands
            missing_bands = [b for b in required_bands if b not in band_map]
            if missing_bands:
                print(f"Skipping scene {scene_name}: missing bands {missing_bands}")
                continue
                
            # Use B4 (Red) as structural/spatial reference
            ref_band = config.data.rgb_bands[0]
            with rasterio.open(band_map[ref_band]) as ref_src:
                ref_meta = ref_src.meta.copy()
                ref_transform = ref_src.transform
                ref_crs = ref_src.crs
                
            ref_meta.update({
                'dtype': 'float32',
                'count': 1,
            })
            
            height, width = ref_meta['height'], ref_meta['width']
            
            # Read and stack RGB
            rgb_data = []
            for b_idx in config.data.rgb_bands:
                with rasterio.open(band_map[b_idx]) as src:
                    rgb_data.append(src.read(1).astype(np.float32))
            rgb_data = np.stack(rgb_data, axis=0) # (3, H, W)
            
            # Warp/Reproject IR bands to reference
            ir_data = []
            for b_idx in config.data.ir_bands:
                ir_path = band_map[b_idx]
                with rasterio.open(ir_path) as src:
                    dest_band = np.zeros((height, width), dtype=np.float32)
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=dest_band,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=ref_transform,
                        dst_crs=ref_crs,
                        resampling=Resampling.cubic_spline
                    )
                    ir_data.append(dest_band)
            ir_data = np.stack(ir_data, axis=0) # (C_ir, H, W)
            
        else:
            # Multi-band scene file
            with rasterio.open(scene) as src:
                ref_meta = src.meta.copy()
                ref_transform = src.transform
                ref_crs = src.crs
                height, width = ref_meta['height'], ref_meta['width']
                
                # Check counts
                if src.count < max(config.data.rgb_bands + config.data.ir_bands):
                    print(f"Skipping stacked scene {scene_name}: insufficient bands ({src.count})")
                    continue
                
                rgb_data = src.read(config.data.rgb_bands).astype(np.float32)
                ir_data = src.read(config.data.ir_bands).astype(np.float32)
        
        # Windowed Tiling
        tile_count = 0
        for y in range(0, height - overlap, stride):
            for x in range(0, width - overlap, stride):
                w = min(tile_size, width - x)
                h = min(tile_size, height - y)
                if w != tile_size or h != tile_size:
                    continue
                
                window = Window(x, y, w, h)
                tile_transform = rasterio.windows.transform(window, ref_transform)
                
                # Crop slices
                tile_rgb = rgb_data[:, y:y+h, x:x+w]
                tile_ir = ir_data[:, y:y+h, x:x+w]
                
                tile_id = f"{scene_name}_tile_{y}_{x}"
                
                # Save RGB tile
                rgb_tile_meta = ref_meta.copy()
                rgb_tile_meta.update({
                    'driver': 'GTiff',
                    'height': h,
                    'width': w,
                    'count': 3,
                    'dtype': 'float32',
                    'crs': ref_crs,
                    'transform': tile_transform
                })
                
                with rasterio.open(tiles_dir / f"{tile_id}_rgb.tif", "w", **rgb_tile_meta) as dst:
                    dst.write(tile_rgb)
                    
                # Save IR tile
                ir_tile_meta = ref_meta.copy()
                ir_tile_meta.update({
                    'driver': 'GTiff',
                    'height': h,
                    'width': w,
                    'count': len(config.data.ir_bands),
                    'dtype': 'float32',
                    'crs': ref_crs,
                    'transform': tile_transform
                })
                
                with rasterio.open(tiles_dir / f"{tile_id}_ir.tif", "w", **ir_tile_meta) as dst:
                    dst.write(tile_ir)
                    
                tile_count += 1
                
        print(f"Created {tile_count} tiles for scene {scene_name}.")

if __name__ == "__main__":
    main()

