import argparse
import rasterio
import numpy as np
from PIL import Image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input GeoTIFF file")
    parser.add_argument("--output", required=True, help="Output PNG file")
    args = parser.parse_args()
    
    print(f"Reading GeoTIFF: {args.input}")
    with rasterio.open(args.input) as src:
        # Read bands
        data = src.read() # Shape: (C, H, W)
        
    num_channels = data.shape[0]
    
    # For RGB output, scale all channels together using joint percentile statistics
    # to preserve the relative color channel offsets (ratios).
    if num_channels >= 3:
        p2 = np.percentile(data[:3], 2)
        p98 = np.percentile(data[:3], 98)
        
        img_channels = []
        for c in range(num_channels):
            channel = data[c]
            if p98 > p2:
                channel_clipped = np.clip(channel, p2, p98)
                channel_scaled = ((channel_clipped - p2) / (p98 - p2) * 255.0).astype(np.uint8)
            else:
                channel_scaled = np.zeros_like(channel, dtype=np.uint8)
            img_channels.append(channel_scaled)
    else:
        # Scale each band to 0-255 using per-channel percentile scaling
        img_channels = []
        for c in range(num_channels):
            channel = data[c]
            p2 = np.percentile(channel, 2)
            p98 = np.percentile(channel, 98)
            if p98 > p2:
                channel_clipped = np.clip(channel, p2, p98)
                channel_scaled = ((channel_clipped - p2) / (p98 - p2) * 255.0).astype(np.uint8)
            else:
                channel_scaled = np.zeros_like(channel, dtype=np.uint8)
            img_channels.append(channel_scaled)
        
    if num_channels == 1:
        # Grayscale mapping
        img_rgb = np.stack([img_channels[0]] * 3, axis=2)
    elif num_channels == 2:
        # False color: Band 0 (NIR) -> Red, Band 1 (Thermal) -> Green, Blue -> Zero
        zeros = np.zeros_like(img_channels[0], dtype=np.uint8)
        img_rgb = np.stack([img_channels[0], img_channels[1], zeros], axis=2)
    else:
        # Standard RGB (take first 3 channels)
        img_rgb = np.stack(img_channels[:3], axis=2)
        
    # Save as PNG
    print(f"Saving visualized PNG to: {args.output}")
    Image.fromarray(img_rgb).save(args.output)
    print("Done!")

if __name__ == "__main__":
    main()
