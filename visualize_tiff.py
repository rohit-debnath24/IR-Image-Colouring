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
        # Read RGB bands
        data = src.read() # Shape: (3, H, W)
        
    # Scale each band to 0-255 using percentile scaling to remove outlier contrast compression
    img_channels = []
    for c in range(3):
        channel = data[c]
        
        # Calculate 2nd and 98th percentiles
        p2 = np.percentile(channel, 2)
        p98 = np.percentile(channel, 98)
        
        if p98 > p2:
            channel_clipped = np.clip(channel, p2, p98)
            channel_scaled = ((channel_clipped - p2) / (p98 - p2) * 255.0).astype(np.uint8)
        else:
            channel_scaled = np.zeros_like(channel, dtype=np.uint8)
        img_channels.append(channel_scaled)
        
    img_rgb = np.stack(img_channels, axis=2) # Shape: (H, W, 3)
    
    # Save as PNG
    print(f"Saving visualized PNG to: {args.output}")
    Image.fromarray(img_rgb).save(args.output)
    print("Done!")

if __name__ == "__main__":
    main()
