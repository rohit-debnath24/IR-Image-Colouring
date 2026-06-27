import base64
import requests
import os

DIAGRAM_PIPELINE = """graph TD
    B4[Red B4 - 30m]
    B3[Green B3 - 30m]
    B2[Blue B2 - 30m]
    B5[NIR B5 - 30m]
    B10[Thermal B10 - 100m]

    RS[Warp Reprojection]
    Stack[Stack RGB]
    Tiler[Tiling 512x512]
    TIFF[16-bit Float Tiles]

    B10 --> RS
    B5 --> RS
    B4 --> Stack
    B3 --> Stack
    B2 --> Stack
    RS --> Tiler
    Stack --> Tiler
    Tiler --> TIFF

    Norm[Z-Score Norm]
    SR[SRResNet 4x SR]
    Color[UNetColorizer]

    TIFF -- "Raw IR" --> Norm
    Norm --> SR
    SR -- "Upscaled IR" --> Color
    SR -- "Bridge Features" --> Color
    Color -- "RGB Prediction" --> Stitch[Cosine Blending]

    ScaleGeo[Scale Transform]
    OutTIFF[GeoTIFF RGB]

    Stitch --> ScaleGeo
    ScaleGeo --> OutTIFF
"""

DIAGRAM_CASCADE = """graph LR
    subgraph SRResNet Module
        InIR["Low-Res IR Input <br> (B, 2, 128, 128)"] --> Conv1["Conv 9x9 + PReLU"]
        Conv1 --> ResBlocks["8 Residual Blocks"]
        ResBlocks --> Conv2["Conv 3x3 + BatchNorm"]
        
        ResBlocks -- "Bridge Features <br> (B, 64, 128, 128)" --> ConcatPoint
        
        Conv2 --> PS["PixelShuffle Upsampler <br> (4x Spatial Scaling)"]
        PS --> ConvOut["Conv 9x9 (Output)"]
        ConvOut --> HighResIR["Upscaled IR prior <br> (B, 2, 512, 512)"]
    end

    subgraph UNetColorizer Module
        HighResIR --> Enc1["Encoder Level 1 <br> (512 -> 256)"]
        Enc1 --> Enc2["Encoder Level 2 <br> (256 -> 128)"]
        
        Enc2 --> ConcatPoint["Channel Concatenation <br> (Total: 128 channels)"]
        ConcatPoint --> Enc3["Encoder Level 3 <br> (128 -> 64)"]
        Enc3 --> Enc4["Encoder Level 4 <br> (64 -> 32)"]
        
        Enc4 --> Bottleneck["Bottleneck Layers <br> (Channel: 512)"]
        
        Bottleneck --> Dec4["Decoder Level 4 <br> (32 -> 64)"]
        Dec4 -- "Skip" --> Enc4
        Dec4 --> Dec3["Decoder Level 3 <br> (64 -> 128)"]
        Dec3 -- "Skip" --> Enc3
        Dec3 --> Dec2["Decoder Level 2 <br> (128 -> 256)"]
        Dec2 -- "Skip" --> Enc2
        Dec2 --> Dec1["Decoder Level 1 <br> (256 -> 512)"]
        Dec1 -- "Skip" --> Enc1
        
        Dec1 --> Final["Final Conv 1x1"]
        Final --> RGB["Visible RGB Output <br> (B, 3, 512, 512)"]
    end
"""

DIAGRAM_LOSS = """graph TD
    Pred["Predicted RGB"] --> L1[L1 Pixel Reconstruction Loss]
    GT["Ground Truth RGB"] --> L1

    Pred --> Grad[Gradient-Domain SSIM Loss]
    GT --> Grad

    Pred --> Sem[Semantic Consistency Loss]
    GT --> Sem
    Frozen["Frozen SegFormer Backbone"] --> Sem

    L1 --> L_total["Weighted Total Loss Optimizer"]
    Grad --> L_total
    Sem --> L_total

    subgraph Uncertainty Weighting Engine
        LV1["Learnable Log-Var s_1"] -- "exp(-s_1)" --> L_total
        LV2["Learnable Log-Var s_2"] -- "exp(-s_2)" --> L_total
        LV3["Learnable Log-Var s_3"] -- "exp(-s_3)" --> L_total
        
        LV1 -- "+ s_1" --> L_total
        LV2 -- "+ s_2" --> L_total
        LV3 -- "+ s_3" --> L_total
    end
"""

def fetch(syntax, name):
    encoded = base64.b64encode(syntax.encode('utf-8')).decode('utf-8')
    url = f"https://mermaid.ink/img/{encoded}"
    print(f"Fetching {name}...")
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            os.makedirs("outputs", exist_ok=True)
            with open(f"outputs/{name}", "wb") as f:
                f.write(r.content)
            print(f"Success: outputs/{name}")
        else:
            print(f"Failed with code {r.status_code}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    fetch(DIAGRAM_PIPELINE, "diag_pipeline.png")
    fetch(DIAGRAM_CASCADE, "diag_cascade.png")
    fetch(DIAGRAM_LOSS, "diag_loss.png")
