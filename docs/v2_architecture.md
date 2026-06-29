# V2 Architecture and Process Flow

This document outlines the proposed **Dual-Stream, Frequency-Decoupled State-Space Mamba Network with Online Semantic and Direction-Aligned Structural Guardrails** designed to solve ISRO Problem Statement 10.

## 1. Architecture Diagram

This diagram details the deep learning neural network components and how the Structural Network (VSSM) interacts with the Colorization Network (ControlNet).

```mermaid
flowchart TD
    subgraph "VSSM Structural Backbone (Linear Complexity)"
        IR_In["Low-Res IR Tensors (128x128)"] --> Stem["Stem Convolution"]
        Stem --> MainStream["Main Stream\n(State-Space Linear Attention)"]
        Stem --> FEM["Frequency Enhancement Module\n(Wavelet Transform / Haar)"]
      
        FEM -- "High-Frequency Textures" --> Fusion(("Feature Fusion"))
        MainStream -- "Global Context" --> Fusion
      
        Fusion --> Up["PixelShuffle Upsampling"]
        Up --> VSSM_Out["HR Structural Tensors (512x512)"]
    end
  
    subgraph "ControlNet-Guided Generator"
        VSSM_Out -- "Queries (Q)" --> CA["Multi-Head Cross-Attention Bridge\n(Injects boundaries to prevent color bleed)"]
        CA -- "Keys/Values (K,V)" --> UNet["U-Net Encoder-Decoder"]
        UNet --> RGB_Out["Predicted RGB (512x512)"]
    end
  
    subgraph "Multi-Task Guardrail Losses"
        RGB_Out -.-> L1["L1 Pixel-wise Loss"]
        RGB_Out -.-> L_Grad["Direction-Aligned Gradient Loss\n(Enforces crisp 90° edges on x/y axes)"]
        RGB_Out -.-> L_Sem["Frozen SegFormer Guardrail\n(Prevents structural hallucinations)"]
    end
```

## 2. Process Flow Diagram

This diagram illustrates the step-by-step data pipeline during inference/production deployment, from raw satellite ingestion to final QGIS-ready output.

```mermaid
sequenceDiagram
    participant Raw as Raw Landsat Data (B4, B5, B10)
    participant Pre as Geospatial Pre-Processor
    participant AI as V2 Deep Cascade Network
    participant Post as Tiling & Blending Engine
    participant GIS as Final GeoTIFF (QGIS Ready)

    Raw->>Pre: Ingest raw satellite bands
  
    Note over Pre: Step 1: Co-Registration<br/>Cubic Spline Reprojection to B4 Grid
    Note over Pre: Step 2: In-Memory Z-Score Normalization
    Note over Pre: Step 3: Extract 128x128 Overlapped Tiles
  
    Pre->>AI: Normalized Float32 Tensors
  
    activate AI
    Note over AI: VSSMNet extracts & upscales structure (4x)
    Note over AI: Cross-Attention injects edges to ControlNet
    Note over AI: ControlNet generates Vibrant RGB colors
    AI-->>Post: 512x512 RGB Prediction Tiles
    deactivate AI
  
    Note over Post: Step 4: 2D Cosine Blending Mask<br/>Smooths 64-pixel overlaps to prevent seams
    Note over Post: Step 5: Geotransform Scaling<br/>Adjust Affine matrix for 4x resolution
  
    Post->>GIS: Stitch tiles & save as 16-bit Float GeoTIFF
```

## 3. Structural Wireframe Diagram (Tensor Dimensions)

This wireframe maps the exact tensor shapes and transformations as data flows through the neural network layers. It serves as a technical blueprint for the PyTorch implementation.

```mermaid
flowchart TD
    %% Input
    Input["Input IR Patch\nShape: (B, 2, 128, 128)"]
  
    %% VSSM Block Wireframe
    subgraph "VSSM Backbone Wireframe"
        Conv1["Initial Conv2d\nShape: (B, 64, 128, 128)"]
        SSM["4x SSMLinearAttention\nShape: (B, 64, 128, 128)"]
        Haar["HaarDownsample\nShape: (B, 256, 64, 64)"]
        FEMProj["FEM Projection + Interpolate\nShape: (B, 64, 128, 128)"]
        Fusion["Element-wise Add\nShape: (B, 64, 128, 128)"]
        PixShuff["PixelShuffle Upsampling\nBridge Features (K,V)\nShape: (B, 64, 512, 512)"]
        ConvOut["HR IR Image\nShape: (B, 2, 512, 512)"]
    end
  
    %% ControlNet Wireframe
    subgraph "ControlNet Colorizer Wireframe"
        Enc1["Encoder Block 1\nShape: (B, 32, 512, 512)"]
        CA1["Cross-Attention 1\nShape: (B, 32, 512, 512)"]
        Pool1["MaxPool2d\nShape: (B, 32, 256, 256)"]
      
        Enc2["Encoder Block 2\nShape: (B, 64, 256, 256)"]
        CA2["Cross-Attention 2\nShape: (B, 64, 256, 256)"]
      
        Dec1["Decoder Block\nShape: (B, 32, 512, 512)"]
        Final["Final Conv2d (RGB)\nShape: (B, 3, 512, 512)"]
    end
  
    %% Connections
    Input --> Conv1
    Conv1 --> SSM
    Conv1 --> Haar --> FEMProj
    SSM --> Fusion
    FEMProj --> Fusion
    Fusion --> PixShuff
    PixShuff --> ConvOut
  
    PixShuff -- "Bridge Features (K,V)" --> CA1
    PixShuff -- "Bridge Features (K,V)" --> CA2
  
    ConvOut --> Enc1
    Enc1 -- "Query (Q)" --> CA1
    CA1 --> Pool1 --> Enc2
    Enc2 -- "Query (Q)" --> CA2
  
    CA2 --> Dec1
    CA1 -.-> Dec1
    Dec1 --> Final
```
