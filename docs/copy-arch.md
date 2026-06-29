wwwwqw

# ISRO Problem Statement 10: Architecture Design Document

## Dual-Stream, Frequency-Decoupled State-Space Mamba Network with Online Semantic and Direction-Aligned Structural Guardrails

---

## 1. High-Level Architecture Overview

This architecture is specifically designed to resolve engineering failures in remote sensing image processing:

- **Seam Artifacts**: Caused by block-wise tiled inference without geometric edge blending
- **Grayscale/Desaturation Trap**: Where models collapse into monotone palettes due to standard pixel-wise losses

```mermaid
flowchart TB
    subgraph Input[Input Stage]
        LR[Low-Resolution Raw IR Input]
    end
  
    subgraph Backbone[State-Space Backbone - VSSM]
        VSSM[Visual State-Space Model<br/>Linear Complexity O of N]
        direction TB
    end
  
    subgraph FEM[Frequency Enhancement Module]
        WD[Multi-Scale Wavelet Decomposition]
        WS[Wavelet Stream - High-Frequency Details]
        SS[Spatial Stream - Color Representation]
    end
  
    subgraph Skip[Cross-Attention Skip Connections]
        CA[Feature-Level Cross-Attention Blocks]
    end
  
    subgraph Output[Generative Output]
        CN[ControlNet-Guided RGB Output]
    end
  
    LR --> VSSM
    VSSM --> WD
    WD --> WS
    WD --> SS
    WS --> CA
    SS --> CA
    CA --> CN
```

---

## 2. Detailed Pipeline Architecture

```mermaid
flowchart TD
    subgraph Stage1[Stage 1: Input Processing]
        IR[Raw IR Satellite Image<br/>Low Resolution]
        PRE[Preprocessing<br/>Normalization + Padding]
        IR --> PRE
    end
  
    subgraph Stage2[Stage 2: VSSM Backbone]
        SSM1[State-Space Block 1]
        SSM2[State-Space Block 2]
        SSM3[State-Space Block N]
        SSMLN[Layer Normalization]
      
        PRE --> SSM1
        SSM1 --> SSM2
        SSM2 --> SSM3
        SSM3 --> SSMLN
    end
  
    subgraph Stage3[Stage 3: Frequency Decoupling]
        DWT[2D Discrete Wavelet Transform]
        LL[LL Band - Low-Low<br/>Approximation]
        LH[LH Band - Low-High<br/>Horizontal Edges]
        HL[HL Band - High-Low<br/>Vertical Edges]
        HH[HH Band - High-High<br/>Diagonal Details]
      
        SSMLN --> DWT
        DWT --> LL
        DWT --> LH
        DWT --> HL
        DWT --> HH
    end
  
    subgraph Stage4[Stage 4: Dual-Stream Processing]
        subgraph TextureStream[Texture Stream]
            TEX1[Conv Block 3x3]
            TEX2[Attention Block]
            TEX3[Feature Fusion]
            LH --> TEX1
            HL --> TEX1
            HH --> TEX1
            TEX1 --> TEX2
            TEX2 --> TEX3
        end
      
        subgraph ColorStream[Color Stream]
            COL1[Conv Block 3x3]
            COL2[Global Average Pooling]
            COL3[Color Embedding]
            LL --> COL1
            COL1 --> COL2
            COL2 --> COL3
        end
    end
  
    subgraph Stage5[Stage 5: Cross-Attention Fusion]
        CAF1[Cross-Attention Block 1]
        CAF2[Cross-Attention Block 2]
        CAF3[Feature Upsampling]
      
        TEX3 --> CAF1
        COL3 --> CAF1
        CAF1 --> CAF2
        CAF2 --> CAF3
    end
  
    subgraph Stage6[Stage 6: ControlNet Generation]
        CTRL[ControlNet Encoder]
        GEN[Generative Decoder]
        RGB[High-Resolution RGB Output]
      
        CAF3 --> CTRL
        CTRL --> GEN
        GEN --> RGB
    end
```

---

## 3. Visual State-Space Model (VSSM) Architecture

The VSSM backbone replaces traditional CNNs and Vision Transformers with linear complexity state-space modeling.

```mermaid
flowchart LR
    subgraph VSSMBlock[VSSM Block]
        direction TB
        IN[Input Feature Map<br/>H x W x C]
      
        subgraph SS2D[2D Selective Scan]
            direction TB
            EXP[Linear Expansion]
            SCAN1[Horizontal Scan]
            SCAN2[Vertical Scan]
            SCAN3[Reverse Horizontal]
            SCAN4[Reverse Vertical]
            MERGE[Feature Merge]
          
            EXP --> SCAN1
            EXP --> SCAN2
            EXP --> SCAN3
            EXP --> SCAN4
            SCAN1 --> MERGE
            SCAN2 --> MERGE
            SCAN3 --> MERGE
            SCAN4 --> MERGE
        end
      
        subgraph MambaCore[Mamba SSM Core]
            direction LR
            HSTATE[Hidden State h_t]
            SSM[State Equation: h_t = A h_t-1 + B x_t]
            OUT[Output: y_t = C h_t]
          
            HSTATE --> SSM
            SSM --> OUT
        end
      
        PROJ[Linear Projection]
        OUT1[Output Feature Map<br/>H x W x C]
      
        IN --> SS2D
        SS2D --> MambaCore
        MambaCore --> PROJ
        PROJ --> OUT1
    end
```

### Key Advantages of VSSM:

| Aspect                  | CNN     | Vision Transformer | VSSM/Mamba |
| ----------------------- | ------- | ------------------ | ---------- |
| Receptive Field         | Limited | Global             | Global     |
| Complexity              | O of N  | O of N squared     | O of N     |
| Memory for Large Images | Low     | High               | Low        |
| Long-range Dependencies | Weak    | Strong             | Strong     |

---

## 4. Frequency Enhancement Module (FEM)

The FEM uses multi-level discrete wavelet decomposition to decouple local textures from global styles.

```mermaid
flowchart TB
    subgraph FEM_Module[Frequency Enhancement Module]
        INPUT[Input Features<br/>H x W x C]
      
        subgraph DWT_L1[Level 1 DWT]
            DWT1[2D DWT]
            LL1[LL1 - Approximation]
            HF1[High-Frequency Bands<br/>LH1, HL1, HH1]
        end
      
        subgraph DWT_L2[Level 2 DWT]
            DWT2[2D DWT]
            LL2[LL2 - Approximation]
            HF2[High-Frequency Bands<br/>LH2, HL2, HH2]
        end
      
        subgraph DWT_L3[Level 3 DWT]
            DWT3[2D DWT]
            LL3[LL3 - Coarse Approximation]
            HF3[High-Frequency Bands<br/>LH3, HL3, HH3]
        end
      
        subgraph TextureBranch[Texture Enhancement Branch]
            direction TB
            HF1 --> TCONV1[Conv 3x3 + ReLU]
            HF2 --> TCONV2[Conv 3x3 + ReLU]
            HF3 --> TCONV3[Conv 3x3 + ReLU]
            TCONV1 --> TFUSE[Multi-Scale Fusion]
            TCONV2 --> TFUSE
            TCONV3 --> TFUSE
        end
      
        subgraph ColorBranch[Color Representation Branch]
            direction TB
            LL3 --> CCONV[Conv 1x1]
            CCONV --> CGAP[Global Average Pooling]
            CGAP --> CFC[Fully Connected]
            CFC --> CEMB[Color Embedding Vector]
        end
      
        INPUT --> DWT1
        LL1 --> DWT2
        LL2 --> DWT3
    end
  
    TFUSE --> TEXTURE_OUT[Enhanced Texture Features]
    CEMB --> COLOR_OUT[Color Representation Vector]
```

### Wavelet Band Descriptions:

- **LL (Low-Low)**: Approximation coefficients - contains global structure and color information
- **LH (Low-High)**: Horizontal edge details
- **HL (High-Low)**: Vertical edge details
- **HH (High-High)**: Diagonal edge details

---

## 5. Cross-Attention Skip Connections

Cross-attention ensures that boundaries recovered in the infrared domain strictly guide color boundaries.

```mermaid
flowchart LR
    subgraph CrossAttention[Cross-Attention Skip Connection]
        direction TB
      
        subgraph QueryStream[Query Stream - Color]
            QIN[Color Features F_c]
            QPROJ[Linear Projection]
            Q[Query Q]
            QIN --> QPROJ --> Q
        end
      
        subgraph KeyValStream[Key-Value Stream - Texture]
            KIN[Texture Features F_t]
            KPROJ[Linear Projection]
            VPROJ[Linear Projection]
            K[Key K]
            V[Value V]
            KIN --> KPROJ --> K
            KIN --> VPROJ --> V
        end
      
        subgraph Attention[Attention Mechanism]
            ATT[Attention Score<br/>softmax of QK^T / sqrt d]
            MUL[Multiply by V]
            OUT[Attended Features]
            Q --> ATT
            K --> ATT
            ATT --> MUL
            V --> MUL
            MUL --> OUT
        end
      
        ADD[Residual Addition]
        NORM[Layer Normalization]
        RESULT[Fused Features]
      
        OUT --> ADD
        QIN --> ADD
        ADD --> NORM --> RESULT
    end
```

### Mathematical Formulation:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

Where:
- Q = Linear_Q(F_color)  [Query from color stream]
- K = Linear_K(F_texture) [Key from texture stream]
- V = Linear_V(F_texture) [Value from texture stream]
- d_k = dimension of key vectors
```

---

## 6. Multi-Task Loss Function Architecture

```mermaid
flowchart TB
    subgraph LossArchitecture[Multi-Task Loss Formulation]
        subgraph Inputs[Model Outputs]
            PRED[Predicted RGB]
            GT[Ground Truth RGB]
        end
      
        subgraph AdversarialLoss[Adversarial Loss L_adv]
            DISC[Discriminator Network]
            FAKE[Fake Detection Score]
            REAL[Real Detection Score]
            ADV[Binary Cross-Entropy]
          
            PRED --> DISC --> FAKE --> ADV
            GT --> REAL --> ADV
        end
      
        subgraph GradientLoss[Direction-Aligned Gradient Loss L_dir-grad]
            direction TB
            subgraph PredGrad[Predicted Gradients]
                PGX[Gradient X: nabla_x pred]
                PGY[Gradient Y: nabla_y pred]
            end
            subgraph GTGrad[GT Gradients]
                GGX[Gradient X: nabla_x gt]
                GGY[Gradient Y: nabla_y gt]
            end
            subgraph MultiScale[Multi-Scale Computation]
                S1[Scale 1: Full Resolution]
                S2[Scale 2: 2x Downsampled]
                S3[Scale 3: 4x Downsampled]
            end
            GLOSS[L_dir-grad = Sum over scales of L1 of nabla_x + L1 of nabla_y]
          
            PRED --> PGX
            PRED --> PGY
            GT --> GGX
            GT --> GGY
            PGX --> S1
            PGY --> S1
            GGX --> S1
            GGY --> S1
            S1 --> GLOSS
            S2 --> GLOSS
            S3 --> GLOSS
        end
      
        subgraph SemanticLoss[Online Semantic Consistency Loss L_sem]
            direction TB
            SEGFORMER[Frozen SegFormer<br/>Pre-trained Land-Cover Classifier]
            PREDLOG[Predicted Logits P_pred]
            GTLOG[GT Logits P_gt]
            KL[KL Divergence<br/>D_KL of P_gt || P_pred]
          
            PRED --> SEGFORMER --> PREDLOG --> KL
            GT --> SEGFORMER --> GTLOG --> KL
        end
      
        subgraph TotalLoss[Total Loss]
            LAMBDA1[lambda_1]
            LAMBDA2[lambda_2]
            LAMBDA3[lambda_3]
            COMBINE[L_total = lambda_1 * L_adv + lambda_2 * L_dir-grad + lambda_3 * L_sem]
          
            ADV --> COMBINE
            LAMBDA1 --> COMBINE
            GLOSS --> COMBINE
            LAMBDA2 --> COMBINE
            KL --> COMBINE
            LAMBDA3 --> COMBINE
        end
    end
```

### Loss Function Equations:

**Total Loss:**

```
L_total = lambda_1 * L_adv + lambda_2 * L_dir-grad + lambda_3 * L_sem
```

**Direction-Aligned Multi-Scale Gradient Loss:**

```
L_dir-grad = Sum over s in scales of [L1(nabla_x pred_s, nabla_x gt_s) + L1(nabla_y pred_s, nabla_y gt_s)]
```

**Online Semantic Consistency Loss:**

```
L_sem = D_KL(P_gt || P_pred) = Sum over c of P_gt(c) * log(P_gt(c) / P_pred(c))
```

---

## 7. Deployment & Inference Pipeline

```mermaid
flowchart TB
    subgraph InferencePipeline[Inference Pipeline with Seamless Tiling]
        subgraph Input[Input Processing]
            LARGE[Large Satellite Image<br/>Multi-Gigabyte]
            TILE[Tile Decomposition<br/>512x512 with 64px Overlap]
            LARGE --> TILE
        end
      
        subgraph Processing[Tile Processing]
            direction TB
            TILES[Tile Batch]
            PAD[Reflection Padding]
            MODEL[VSSM-FEM Model]
            UNPAD[Remove Padding]
            OUTTILE[Output Tile]
          
            TILES --> PAD --> MODEL --> UNPAD --> OUTTILE
        end
      
        subgraph Blending[Seamless Stitching]
            direction TB
            subgraph CosineBlend[2D Cosine Blending Mask]
                FORMULA[w_blend x = 1/2 * 1 + cos of pi*x/d_over]
                VISUAL[Smooth Alpha Transition<br/>0 at edges, 1 at center]
            end
            APPLY[Apply Blending Weights]
            STITCH[Tile Composition]
          
            OUTTILE --> APPLY
            CosineBlend --> APPLY
            APPLY --> STITCH
        end
      
        subgraph Output[Geospatial Output]
            COMPILE[Compile Output Tiles]
            GEOTIFF[16-bit Float GeoTIFF]
            CRS[Preserve CRS + Affine Transform]
            GIS[Ready for QGIS/ArcGIS]
          
            STITCH --> COMPILE --> GEOTIFF
            GEOTIFF --> CRS --> GIS
        end
    end
```

### Cosine Blending Weight Formula:

```
w_blend(x) = 1/2 * [1 + cos(pi * x / d_over)]

Where:
- x = position within overlap region [0, d_over]
- d_over = overlap distance (64 pixels)
```

---

## 8. Complete System Architecture

```mermaid
flowchart TB
    subgraph CompleteSystem[Complete ISRO PS-10 System]
        subgraph DataIngestion[Data Ingestion Layer]
            SAT[ISRO Satellite Data]
            META[Metadata Extraction]
            PREPROC[Preprocessing Pipeline]
            SAT --> META --> PREPROC
        end
      
        subgraph TrainingPipeline[Training Pipeline]
            direction TB
            TRAIN_DATA[Training Dataset<br/>IR-RGB Pairs]
            AUG[Data Augmentation<br/>Rotation, Flip, Scale]
            MODEL[VSSM-FEM Network]
            LOSS[Multi-Task Loss]
            OPT[AdamW Optimizer]
            CHECK[Checkpoint Saving]
          
            TRAIN_DATA --> AUG --> MODEL --> LOSS --> OPT --> CHECK
        end
      
        subgraph InferencePipeline[Inference Pipeline]
            direction TB
            INF_INPUT[Input IR Image]
            TILE_PROC[Tiled Processing<br/>512x512 + 64px overlap]
            BLEND[Cosine Blending]
            GEO_OUT[GeoTIFF Output]
          
            INF_INPUT --> TILE_PROC --> BLEND --> GEO_OUT
        end
      
        subgraph Validation[Validation & Metrics]
            direction TB
            PSNR[PSNR]
            SSIM[SSIM]
            FID[FID Score]
            SEM[Semantic Accuracy]
        end
      
        PREPROC --> TrainingPipeline
        PREPROC --> InferencePipeline
        GEO_OUT --> Validation
    end
```

---

## 9. Key Design Decisions Summary

| Component                      | Design Choice               | Rationale                                                                  |
| ------------------------------ | --------------------------- | -------------------------------------------------------------------------- |
| **Backbone**             | VSSM/Mamba                  | Linear O(N) complexity for large satellite images, global receptive field  |
| **Frequency Decoupling** | Multi-level DWT             | Separates texture (high-freq) from color (low-freq), prevents entanglement |
| **Skip Connections**     | Cross-Attention             | Ensures IR boundaries guide color boundaries, prevents bleeding            |
| **Loss Function**        | Multi-task composite        | Prevents grayscale collapse, enforces structural and semantic consistency  |
| **Tiling Strategy**      | 64px overlap + cosine blend | Eliminates seam artifacts from block-wise inference                        |
| **Output Format**        | 16-bit Float GeoTIFF        | Preserves CRS and geotransform for GIS integration                         |

---

## 10. Implementation Recommendations

### Framework Stack:

- **Deep Learning**: PyTorch 2.0+ with `mamba-ssm` package
- **Wavelet Transforms**: PyWavelets (`pywt`)
- **Geospatial**: `rasterio`, `GDAL` for GeoTIFF I/O
- **Semantic Loss**: Pre-trained SegFormer from `transformers`

### Training Configuration:

```yaml
batch_size: 8
tile_size: 512
overlap: 64
learning_rate: 1e-4
lambda_1: 1.0  # Adversarial weight
lambda_2: 10.0  # Gradient loss weight
lambda_3: 5.0   # Semantic loss weight
epochs: 200
```

### Hardware Requirements:

- **Training**: 4x NVIDIA A100 80GB (multi-GPU DDP)
- **Inference**: 1x NVIDIA RTX 4090 24GB (single GPU sufficient)

---

*Document generated for ISRO Problem Statement 10 - Satellite Image Super-Resolution and Colorization*