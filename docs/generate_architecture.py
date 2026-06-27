import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """Canvas for adding page numbers and running footers dynamically."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # Footer text
        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(612 - 54, 36, footer_text)
        self.drawString(54, 36, "IR-Colorize v2.0: Technical Architecture Specification (ISRO PS-10)")
        
        # Thin divider line above footer
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 50, 612 - 54, 50)
        
        self.restoreState()

def build_pdf(filename="IR_Colorize_v2_Architecture.pdf"):
    # Target 0.75-inch margins (54 points)
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=64
    )
    
    styles = getSampleStyleSheet()
    
    # Color Palette
    PRIMARY = colors.HexColor("#1E3A8A")   # Deep Slate Blue
    SECONDARY = colors.HexColor("#0D9488") # Teal Accent
    TEXT_COLOR = colors.HexColor("#1E293B")# Charcoal Text
    BG_LIGHT = colors.HexColor("#F8FAFC")  # Code/Table background
    
    # Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=PRIMARY,
        spaceAfter=4
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#475569"),
        spaceAfter=12
    )
    
    h1_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=PRIMARY,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=TEXT_COLOR,
        spaceAfter=6
    )
    
    code_style = ParagraphStyle(
        'CodeBlock',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#0F172A"),
        backColor=BG_LIGHT,
        borderColor=colors.HexColor("#E2E8F0"),
        borderWidth=0.5,
        borderPadding=8,
        spaceAfter=8
    )

    story = []
    
    # --- HEADER BLOCK ---
    story.append(Paragraph("IR-Colorize v2.0", title_style))
    story.append(Paragraph("Advanced Multimodal System Workflow & Architecture (ISRO Problem Statement 10)", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY, spaceAfter=12))
    
    # --- SECTION 1 ---
    story.append(Paragraph("1. Upgraded Architecture & Dataflow Overview", h1_style))
    intro_p1 = (
        "The IR-Colorize v2.0 framework updates the baseline sequential cascade into an "
        "<b>Interlinked Feature-Bridged Cascade Architecture</b> specifically optimized for earth observation "
        "and object interpretation. Instead of processing structural enhancement and colorization independently, "
        "high-frequency edge mappings from the Super-Resolution decoder are fused into the generative pipeline via cross-attention "
        "gates. This satisfies the core requirements of <b>ISRO Problem Statement 10</b> by sharpening faint textures while "
        "eliminating color bleeding and generative hallucinations."
    )
    story.append(Paragraph(intro_p1, body_style))
    
    diagram_text = (
        "┌──────────────────────────────┐<br/>"
        "│  Phase 1: Data Ingestion      │ ──► Landsat 8/9 Scenes (16-bit HDR GeoTIFFs)<br/>"
        "└──────────────┬───────────────┘<br/>"
        "               ▼<br/>"
        "┌──────────────────────────────┐<br/>"
        "│  Phase 2: Local Normalization│ ──► Active Per-Tile Z-Score / MinMax<br/>"
        "└──────────────┬───────────────┘<br/>"
        "               ▼<br/>"
        "┌─────────────────────────────────────────────────────────────────────────────┐<br/>"
        "│  Phase 3: Deep Feature-Bridged Cascade Pipeline                             │<br/>"
        "│                                                                             │<br/>"
        "│   Low-Res IR  ──►  [ Stage 1: Super-Resolution (Real-ESRGAN/SRGAN) ]        │<br/>"
        "│                                      │                                      │<br/>"
        "│                                      │ (Cross-Attention Skip Connections)   │<br/>"
        "│                                      ▼                                      │<br/>"
        "│   High-Res IR Prior  ──►  [ Stage 2: ControlNet-Guided Latent Diffusion ]   │<br/>"
        "└──────────────────────────────────────┬──────────────────────────────────────┘<br/>"
        "               ▼<br/>"
        "┌─────────────────────────────────────────────────────────────────────────────┐<br/>"
        "│  Phase 4: Multi-Task Training & Dynamic Loss Guardrails                     │<br/>"
        "│                                                                             │<br/>"
        "│                    ┌──► L_standard (Pixel L1 + Adversarial)                 │<br/>"
        "│   Predicted RGB ───┼──► L_grad (Gradient Domain SSIM / Edge Constraint)      │<br/>"
        "│                    └──► L_sem (Frozen SegFormer/U-Net KL-Divergence)        │<br/>"
        "│                                                                             │<br/>"
        "│   [ Self-Governing Optimization via Homoscedastic Uncertainty Weighting ]    │<br/>"
        "└──────────────────────────────────────┬──────────────────────────────────────┘<br/>"
        "               ▼<br/>"
        "┌─────────────────────────────────────────────────────────────────────────────┐<br/>"
        "│  Phase 5: Accelerated GIS Inference                                         │<br/>"
        "│                                                                             │<br/>"
        "│   On-The-Fly Tiling ──► TensorRT FP16 Execution ──► Cosine Seam Feathering  │<br/>"
        "└─────────────────────────────────────────────────────────────────────────────┘"
    )
    story.append(Paragraph(diagram_text, code_style))
    story.append(Spacer(1, 4))
    
    # --- SECTION 2 ---
    story.append(Paragraph("2. Detailed End-to-End Workflow Phases", h1_style))
    
    phases = [
        ("Phase 1: Geospatial Ingestion & Alignment", 
         "Raw Landsat 8/9 Level-2 bands are mapped via <i>rasterio</i>. Near-Infrared (Band 5) and Thermal "
         "Infrared (Band 10) rasters are dynamically reprojected and resampled using cubic spline interpolation to match the high-resolution visible coordinate grid. Aligned grids are chunked into 512x512 windows with 64px overlaps and saved as 16-bit Float GeoTIFFs to protect metadata (CRS/Geotransform) and prevent dynamic range crushing."),
        
        ("Phase 2: Adaptive Local Normalization", 
         "Global image scaling fails across scenes due to varying regional thermal signatures across seasons. The framework computes Z-Score or MinMax statistical parameters independently on each individual tile using active pixel histograms to balance local contrast anomalies."),
         
        ("Phase 3: Interlinked Feature-Bridged Cascade", 
         "The low-resolution IR inputs target a primary Super-Resolution network (Real-ESRGAN/SRGAN). Beyond producing upscaled images, structural intermediate feature maps from the network's internal decoder are linked directly via cross-attention layers into the latent layers of the Colorization Engine (ControlNet-Guided Latent Diffusion). This forces generative texture layouts to directly trace physical boundaries instead of leaking color outlines."),
         
        ("Phase 4: Multi-Task Losses & Homoscedastic Weights", 
         "To block hallucination, outputs are evaluated concurrently against a structural Gradient-Domain SSIM Loss and a Semantic Consistency Loss (built on a frozen SegFormer engine that heavily penalizes land-cover shifts via KL-Divergence). Task balancing is entirely self-governing; weights are dynamically computed at backpropagation using trainable homoscedastic uncertainty parameters."),
         
        ("Phase 5: Production Edge Inference", 
         "Full satellite scenes are parsed on-the-fly and processed through neural weights compiled into native <b>TensorRT FP16 execution graphs</b>. Reconstructed mosaic seams undergo distance-weighted cosine feathering before the original CRS headers are passed directly back to compile the final georeferenced output file.")
    ]
    
    for title, desc in phases:
        story.append(Paragraph(f"<b>• {title}</b>", body_style))
        story.append(Paragraph(desc, ParagraphStyle('Indented', parent=body_style, leftIndent=12, spaceAfter=5)))
    
    story.append(Spacer(1, 4))

    # --- SECTION 3 ---
    story.append(Paragraph("3. Production YAML Deployment Specifications", h1_style))
    
    yaml_mock = (
        "system:<br/>"
        "  version: \"2.0\"<br/>"
        "  device: \"cuda\"<br/>"
        "  mixed_precision: \"fp16\"<br/>"
        "stage: \"joint\" # Choices: [sr, color, joint]<br/>"
        "data:<br/>"
        "  ingestion:<br/>"
        "    tile_size: 512<br/>"
        "    overlap: 64<br/>"
        "    format: \"Float32_GeoTIFF\"<br/>"
        "  normalization:<br/>"
        "    algorithm: \"per_tile_zscore\"<br/>"
        "model:<br/>"
        "  sr:<br/>"
        "    arch: \"real_esrgan\"<br/>"
        "    extract_bridge_features: true<br/>"
        "  color:<br/>"
        "    arch: \"latent_diffusion_controlnet\"<br/>"
        "    conditioning_type: \"cross_attention_plus_spatial\"<br/>"
        "optimization:<br/>"
        "  loss_weighting: \"homoscedastic_uncertainty\" # Adaptive auto-tuning<br/>"
        "inference:<br/>"
        "  compiler: \"tensorrt\"<br/>"
        "  precision: \"fp16\"<br/>"
        "  blending_algorithm: \"cosine_feathering\""
    )
    story.append(Paragraph(yaml_mock, code_style))
    story.append(Spacer(1, 4))

    # --- SECTION 4 ---
    story.append(Paragraph("4. Target Evaluation Benchmarks", h1_style))
    
    # Robust raw string layout to eliminate multi-argument mismatch errors across reportlab versions
    table_data = [
        ["Evaluation Metric", "Legacy Pipeline (v1.0)", "Upgraded Architecture (v2.0)"],
        ["Peak Signal-to-Noise Ratio (PSNR)", "> 28.0 dB", "> 31.5 dB"],
        ["Structural Similarity Index (SSIM)", "> 0.85", "> 0.92"],
        ["Average Inference Latency (Per Tile)", "< 500 ms", "< 85 ms (TensorRT FP16)"],
        ["Semantic Hallucination Artifacts", "Frequent / Unbounded", "Mitigated via Frozen SegFormer"]
    ]
    
    # 504 points total available printable space (612 page width - 108 margin)
    t = Table(table_data, colWidths='')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#CBD5E1")),
        ('TEXTCOLOR', (0,0), (-1,0), PRIMARY),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 9.5),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, BG_LIGHT]),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,1), (-1,-1), 9),
        ('TEXTCOLOR', (0,1), (-1,-1), TEXT_COLOR),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    
    story.append(t)
    
    # Compile the final layout using our dynamic two-pass canvas handler
    doc.build(story, canvasmaker=NumberedCanvas)

if __name__ == "__main__":
    build_pdf()