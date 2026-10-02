"""Generate pristine, publication-grade deployment architecture diagram for Thesis Chapter 2.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Setup canvas: wide publication format (15.4 x 9.8 in)
fig, ax = plt.subplots(figsize=(15.4, 9.8), dpi=300)
ax.set_xlim(0, 15.4)
ax.set_ylim(0, 9.8)
ax.axis("off")

# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------
def draw_tier_panel(ax, x, y, w, h, title, bg_color, border_color, badge_bg, badge_fg="#FFFFFF"):
    """Draw a tier background panel with an elegant header pill badge placed safely."""
    p = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.10",
        facecolor=bg_color,
        edgecolor=border_color,
        linewidth=0.8,
        linestyle="-",
        zorder=1
    )
    ax.add_patch(p)
    # Header badge centered horizontally
    bx = x + w / 2.0
    ax.text(
        bx, y + h, f"  {title}  ",
        fontsize=8.3, fontweight="bold", color=badge_fg,
        fontfamily="sans-serif", va="center", ha="center", zorder=3,
        bbox=dict(boxstyle="round,pad=0.26", facecolor=badge_bg, edgecolor="none")
    )

def draw_card(ax, x, y, w, h, title, subtitle, items, border, title_col, sub_col="#475569", bg="#FFFFFF"):
    """Draw a crisp white card with colored border, title, subtitle, and bullet items."""
    p = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.10",
        facecolor=bg,
        edgecolor=border,
        linewidth=1.3,
        zorder=4
    )
    ax.add_patch(p)
    # Title
    ax.text(x + 0.22, y + h - 0.24, title, fontsize=9.5, fontweight="bold",
            color=title_col, fontfamily="sans-serif", va="top", zorder=5)
    # Subtitle
    if subtitle:
        ax.text(x + 0.22, y + h - 0.50, subtitle, fontsize=7.5, fontweight="bold",
                color=sub_col, fontfamily="sans-serif", va="top", zorder=5)
        text_y = y + h - 0.76
    else:
        text_y = y + h - 0.54

    # Bullet items
    item_str = "\n".join(f"• {it}" for it in items)
    ax.text(x + 0.22, text_y, item_str, fontsize=7.2, color="#1E293B",
            fontfamily="sans-serif", va="top", linespacing=1.28, zorder=5)

def draw_arrow(ax, start, end, label, color="#2563EB", bg_col="#EFF6FF", border_col="#BFDBFE", text_col="#1E40AF"):
    """Draw a clean flow arrow with an annotated pill badge centered in the clear gutter."""
    ax.annotate(
        "", xy=end, xytext=start,
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=2.0,
            mutation_scale=14,
            shrinkA=1,
            shrinkB=1
        ),
        zorder=6
    )
    # Midpoint
    mx = (start[0] + end[0]) / 2.0
    my = (start[1] + end[1]) / 2.0
    ax.text(
        mx, my, label,
        fontsize=6.8, fontweight="bold", color=text_col,
        ha="center", va="center", zorder=8,
        bbox=dict(boxstyle="round,pad=0.24", facecolor=bg_col, edgecolor=border_col, lw=0.8)
    )

# =========================================================================
# TIER 1: CLIENT PRESENTATION LAYER (Next.js 14)
# Y: 7.20 to 9.55 (h = 2.35)
# =========================================================================
draw_tier_panel(
    ax, x=0.5, y=7.20, w=14.4, h=2.35,
    title="TIER 1: CLIENT PRESENTATION TIER (NEXT.JS 14 ON VERCEL)",
    bg_color="#F8FAFC", border_color="#CBD5E1", badge_bg="#1E293B"
)

# Card 1.1: Client Ingestion Interface (Top-Left)
draw_card(
    ax, x=0.8, y=7.35, w=6.4, h=1.95,
    title="Client Ingestion Interface (Next.js 14)",
    subtitle="Web Browser / Forensic Analyst Workstation",
    items=[
        "Multi-Format Ingestion: Video (MP4, AVI, WebM) or Audio (WAV, MP3, FLAC)",
        "Client-Side Pre-validation: MIME type inspection & integrity SHA-256 check",
        "Resilient Upload Streaming: Ephemeral upload tracking & progress telemetry",
        "Adaptive Path Routing: Dispatches to /predict (A/V) or /predict-audio (Audio)"
    ],
    border="#2563EB", title_col="#1E3A8A", sub_col="#2563EB"
)

# Card 1.2: Forensic Verdict Dashboard (Top-Right)
draw_card(
    ax, x=8.2, y=7.35, w=6.4, h=1.95,
    title="Forensic Verdict Dashboard (Client UI)",
    subtitle="Interactive Human-in-the-Loop Inspection View",
    items=[
        "Per-Modality Risk Gauges: Visual face manipulation vs voice spoof meters",
        "4-Quadrant Attribution Badge: Discrete classification (RVRA / RVFA / FVRA / FVFA)",
        "Dual Temporal Timelines: Frame-accurate manipulation boundary overlay",
        "Audio-Visual Sync Curve: Phoneme-viseme temporal agreement trajectory",
        "Mandatory Legal Disclaimer: 'Probabilistic decision support -- not autonomous legal proof'"
    ],
    border="#059669", title_col="#065F46", sub_col="#059669"
)

# =========================================================================
# TIER 2: EDGE SECURITY & SERIALIZATION GATEWAY
# Y: 4.25 to 6.45 (h = 2.20)
# =========================================================================
draw_tier_panel(
    ax, x=0.5, y=4.25, w=14.4, h=2.20,
    title="TIER 2: EDGE SECURITY GATEWAY & SERIALIZATION CONTRACT LAYER",
    bg_color="#F8FAFC", border_color="#CBD5E1", badge_bg="#334155"
)

# Card 2.1: Edge Security & API Gateway (Middle-Left)
draw_card(
    ax, x=0.8, y=4.40, w=6.4, h=1.80,
    title="Edge Security & Reverse Proxy (Vercel Edge)",
    subtitle="Serverless Ingestion Proxy & Traffic Controller",
    items=[
        "Payload Sanitization: Strict MIME validation & upload size cap (<= 50 MB)",
        "Threat Protection: Sliding-window IP rate limiting & CORS origin security",
        "Secure Reverse Tunnel: Authenticated TLS reverse-proxy to Hugging Face Spaces",
        "Fault Isolation: Upstream timeout suppression & graceful client error mapping"
    ],
    border="#475569", title_col="#0F172A", sub_col="#475569"
)

# Card 2.2: Structured JSON Response Contract (Middle-Right)
draw_card(
    ax, x=8.2, y=4.40, w=6.4, h=1.80,
    title="Structured JSON Response Contract",
    subtitle="RFC-8259 Serialized Forensic Calibration Protocol",
    items=[
        "Calibrated Probabilities: y_v, y_a scaled with Platt temperatures (Tv = 1.12, Ta = 1.08)",
        "Quadrant Attribution: q in {RVRA, RVFA, FVRA, FVFA} (reported if both streams present)",
        "Temporal Localization: Manipulation masks mv(t), ma(t) in [0, 1] across frames",
        "Phoneme-Viseme Sync Curve: Windowed InfoNCE agreement curve st in [-1, +1]",
        "Stream Suppression: Suppresses verdict & timeline for missing/silent modality"
    ],
    border="#D97706", title_col="#92400E", sub_col="#B45309"
)

# =========================================================================
# TIER 3: CONTAINERIZED INFERENCE MICROSERVICE (FastAPI on HF Spaces)
# Y: 1.15 to 3.50 (h = 2.35)
# =========================================================================
draw_tier_panel(
    ax, x=0.5, y=1.15, w=14.4, h=2.35,
    title="TIER 3: CONTAINERIZED INFERENCE MICROSERVICE (FASTAPI ON HUGGING FACE SPACES / DOCKER)",
    bg_color="#F0FDF4", border_color="#A7F3D0", badge_bg="#0D9488"
)

# Card 3.1: Stream Demuxing & Preprocessing (Bottom-Left)
draw_card(
    ax, x=0.8, y=1.30, w=4.0, h=1.95,
    title="Stream Preprocessing",
    subtitle="FFmpeg Demuxing & Landmark Tracking",
    items=[
        "A/V Demuxing: Synchronized isolation",
        "Video: RetinaFace tracking @ 25 fps",
        "  Canonical 224x224 face & 96x96 mouth",
        "Audio: 16 kHz mono resampling",
        "  Peak loudness normalization & STFT",
        "In-Memory Pipe: Volatile RAM buffers"
    ],
    border="#0D9488", title_col="#115E59", sub_col="#0D9488"
)

# Card 3.2: Missing-Modality Router (Bottom-Center)
draw_card(
    ax, x=5.6, y=1.30, w=4.0, h=1.95,
    title="Missing-Modality Router",
    subtitle="Null-Token Adaptation Pipeline",
    items=[
        "Stream Presence Audit: Video/Audio check",
        "Missing Video Stream (Audio-Only):",
        "  Injects learnable null tokens Ø_v in R^(Lv x d)",
        "Missing Audio Stream (Silent Video):",
        "  Injects learnable null tokens Ø_a in R^(La x d)",
        "Disentanglement: Forces synchrony z_c <- 0"
    ],
    border="#0D9488", title_col="#115E59", sub_col="#0D9488"
)

# Card 3.3: DAVID-Net-Lite Distilled Engine (Bottom-Right)
draw_card(
    ax, x=10.4, y=1.30, w=4.2, h=1.95,
    title="DAVID-Net-Lite Engine",
    subtitle="Distilled Multi-Task Student Transformer",
    items=[
        "Student Backbones: ViT-B (Video) + WavLM",
        "Disentangled Transformer: 4 layers, d = 256",
        "Cross-Attention: Latents z_v, z_a & sync state z_c",
        "Decision Heads (MLP + GELU):",
        "  Hv -> Pv, Ha -> Pa, Hquad -> 4 Quadrants",
        "  Hloc -> mv(t), ma(t), Hsync -> Agreement st"
    ],
    border="#0D9488", title_col="#115E59", sub_col="#0D9488"
)

# =========================================================================
# FLOW ARROWS & DATA TRANSMISSION BADGES
# Symmetrical rails: Left vertical at x = 2.8, Right vertical at x = 12.5
# Clear Gutter 1: Y in [6.45, 7.20] (height = 0.75, mid = 6.775)
# Clear Gutter 2: Y in [3.50, 4.25] (height = 0.75, mid = 3.825)
# =========================================================================

# Arrow 1: Client Ingestion (Top-Left) -> Edge Gateway (Middle-Left) [Straight Down at x = 2.8]
draw_arrow(
    ax,
    start=(2.8, 7.35),
    end=(2.8, 6.20),
    label="1. HTTPS Multipart POST Upload",
    color="#2563EB", bg_col="#EFF6FF", border_col="#BFDBFE", text_col="#1E40AF"
)

# Arrow 2: Edge Gateway (Middle-Left) -> Stream Preprocessing (Bottom-Left) [Straight Down at x = 2.8]
draw_arrow(
    ax,
    start=(2.8, 4.40),
    end=(2.8, 3.25),
    label="2. Authenticated TLS Stream (/predict)",
    color="#0284C7", bg_col="#F0F9FF", border_col="#BAE6FD", text_col="#0369A1"
)

# Arrow 3: Stream Preprocessing -> Missing-Modality Router [Straight Right in Gap 1]
draw_arrow(
    ax,
    start=(4.8, 2.27),
    end=(5.6, 2.27),
    label="3. Demuxed\nStreams",
    color="#0D9488", bg_col="#F0FDFA", border_col="#99F6E4", text_col="#0F766E"
)

# Arrow 4: Missing-Modality Router -> DAVID-Net-Lite Engine [Straight Right in Gap 2]
draw_arrow(
    ax,
    start=(9.6, 2.27),
    end=(10.4, 2.27),
    label="4. Aligned\nTokens",
    color="#0D9488", bg_col="#F0FDFA", border_col="#99F6E4", text_col="#0F766E"
)

# Arrow 5: DAVID-Net-Lite Engine (Bottom-Right) -> JSON Contract (Middle-Right) [Straight Up at x = 12.5]
draw_arrow(
    ax,
    start=(12.5, 3.25),
    end=(12.5, 4.40),
    label="5. Raw Logits, Timelines & Latents",
    color="#D97706", bg_col="#FEF3C7", border_col="#FDE68A", text_col="#92400E"
)

# Arrow 6: JSON Contract (Middle-Right) -> Verdict Dashboard (Top-Right) [Straight Up at x = 12.5]
draw_arrow(
    ax,
    start=(12.5, 6.20),
    end=(12.5, 7.35),
    label="6. Calibrated JSON Forensic Verdict",
    color="#059669", bg_col="#ECFDF5", border_col="#A7F3D0", text_col="#065F46"
)

# =========================================================================
# FOOTER: PRIVACY & ZERO-RETENTION GUARANTEE
# =========================================================================
p_foot = patches.FancyBboxPatch(
    (0.5, 0.22), 14.4, 0.58,
    boxstyle="round,pad=0.10",
    facecolor="#F1F5F9",
    edgecolor="#CBD5E1",
    linewidth=0.9,
    zorder=2
)
ax.add_patch(p_foot)
ax.text(
    7.7, 0.51,
    "Stateless Execution & Zero-Retention Privacy Guarantee: All media decoded strictly in volatile memory. No user media or biometric embeddings are written to disk or retained after socket close.",
    fontsize=7.8, fontweight="bold", color="#334155", ha="center", va="center", zorder=4
)

plt.tight_layout()
out_dir = Path("report/figures/generated")
out_dir.mkdir(parents=True, exist_ok=True)
pdf_path = out_dir / "fig_deployment.pdf"
png_path = out_dir / "fig_deployment.png"
plt.savefig(pdf_path, bbox_inches="tight", dpi=300)
plt.savefig(png_path, bbox_inches="tight", dpi=300)
print(f"Generated {pdf_path} and {png_path} successfully!")
