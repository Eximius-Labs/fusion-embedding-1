"""Model-overview figures for the fusion-embedding family (HF model cards).

One generator, two outputs (assets/fe1_model_overview.png, assets/fe2_model_overview.png)
so the figures stay visually parallel: identical layout and palette, with FE2 adding the
adapter tabs and modality-gate chips that ARE the generational difference.

Palette = Eximius Labs brand (org banner): navy ink, cream sand panel, terracotta accent,
steel-blue frozen styling, sand-gold Matryoshka ramp.

Run: uv run --with matplotlib python assets/make_model_overview_figures.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

INK = "#1b2f4b"      # deep navy — structural ink
GRAY = "#7d8a9c"     # blue-gray — secondary text/arrows
LGRAY = "#c6cdd8"    # muted blue-gray — inactive edges
ACC = "#C2622E"      # terracotta — the active/trained accent
ACC_FILL = "#f9e9dc"
CHIP = "#ffffff"
PANEL = "#fbf7ef"    # cream sand

HERE = os.path.dirname(os.path.abspath(__file__))


def draw(fe2: bool, out_path: str) -> None:
    width = 110 if fe2 else 94
    fig, ax = plt.subplots(figsize=(11.0 if fe2 else 9.4, 6.2), dpi=200)
    ax.set_xlim(0, width); ax.set_ylim(7, 62); ax.axis("off")
    fig.patch.set_facecolor("white")

    def rbox(x, y, w, h, fc=CHIP, ec=INK, lw=1.4, r=1.0, z=2):
        b = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.3,rounding_size={r}",
                           fc=fc, ec=ec, lw=lw, zorder=z)
        ax.add_patch(b); return b

    def txt(x, y, s, size=9, color=INK, weight="normal", ha="center", va="center", z=5, rot=0):
        ax.text(x, y, s, fontsize=size, color=color, weight=weight, ha=ha, va=va,
                zorder=z, rotation=rot)

    def arr(x1, y1, x2, y2, color=GRAY, lw=1.2, z=1):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=10, color=color, lw=lw, zorder=z))

    bar_w = 76 if fe2 else 60
    mid = 4 + bar_w / 2

    # ---------- input row ----------
    rbox(4, 55.5, bar_w, 5)
    rbox(5.5, 56.4, 10, 3.2, fc=ACC, ec=ACC, r=0.8)
    txt(10.5, 58, "Audio", 9, color="white", weight="bold")
    txt(18.5, 58, "◉", 10, color=GRAY)
    txt(mid + 6, 58, "a 10-second clip: a dog barks while rain falls on a metal roof",
        9.0 if fe2 else 8.2)

    # ---------- preprocessing bar ----------
    rbox(4, 48.5, bar_w, 4.2)
    txt(mid, 50.6, "frozen Qwen2.5-Omni audio tower  →  FusionResampler (trained, 16.4M)  →  64 audio tokens",
        9 if fe2 else 7.9)
    arr(mid, 55.3, mid, 53.2)

    # ---------- main block ----------
    main_w = 62 if fe2 else 62
    rbox(4, 13, main_w, 32, fc=PANEL, lw=1.7, r=1.6)
    txt(7, 42.2, "fusion-embedding-2" if fe2 else "fusion-embedding-1", 12, weight="bold", ha="left")
    txt(4 + main_w - 2, 14.7, "Qwen3-VL-Embedding-2B (byte-frozen)", 7.5, color=GRAY, ha="right")
    arr(mid, 48.2, mid, 45.6)

    inner_w = 46 if fe2 else 52

    def layer(y, name, h=8, dots=False):
        rbox(8, y, inner_w, h, fc=CHIP, lw=1.3, r=1.2)
        if not dots:
            txt(11, y + 6.4, name, 8.2, ha="left")
            rbox(10.5, y + 1.2, 17, 4, fc=CHIP, lw=1.0, r=0.6)
            txt(19, y + 3.2, "Self-Attention", 7.8)
            f = rbox(30.5, y + 1.2, inner_w - 13 - 16.5, 4, fc=CHIP, lw=1.0, r=0.6)
            f.set_hatch("xx"); f.set_edgecolor("#9fb0c6"); f.set_facecolor("#f4f7fb")
            fx = 30.5 + (inner_w - 13 - 16.5) / 2
            ax.add_patch(Rectangle((fx - 2.4, y + 2.2), 4.8, 2.0, fc="white", ec="none", zorder=5))
            txt(fx, y + 3.2, "FFN", 7.8, z=6)
        else:
            txt(8 + inner_w / 2, y + h / 2, ". . .", 11)
        if fe2:  # adapter tab on every layer — the generational difference
            rbox(8 + inner_w + 0.8, y + 0.6, 4.6, h - 1.2, fc=ACC_FILL, ec=ACC, lw=1.4, r=0.6)
            if h >= 6:
                txt(8 + inner_w + 3.1, y + h / 2, "Adapter", 6.8, color=ACC, rot=90, weight="bold")

    layer(33, "Layer 1")
    layer(27.2, "", h=4.2, dots=True)
    layer(16.5, "Layer 28")
    flow_x = 8 + inner_w / 2 - 3
    arr(flow_x, 32.6, flow_x, 31.8)
    arr(flow_x, 26.8, flow_x, 25.0)

    right_x = 92 if fe2 else 76

    if fe2:
        # ---------- modality-gate chips ----------
        chips = [("Audio — adapters ON", True, 36.5), ("Text — bypass", False, 30.5),
                 ("Image — bypass", False, 24.5), ("Video — bypass", False, 18.5)]
        for label_s, active, cy in chips:
            rbox(69, cy, 20, 4.4, fc=ACC if active else CHIP,
                 ec=ACC if active else LGRAY, lw=1.4, r=1.4)
            txt(79, cy + 2.2, label_s, 8, color="white" if active else GRAY,
                weight="bold" if active else "normal")
        for y in (37, 29.3, 20.5):
            ax.plot([59.8, 69], [y, 38.7], color=ACC, lw=0.9, alpha=0.55, zorder=1)

    # ---------- right column ----------
    rbox(right_x, 39, 16, 8)
    txt(right_x + 8, 45.2, "Last-Token Pooling", 8.6, weight="bold")
    xx = right_x + 1.2
    for _ in range(4):
        ax.add_patch(Rectangle((xx, 40.6), 1.8, 1.8, fc="#cfd9e6", ec="none", zorder=4)); xx += 2.2
    txt(xx + 0.5, 41.5, "...", 8, color=GRAY); xx += 1.9
    ax.add_patch(Rectangle((xx, 40.6), 3.0, 1.8, fc=ACC, ec="none", zorder=4))
    txt(xx + 1.5, 41.5, "EOS", 6.4, color="white", weight="bold")
    arr(4 + main_w + 0.4, 43, right_x - 0.4, 43, lw=1.2)          # horizontal into pooling

    rbox(right_x, 15.5, 16, 21.5)
    txt(right_x + 8, 34.8, "Matryoshka", 8.6, weight="bold")
    txt(right_x + 8, 32.8, "Representation Learning", 7.4)
    dims = ["2048 dims", "1024 dims", "512 dims", "256 dims", "64 dims"]
    shades = ["#f4ecdb", "#ecdfc2", "#e3d0a8", "#d9c08d", "#cfb074"]
    bx, by, bw, bh = right_x + 1.2, 16.7, 13.6, 14.2
    for i, (d, sh) in enumerate(zip(dims, shades)):
        rbox(bx + i * 1.15, by, bw - i * 2.3, bh - i * 2.55, fc=sh, ec=INK, lw=0.9, r=0.7)
        txt(right_x + 8, by + bh - i * 2.55 - 1.15, d, 6.8)
    arr(right_x + 8, 38.6, right_x + 8, 37.6)

    rbox(right_x, 9, 16, 4.6)
    txt(right_x + 8, 11.3, "Embedding Vector", 8.6, weight="bold")
    arr(right_x + 8, 15.1, right_x + 8, 14.1)

    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out_path)


if __name__ == "__main__":
    draw(fe2=True, out_path=os.path.join(HERE, "fe2_model_overview.png"))
    draw(fe2=False, out_path=os.path.join(HERE, "fe1_model_overview.png"))
