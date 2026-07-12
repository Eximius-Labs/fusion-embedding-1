"""Family figures for the public repo and model cards (brand palette).

Generates, into assets/:
    fe_family_architecture.png  -- the two-generation architecture diagram
                                   (same figure as the technical report's Figure 1)
    fe_positioning.png          -- unified-model positioning on VGGSound-696,
                                   trained-parameters axis (same as the report)

Run: uv run --with matplotlib python assets/make_family_figures.py

These are the canonical repo/card copies of the paper figures; edit here and in
the paper's figure source together to keep GitHub, the HF cards, and the paper
consistent. The architecture figure self-checks label containment and box
overlap before saving. Positioning provenance: published card numbers
(VGGSound-696, average R@10 over both directions); trained-parameter counts:
ImageBind ~0.21B of 1.2B total (CVPR'23 paper, sec. 4); LanguageBind ~0.30B
audio-tower full fine-tune (the benchmarked released checkpoint).
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ brand ----
INK = "#1b2f4b"       # deep navy -- structural ink
GRAY = "#7d8a9c"      # blue-gray -- secondary text / arrows
LGRAY = "#c6cdd8"     # muted blue-gray -- inactive edges
ACC = "#C2622E"       # terracotta -- the ONE trained/active accent
ACC_FILL = "#f9e9dc"  # terracotta tint fill
BLUE = "#3A6EA5"      # steel blue -- fusion-embedding-1 series
PANEL = "#fbf7ef"     # cream sand panel
FR_FILL = "#f4f7fb"   # frozen-component fill
FR_EDGE = "#9fb0c6"   # frozen-component hatch/edge
TOK = "#cfd9e6"       # frozen token squares
GOLD = ["#f4ecdb", "#ecdfc2", "#e3d0a8", "#d9c08d", "#cfb074"]  # MRL ramp
LGRID = "#e3e7ee"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "text.color": INK,
})


def save(fig, name: str) -> None:
    fig.savefig(os.path.join(HERE, f"{name}.png"), dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"wrote {name}.png")


# ================================================== Figure 1: architecture ====
def fig_family_architecture() -> None:
    """Two-generation family diagram, one left-to-right flow, print-first.

    Axis units are 20/inch (130 x 80 units on a 6.5 x 4.0 in canvas), so a
    fontsize of N renders at N pt in the final PDF. Design rules: no layer
    internals (the text explains self-attention/FFN; the figure's job is the
    two-generation story), fonts 6.2-9 pt, hatching only on the audio tower,
    one terracotta accent for everything trained, generous margins.

    Self-checking: every label registered against a box must render inside
    it, and no two top-level boxes may overlap -- the script refuses to save
    otherwise.
    """
    fig = plt.figure(figsize=(6.5, 4.0))
    ax = fig.add_axes((0, 0, 1, 1))  # full-bleed: exactly 20 units per inch,
    ax.set_xlim(0, 130)              # so fontsize N really renders at N pt
    ax.set_ylim(0, 80)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    constraints: list[tuple] = []   # (label, text_artist, x, y, w, h)
    top_boxes: list[tuple] = []     # (name, x, y, w, h) -- must not overlap

    def rbox(x, y, w, h, fc="white", ec=INK, lw=1.1, r=1.0, z=2, top=None):
        b = FancyBboxPatch((x, y), w, h,
                           boxstyle=f"round,pad=0.3,rounding_size={r}",
                           fc=fc, ec=ec, lw=lw, zorder=z)
        ax.add_patch(b)
        if top:
            top_boxes.append((top, x, y, w, h))
        return b

    def txt(x, y, s, size=7, color=INK, weight="normal", ha="center",
            va="center", z=6, style="normal", inside=None):
        t = ax.text(x, y, s, fontsize=size, color=color, weight=weight,
                    ha=ha, va=va, zorder=z, style=style)
        if inside is not None:
            constraints.append((s, t, *inside))
        return t

    def arr(x1, y1, x2, y2, color=GRAY, lw=1.1, z=1):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=9, color=color, lw=lw,
                                     zorder=z, shrinkA=0, shrinkB=0))

    def text_backing(x, y, w, h, z=4):
        ax.add_patch(Rectangle((x, y), w, h, fc="white", ec="none", zorder=z))

    # ---------------- audio pipeline (top band, centre y = 71.5) ------------
    ab = (2, 68.5, 13, 6)
    rbox(*ab, fc=ACC, ec=ACC, top="audio-chip")
    txt(8.5, 71.5, "Audio", 8, color="white", weight="bold", inside=ab)
    arr(15.9, 71.5, 18.4, 71.5)

    tb = (19, 66.5, 25, 10)
    b = rbox(*tb, fc=FR_FILL, ec=FR_EDGE, top="tower")
    b.set_hatch("xx")
    text_backing(20.8, 67.6, 21.4, 7.8)
    txt(31.5, 74.0, "Qwen2.5-Omni", 8, weight="bold", inside=tb)
    txt(31.5, 71.4, "audio tower", 7, inside=tb)
    txt(31.5, 68.9, "frozen", 6.5, color=GRAY, style="italic", inside=tb)
    arr(44.9, 71.5, 47.0, 71.5)

    rb = (47.6, 66.5, 25, 10)
    rbox(*rb, fc=ACC_FILL, ec=ACC, lw=1.6, top="resampler")
    txt(60.1, 74.0, "FusionResampler", 7, color=ACC, weight="bold", inside=rb)
    txt(60.1, 71.4, "trained · 16.4M", 7, color=ACC, inside=rb)
    txt(60.1, 68.9, "generation 1", 6.5, color=ACC, style="italic", inside=rb)
    arr(73.1, 71.5, 75.4, 71.5)

    tx = 76.2
    for _ in range(5):
        ax.add_patch(Rectangle((tx, 70.5), 2.0, 2.0, fc=ACC_FILL, ec=ACC,
                               lw=0.9, zorder=4))
        tx += 2.5
    txt(tx + 1.0, 71.5, "...", 7.5, color=GRAY)
    txt(76.2, 75.6, "64 audio tokens", 6.5, color=GRAY, ha="left")
    arr(80.2, 69.9, 80.2, 60.9, color=GRAY, lw=1.1)

    # ---------------- text / image / video inputs (left) --------------------
    for name, cy in (("Text", 41), ("Image", 33), ("Video", 25)):
        cb = (2, cy, 13, 6)
        rbox(*cb, fc=FR_FILL, ec=FR_EDGE, top=f"chip-{name}")
        txt(8.5, cy + 3, name, 8, color=BLUE, weight="bold", inside=cb)
        arr(15.9, cy + 3, 41.2, cy + 3, color=FR_EDGE)
    txt(2, 20.3, "native paths", 6.5, color=GRAY, ha="left")
    txt(2, 18.1, "unmodified", 6.5, color=GRAY, ha="left", style="italic")

    # ---------------- frozen decoder panel (centre) -------------------------
    panel = (42, 12, 52, 48)
    rbox(*panel, fc=PANEL, ec=INK, lw=1.5, r=1.6, z=1, top="panel")
    txt(45, 55.6, "Qwen3-VL-Embedding-2B", 9, weight="bold", ha="left",
        inside=panel)
    txt(45, 52.7, "decoder · byte-frozen", 7, color=GRAY, ha="left",
        inside=panel)

    stack_x, stack_w = 46.5, 32
    tab_x = stack_x + stack_w + 1.6
    tab_w = 4.4

    def layer(y, name, h=8.0, dots=False):
        lb = (stack_x, y, stack_w, h)
        rbox(*lb, fc="white", ec=INK, lw=1.1, z=3)
        if dots:
            txt(stack_x + stack_w / 2, y + h / 2, ". . .", 10, z=6, inside=lb)
        else:
            txt(stack_x + stack_w / 2, y + h / 2, name, 7.5, z=6, inside=lb)
        # adapter tab -- the generational difference (terracotta outline)
        rbox(tab_x, y + 0.6, tab_w, h - 1.2, fc="white", ec=ACC, lw=1.4,
             r=0.7, z=3)

    layer(40, "Layer 1")
    layer(35.2, "", h=3.6, dots=True)
    layer(25, "Layer 28")
    fx = stack_x + stack_w / 2
    arr(fx, 39.8, fx, 39.0, lw=1.0)
    arr(fx, 35.0, fx, 33.4, lw=1.0)

    # the gate story, told once, directly under the stack (no chip cluster)
    gc = (stack_x + (stack_w + tab_w + 1.6) / 2)
    txt(gc, 20.7, "gated adapters · trained · +44.2M", 7, color=ACC,
        weight="bold", inside=panel)
    txt(gc, 18.2, "generation 2 — gate fires only on audio", 6.5, color=ACC,
        style="italic", inside=panel)
    txt(gc, 15.8, "non-audio inputs bypass — bitwise-identical", 6.5,
        color=GRAY, style="italic", inside=panel)

    # ---------------- right column: pooling -> MRL -> vector ----------------
    rcx, rcw = 100, 26
    mid = rcx + rcw / 2
    arr(94.5, 51.0, rcx - 0.5, 51.0, lw=1.1)

    pb = (rcx, 45.5, rcw, 11)
    rbox(*pb, top="pooling")
    txt(mid, 53.3, "last-token pooling", 7.5, weight="bold", inside=pb)
    xx = rcx + 5.4
    for _ in range(3):
        ax.add_patch(Rectangle((xx, 47.4), 1.8, 1.8, fc=TOK, ec="none",
                               zorder=4))
        xx += 2.3
    txt(xx + 0.8, 48.3, "...", 7, color=GRAY)
    ax.add_patch(Rectangle((xx + 2.2, 47.15), 5.2, 2.3, fc=ACC, ec="none",
                           zorder=4))
    txt(xx + 4.8, 48.3, "EOS", 6.2, color="white", weight="bold", inside=pb)

    arr(mid, 44.9, mid, 43.1)
    mb = (rcx, 17.5, rcw, 24)
    rbox(*mb, top="matryoshka")
    txt(mid, 38.7, "Matryoshka", 7.5, weight="bold", inside=mb)
    bx, by, bw, bh = rcx + 1.8, 18.7, rcw - 3.6, 18.4
    dims = ["2048", "1024", "512", "256", "64"]
    for i, (d, sh) in enumerate(zip(dims, GOLD)):
        rbox(bx + i * 1.25, by, bw - i * 2.5, bh - i * 3.1, fc=sh, ec=INK,
             lw=0.8, r=0.7, z=3 + i)
        txt(mid, by + bh - i * 3.1 - 1.45, d, 6.5, z=9)

    arr(mid, 16.9, mid, 15.4)
    eb = (rcx, 10.2, rcw, 4.8)
    rbox(*eb, top="embedding")
    txt(mid, 12.6, "embedding", 7.5, weight="bold", inside=eb)

    # ---------------- generation key (bottom strip, two rows) ---------------
    for ky, fc_sw, s in (
        (5.6, ACC_FILL,
         "fusion-embedding-1  =  frozen base + trained FusionResampler "
         "(16.4M)"),
        (1.9, "white",
         "fusion-embedding-2  =  FE1 + modality-gated deep adapters "
         "(+44.2M, audio-only)"),
    ):
        ax.add_patch(Rectangle((4, ky - 1.1), 2.6, 2.6, fc=fc_sw, ec=ACC,
                               lw=1.4, zorder=4))
        txt(8, ky + 0.15, s, 7, ha="left")

    # -------- self-checks: label containment + top-level box overlap --------
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    bad = []
    for label, artist, x, y, w, h in constraints:
        tb_disp = artist.get_window_extent(renderer=ren)
        (tx0, ty0), (tx1, ty1) = inv.transform([(tb_disp.x0, tb_disp.y0),
                                                (tb_disp.x1, tb_disp.y1)])
        pad = 0.3
        if (tx0 < x - pad or tx1 > x + w + pad or
                ty0 < y - pad or ty1 > y + h + pad):
            bad.append(f"  text {label!r} leaves its box")
    for i, (n1, x1, y1, w1, h1) in enumerate(top_boxes):
        for n2, x2, y2, w2, h2 in top_boxes[i + 1:]:
            gap = 0.7  # rounded pads
            if (x1 < x2 + w2 + gap and x2 < x1 + w1 + gap and
                    y1 < y2 + h2 + gap and y2 < y1 + h1 + gap):
                bad.append(f"  boxes {n1!r} and {n2!r} overlap")
    if bad:
        raise SystemExit("figure self-check failed:\n" + "\n".join(bad))
    print(f"self-check: {len(constraints)} labels, {len(top_boxes)} boxes OK")

    save(fig, "fe_family_architecture")


# ==================================================== Figure 2: positioning ===
# x-axis = TRAINED parameters (the efficiency story), verified counts:
# ImageBind trains audio/depth/thermal/IMU encoders + heads ~0.21B of 1.2B total
# (CVPR'23 paper §4); LanguageBind's benchmarked Audio_FT checkpoint fully
# fine-tunes its ~0.30B audio tower against the frozen text tower (their §3.1 +
# HF ckpt sizes). See docs/paper/research_fix_numbers.md §5 for provenance.
# (label, trained_params_B, a2t_avg_R10, a2i_avg_R10, kind)  kind: fe2|fe1|base
POINTS = [
    ("fusion-embedding-2\n(60.6M trained · 2.8B total)", 0.0606, 0.673, 0.411, "fe2"),
    ("fusion-embedding-1 v0.3\n(16.4M trained · 2.8B total)", 0.0164, 0.635, 0.418, "fe1"),
    ("LanguageBind\n(~0.30B trained)", 0.304, 0.439, 0.390, "base"),
    ("ImageBind-Huge\n(~0.21B trained)", 0.215, 0.376, 0.719, "base"),
]
GEMINI = {"a2t": 0.377, "a2i": 0.314}

PANELS = [
    ("Cross-modal audio ↔ text", "a2t"),
    ("Emergent audio ↔ image", "a2i"),
]

# per-panel label offsets (dx = log-x multiplier, dy) to avoid collisions
OFFS = {
    ("a2t", "fusion-embedding-2\n(60.6M trained · 2.8B total)"): (1.30, 0.012),
    ("a2t", "fusion-embedding-1 v0.3\n(16.4M trained · 2.8B total)"): (0.80, -0.082),
    ("a2t", "LanguageBind\n(~0.30B trained)"): (0.35, 0.038),
    ("a2t", "ImageBind-Huge\n(~0.21B trained)"): (0.35, -0.055),
    ("a2i", "fusion-embedding-2\n(60.6M trained · 2.8B total)"): (1.30, 0.022),
    ("a2i", "fusion-embedding-1 v0.3\n(16.4M trained · 2.8B total)"): (0.60, 0.055),
    ("a2i", "LanguageBind\n(~0.30B trained)"): (0.55, -0.048),
    ("a2i", "ImageBind-Huge\n(~0.21B trained)"): (0.24, 0.030),
}
# per-panel Gemini reference-line caption placement (x, ha, dy)
GEMINI_CAP = {
    "a2t": (0.0095, "left", 0.011),
    "a2i": (1.02, "right", -0.030),
}


def fig_positioning() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))
    fig.patch.set_facecolor("white")

    for ax, (title, key) in zip(axes, PANELS):
        ax.set_facecolor("white")
        ax.set_title(title, fontsize=9, weight="bold", color=INK, pad=7)
        ax.set_xscale("log")
        ax.set_xlim(0.009, 1.1)
        ax.set_xticks([0.01, 0.03, 0.1, 0.3, 1.0])
        ax.set_xticklabels(["10M", "30M", "100M", "300M", "1B"], fontsize=7,
                           color=INK)
        ax.minorticks_off()
        ax.set_xlabel("Trained parameters", fontsize=7.5, color=INK)
        ax.grid(axis="y", color=LGRID, lw=0.6, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRAY)
            ax.spines[s].set_linewidth(0.7)
        ax.tick_params(colors=GRAY, labelsize=7, width=0.7)

        for label, pB, a2t, a2i, kind in POINTS:
            y = a2t if key == "a2t" else a2i
            if kind == "fe2":
                ax.scatter([pB], [y], s=62, c=ACC, ec="white", lw=1.0,
                           zorder=5)
            elif kind == "fe1":
                ax.scatter([pB], [y], s=54, c=BLUE, ec="white", lw=1.0,
                           zorder=5)
            else:
                ax.scatter([pB], [y], s=44, c=GRAY, ec="white", lw=0.9,
                           zorder=4)
            dx, dy = OFFS[(key, label)]
            name = label.split("\n")[0] + ("  (supervised pair)" if
                    (key == "a2i" and "ImageBind" in label) else "")
            sub = label.split("\n")[1]
            ours = kind in ("fe2", "fe1")
            ax.text(pB * dx, y + dy + 0.014, name, fontsize=6.4 if ours else 6.2,
                    color=INK, weight="bold" if ours else "normal", ha="left",
                    zorder=6)
            ax.text(pB * dx, y + dy - 0.012, sub, fontsize=5.8, color=GRAY,
                    ha="left", zorder=6)

        g = GEMINI[key]
        ax.axhline(g, color=GRAY, lw=0.9, ls=(0, (4, 3)), zorder=1)
        gx, gha, gdy = GEMINI_CAP[key]
        ax.text(gx, g + gdy, "Gemini Embedding 2 (API, size undisclosed)",
                fontsize=6, color=GRAY, ha=gha)

    axes[0].set_ylabel("Average R@10 (both directions)", fontsize=7.5,
                       color=INK)
    axes[0].set_ylim(0.28, 0.75)
    axes[1].set_ylim(0.25, 0.80)

    h_fe2 = plt.Line2D([], [], marker="o", ls="", ms=7, mfc=ACC, mec="white",
                       label="fusion-embedding-2")
    h_fe1 = plt.Line2D([], [], marker="o", ls="", ms=6.5, mfc=BLUE,
                       mec="white", label="fusion-embedding-1")
    h_b = plt.Line2D([], [], marker="o", ls="", ms=6, mfc=GRAY, mec="white",
                     label="baselines")
    fig.legend(handles=[h_fe2, h_fe1, h_b], loc="lower center", ncol=3,
               frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=(0, 0.06, 1, 1.0))
    save(fig, "fe_positioning")


if __name__ == "__main__":
    fig_family_architecture()
    fig_positioning()
    print("done")
