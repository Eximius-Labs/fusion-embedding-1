"""Ember (thermal sense pack) architecture overview (HF pack card).

Extends the model-overview visual language (make_model_overview_figures.py): same
palette, same layer/pooling/Matryoshka grammar, with the generational delta of THIS
artifact front and center: two gated adapter stacks side by side on the frozen
decoder (audio pack, thermal pack), native paths bypassing both bit-for-bit.

Self-checking (make_family_figures.py pattern): label containment and top-level box
overlap are verified after draw; the script refuses to save on violation.

Run: uv run --with matplotlib python assets/make_thermal_pack_figure.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

INK = "#1b2f4b"
GRAY = "#7d8a9c"
LGRAY = "#c6cdd8"
ACC = "#C2622E"          # terracotta — trained
ACC_FILL = "#f9e9dc"
THERM = "#a34a1f"        # deeper terracotta for the thermal (new) pack
THERM_FILL = "#f6ddc9"
CHIP = "#ffffff"
PANEL = "#fbf7ef"

HERE = os.path.dirname(os.path.abspath(__file__))


def draw(out_path: str) -> None:
    width = 112
    fig, ax = plt.subplots(figsize=(11.2, 6.9), dpi=200)
    ax.set_xlim(0, width); ax.set_ylim(0, 62); ax.axis("off")
    fig.patch.set_facecolor("white")

    constraints: list[tuple] = []   # (label, artist, x, y, w, h)
    top_boxes: list[tuple] = []     # (name, x, y, w, h)

    def rbox(x, y, w, h, fc=CHIP, ec=INK, lw=1.4, r=1.0, z=2, top=None):
        b = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.3,rounding_size={r}",
                           fc=fc, ec=ec, lw=lw, zorder=z)
        ax.add_patch(b)
        if top:
            top_boxes.append((top, x, y, w, h))
        return b

    def txt(x, y, s, size=9, color=INK, weight="normal", ha="center", va="center",
            z=5, rot=0, contain=None):
        t = ax.text(x, y, s, fontsize=size, color=color, weight=weight, ha=ha, va=va,
                    zorder=z, rotation=rot)
        if contain:
            constraints.append((s[:28], t, *contain))
        return t

    def arr(x1, y1, x2, y2, color=GRAY, lw=1.2, z=1):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=10, color=color, lw=lw, zorder=z))

    bar_w = 76
    mid = 4 + bar_w / 2

    # ---------- input row ----------
    rbox(4, 55.5, bar_w, 5, top="input")
    rbox(5.5, 56.4, 12, 3.2, fc=THERM, ec=THERM, r=0.8)
    txt(11.5, 58, "Thermal", 9, color="white", weight="bold",
        contain=(5.5, 56.4, 12, 3.2))
    txt(20.5, 58, "◉", 10, color=GRAY)
    txt(mid + 8, 58, "a thermal frame: two pedestrians on an unlit road, engines still warm",
        8.6, contain=(21, 55.5, bar_w - 18, 5))

    # ---------- preprocessing bar ----------
    rbox(4, 48.5, bar_w, 4.2, top="preproc")
    txt(mid, 50.6, "single-channel thermal  →  replicate ×3  →  frozen Qwen3-VL vision tower  →  visual tokens",
        8.6, contain=(4, 48.5, bar_w, 4.2))
    arr(mid, 55.3, mid, 53.2)

    # ---------- main block ----------
    main_w = 62
    rbox(4, 11.5, main_w, 33.5, fc=PANEL, lw=1.7, r=1.6, top="main")
    txt(7, 42.4, "fusion-embedding-2  +  Ember (thermal)", 11.5, weight="bold", ha="left",
        contain=(4, 40.6, main_w, 4))
    txt(4 + main_w - 2, 13.1, "Qwen3-VL-Embedding-2B (byte-frozen)", 7.5, color=GRAY,
        ha="right", contain=(4, 11.5, main_w, 3.4))
    arr(mid, 48.2, mid, 45.6)

    inner_w = 40

    def layer(y, name, h=8, dots=False):
        rbox(8, y, inner_w, h, fc=CHIP, lw=1.3, r=1.2)
        if not dots:
            txt(11, y + 6.4, name, 8.2, ha="left", contain=(8, y, inner_w, h))
            rbox(10.5, y + 1.2, 14.5, 4, fc=CHIP, lw=1.0, r=0.6)
            txt(17.75, y + 3.2, "Self-Attn", 7.6, contain=(10.5, y + 1.2, 14.5, 4))
            f = rbox(27, y + 1.2, inner_w - 21, 4, fc=CHIP, lw=1.0, r=0.6)
            f.set_hatch("xx"); f.set_edgecolor("#9fb0c6"); f.set_facecolor("#f4f7fb")
            fx = 27 + (inner_w - 21) / 2
            ax.add_patch(Rectangle((fx - 2.4, y + 2.2), 4.8, 2.0, fc="white",
                                   ec="none", zorder=5))
            txt(fx, y + 3.2, "FFN", 7.8, z=6)
        else:
            txt(8 + inner_w / 2, y + h / 2, ". . .", 11)
        # two gated adapter stacks, side by side, on every layer
        ax_x = 8 + inner_w + 0.8
        rbox(ax_x, y + 0.6, 4.6, h - 1.2, fc=ACC_FILL, ec=ACC, lw=1.4, r=0.6)
        if h >= 6:
            txt(ax_x + 2.3, y + h / 2, "Audio", 6.6, color=ACC, rot=90, weight="bold")
        rbox(ax_x + 5.6, y + 0.6, 4.6, h - 1.2, fc=THERM_FILL, ec=THERM, lw=1.4, r=0.6)
        if h >= 6:
            txt(ax_x + 7.9, y + h / 2, "Thermal", 6.6, color=THERM, rot=90, weight="bold")

    layer(33, "Layer 1")
    layer(27.2, "", h=4.2, dots=True)
    layer(16.5, "Layer 28")
    flow_x = 8 + inner_w / 2 - 3
    arr(flow_x, 32.6, flow_x, 31.8)
    arr(flow_x, 26.8, flow_x, 25.0)
    txt(8 + inner_w + 6.2, 14.2, "+44.2M trained", 6.8, color=THERM, weight="bold",
        contain=(4, 11.5, main_w, 4.2))

    # ---------- modality-gate legend: BELOW the flow axis (the pipeline corridor
    # decoder -> pooling stays empty; nothing crosses the legend's bounding box) ----------
    leg_x = 67.5
    txt(leg_x, 38.4, "Modality gates", 8, color=GRAY, ha="left", weight="bold",
        contain=(leg_x - 1, 37.0, 24, 3))
    chips = [("Thermal — Ember gate ON", THERM, "fill", 32.4),
             ("Audio — audio gate ON", ACC, "fill", 27.6),
             ("Text — bypass · bit-identical", None, "off", 22.8),
             ("Image — bypass · bit-identical", None, "off", 18.0),
             ("Video — bypass · bit-identical", None, "off", 13.2)]
    for label_s, col, kind, cy in chips:
        active = kind == "fill"
        rbox(leg_x, cy, 21.5, 4.0, fc=col if active else CHIP,
             ec=col if active else LGRAY, lw=1.4, r=1.4, top=f"chip:{label_s[:12]}")
        txt(leg_x + 10.75, cy + 2.0, label_s, 7.4, color="white" if active else GRAY,
            weight="bold" if active else "normal", contain=(leg_x, cy, 21.5, 4.0))

    right_x = 94

    # ---------- right column ----------
    rbox(right_x, 39, 16, 8, top="pooling")
    txt(right_x + 8, 45.2, "Last-Token Pooling", 8.4, weight="bold",
        contain=(right_x, 39, 16, 8))
    xx = right_x + 1.2
    for _ in range(4):
        ax.add_patch(Rectangle((xx, 40.6), 1.8, 1.8, fc="#cfd9e6", ec="none", zorder=4))
        xx += 2.2
    txt(xx + 0.5, 41.5, "...", 8, color=GRAY); xx += 1.9
    ax.add_patch(Rectangle((xx, 40.6), 3.0, 1.8, fc=ACC, ec="none", zorder=4))
    txt(xx + 1.5, 41.5, "EOS", 6.4, color="white", weight="bold")
    arr(4 + main_w + 0.4, 43, right_x - 0.4, 43, lw=1.2)  # decoder -> pooling

    rbox(right_x, 15.5, 16, 21.5, top="matryoshka")
    txt(right_x + 8, 34.8, "Matryoshka", 8.6, weight="bold",
        contain=(right_x, 15.5, 16, 21.5))
    txt(right_x + 8, 32.8, "Representation Learning", 7.2,
        contain=(right_x, 15.5, 16, 21.5))
    dims = ["2048 dims", "1024 dims", "512 dims", "256 dims", "64 dims"]
    shades = ["#f4ecdb", "#ecdfc2", "#e3d0a8", "#d9c08d", "#cfb074"]
    bx, by, bw, bh = right_x + 1.2, 16.7, 13.6, 14.2
    for i, (d, sh) in enumerate(zip(dims, shades)):
        rbox(bx + i * 1.15, by, bw - i * 2.3, bh - i * 2.55, fc=sh, ec=INK, lw=0.9, r=0.7)
        txt(right_x + 8, by + bh - i * 2.55 - 1.15, d, 6.8)
    arr(right_x + 8, 38.6, right_x + 8, 37.6)

    rbox(right_x, 9, 16, 4.6, top="embedding")
    txt(right_x + 8, 11.3, "Embedding Vector", 8.4, weight="bold",
        contain=(right_x, 9, 16, 4.6))
    arr(right_x + 8, 15.1, right_x + 8, 14.1)

    # ---------- caption (bottom-left, clear of the legend block) ----------
    cap_lines = ["Packs are separable artifacts, independently loadable. Gates are mutually",
                 "exclusive by input modality: a thermal encode opens only the thermal gate;",
                 "text, image, video, and audio outputs are bit-for-bit the base model's."]
    cap_boxes = []
    for i, line in enumerate(cap_lines):
        cb = (2, 9.2 - i * 2.4 - 1.4, 76, 2.8)
        txt(4, 9.2 - i * 2.4, line, 7.6, color=GRAY, ha="left", contain=cb)
        cap_boxes.append(cb)

    # ---------- self-checks ----------
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    bad = []
    for label, artist, x, y, w, h in constraints:
        tb = artist.get_window_extent(renderer=ren)
        (tx0, ty0), (tx1, ty1) = inv.transform([(tb.x0, tb.y0), (tb.x1, tb.y1)])
        pad = 0.35
        if (tx0 < x - pad or tx1 > x + w + pad or ty0 < y - pad or ty1 > y + h + pad):
            bad.append(f"  text {label!r} leaves its box "
                       f"({tx0:.1f},{ty0:.1f})-({tx1:.1f},{ty1:.1f}) vs "
                       f"({x},{y})+({w},{h})")
    for i, (n1, x1, y1, w1, h1) in enumerate(top_boxes):
        for n2, x2, y2, w2, h2 in top_boxes[i + 1:]:
            gap = 0.7
            if (x1 < x2 + w2 + gap and x2 < x1 + w1 + gap and
                    y1 < y2 + h2 + gap and y2 < y1 + h1 + gap):
                bad.append(f"  boxes {n1!r} and {n2!r} overlap")
    ay, ax0, ax1 = 43.0, 4 + main_w + 0.4, right_x - 0.4   # flow-arrow segment
    leg_boxes = [(n, x, y, w, h) for n, x, y, w, h in top_boxes if n.startswith("chip:")]
    leg_top = max(y + h for _, _, y, _, h in leg_boxes) + 4.6      # + title band
    leg_bot = min(y for _, _, y, _, h in leg_boxes)
    if leg_top > ay - 2.0:
        bad.append(f"  legend top {leg_top:.1f} grazes the flow arrow at y={ay}")
    if leg_bot < 11.5:
        bad.append(f"  legend bottom {leg_bot:.1f} extends below the decoder box (11.5)")
    for bn, bx0, by0, bw, bh in top_boxes:
        if bn.startswith("chip:") and bx0 < ax1 and ax0 < bx0 + bw and by0 < ay < by0 + bh:
            bad.append(f"  flow arrow crosses legend box {bn!r}")
    cap_top = max(y + h for _, y, _, h in [(b[0], b[1], b[2], b[3]) for b in cap_boxes])
    if cap_top > 11.5:
        bad.append(f"  caption block top {cap_top:.1f} overlaps the decoder box")
    for _, x, y, w, h in [(0, b[0], b[1], b[2], b[3]) for b in cap_boxes]:
        for n2, x2, y2, w2, h2 in top_boxes:
            if n2.startswith("chip:") and x < x2 + w2 and x2 < x + w and y < y2 + h2 and y2 < y + h:
                bad.append(f"  caption line overlaps legend box {n2!r}")
    if bad:
        raise SystemExit("figure self-check failed:\n" + "\n".join(bad))
    print(f"self-check: {len(constraints)} labels, {len(top_boxes)} boxes OK")

    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out_path)


if __name__ == "__main__":
    draw(os.path.join(HERE, "fe2_ember_overview.png"))
