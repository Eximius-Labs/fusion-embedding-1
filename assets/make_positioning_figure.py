"""Unified-model positioning scatter (2 panels) for README/model cards.

Score vs parameters, jina-v5-style: FE2 terracotta, FE1 steel blue (validated pair),
baselines in muted gray, every point direct-labeled, API models (undisclosed size) as
dashed reference lines. All scores are the published card numbers: VGGSound-696,
average of both retrieval directions, R@10. Param counts measured from safetensors
metadata (ImageBind 1.201B, Qwen3-VL-Embedding-2B 2.128B) or the standard CLIP-L
pairing (LanguageBind 0.43B); ours = frozen base 2.128B + frozen tower ~0.64B + trained
components.

Run: uv run --with matplotlib python assets/make_positioning_figure.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#1b2f4b"; GRAY = "#7d8a9c"; LGRID = "#e3e7ee"; ACC = "#C2622E"; ACC2 = "#3A6EA5"

HERE = os.path.dirname(os.path.abspath(__file__))

# (label, params_B, a2t_avg_R10, a2i_avg_R10, kind)  kind: fe2|fe1|base
POINTS = [
    ("fusion-embedding-2\n(2.8B · 60.6M trained)", 2.83, 0.673, 0.411, "fe2"),
    ("fusion-embedding-1 v0.3\n(2.8B · 16.4M trained)", 2.78, 0.635, 0.418, "fe1"),
    ("LanguageBind\n(0.43B)", 0.43, 0.439, 0.390, "base"),
    ("ImageBind-Huge\n(1.2B)", 1.201, 0.376, 0.719, "base"),
]
GEMINI = {"a2t": 0.377, "a2i": 0.314}

PANELS = [
    ("Cross-modal audio ↔ text", "a2t", 2),
    ("Emergent audio ↔ image", "a2i", 3),
]

# per-panel label offsets (dx in log-x multiplier, dy) to avoid collisions
OFFS = {
    ("a2t", "fusion-embedding-2\n(2.8B · 60.6M trained)"): (0.44, 0.030),
    ("a2t", "fusion-embedding-1 v0.3\n(2.8B · 16.4M trained)"): (0.62, -0.070),
    ("a2t", "LanguageBind\n(0.43B)"): (1.18, 0.022),
    ("a2t", "ImageBind-Huge\n(1.2B)"): (1.12, 0.000),
    ("a2i", "fusion-embedding-2\n(2.8B · 60.6M trained)"): (0.52, -0.062),
    ("a2i", "fusion-embedding-1 v0.3\n(2.8B · 16.4M trained)"): (0.40, 0.042),
    ("a2i", "LanguageBind\n(0.43B)"): (0.72, 0.038),
    ("a2i", "ImageBind-Huge\n(1.2B)"): (0.62, 0.032),
}

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), dpi=200)
fig.patch.set_facecolor("white")

for ax, (title, key, col) in zip(axes, PANELS):
    ax.set_facecolor("white")
    ax.set_title(title, fontsize=11.5, weight="bold", color=INK, pad=10)
    ax.set_xscale("log")
    ax.set_xlim(0.28, 6.0)
    ax.set_xticks([0.3, 0.5, 1.0, 2.0, 4.0])
    ax.set_xticklabels(["300M", "500M", "1B", "2B", "4B"], fontsize=8.5, color=INK)
    ax.minorticks_off()
    ax.set_xlabel("Parameters", fontsize=9.5, color=INK)
    ax.grid(axis="y", color=LGRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRAY)
    ax.tick_params(colors=GRAY, labelsize=8.5)

    for label, pB, a2t, a2i, kind in POINTS:
        y = a2t if key == "a2t" else a2i
        if kind == "fe2":
            ax.scatter([pB], [y], s=130, c=ACC, ec="white", lw=1.2, zorder=5)
        elif kind == "fe1":
            ax.scatter([pB], [y], s=115, c=ACC2, ec="white", lw=1.2, zorder=5)
        else:
            ax.scatter([pB], [y], s=95, c=GRAY, ec="white", lw=1.0, zorder=4)
        dx, dy = OFFS[(key, label)]
        name = label.split("\n")[0] + ("  (supervised pair)" if
                (key == "a2i" and "ImageBind" in label) else "")
        sub = label.split("\n")[1]
        ours = kind in ("fe2", "fe1")
        ax.text(pB * dx, y + dy + 0.016, name, fontsize=8.2 if ours else 7.8,
                color=INK, weight="bold" if ours else "normal", ha="left", zorder=6)
        ax.text(pB * dx, y + dy - 0.012, sub, fontsize=7.0, color=GRAY, ha="left", zorder=6)

    g = GEMINI[key]
    ax.axhline(g, color=GRAY, lw=1.1, ls=(0, (4, 3)), zorder=1)
    ax.text(0.30, g + 0.012, "Gemini Embedding 2 (API, size undisclosed)",
            fontsize=7.4, color=GRAY, ha="left")

axes[0].set_ylabel("Average R@10 (both directions)", fontsize=9.5, color=INK)
axes[0].set_ylim(0.30, 0.75)
axes[1].set_ylim(0.25, 0.80)

# legend: two series (ours vs baselines)
h_fe2 = plt.Line2D([], [], marker="o", ls="", ms=9, mfc=ACC, mec="white", label="fusion-embedding-2")
h_fe1 = plt.Line2D([], [], marker="o", ls="", ms=8.5, mfc=ACC2, mec="white", label="fusion-embedding-1")
h_b = plt.Line2D([], [], marker="o", ls="", ms=8, mfc=GRAY, mec="white", label="baselines")
fig.legend(handles=[h_fe2, h_fe1, h_b], loc="lower center", ncol=3, frameon=False,
           fontsize=8.4, bbox_to_anchor=(0.5, -0.015))

fig.text(0.005, 0.995, "VGGSound-696 cross-modal retrieval — unified embedding models · R@10 averaged over both directions",
         fontsize=8.2, color=GRAY, ha="left", va="top", style="italic")
fig.tight_layout(rect=(0, 0.05, 1, 0.97))
out = os.path.join(HERE, "fe_positioning.png")
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("saved", out)
