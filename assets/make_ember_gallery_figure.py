"""Ember text->thermal retrieval gallery figure (HF pack card, fig 2).

Composes fe2_ember_retrieval_gallery.png from the artifacts produced by
scripts/make_thermal_gallery.py (gallery.json + thumb_<i>.png): one row per
query, top-5 ranked thermal thumbnails, the exact match outlined in green with
its rank. Follows the audio gallery pattern and the brand palette.

Self-checking: every caption line must fit the figure width; the script refuses
to save on violation.

Run: uv run --with matplotlib python assets/make_ember_gallery_figure.py --data-dir <dir>
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

INK = "#1b2f4b"
GRAY = "#7d8a9c"
GREEN = "#2e8b57"
HERE = os.path.dirname(os.path.abspath(__file__))

DROP_BUCKETS: set = set()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "fe2_ember_retrieval_gallery.png"))
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.data_dir, "gallery.json"), encoding="utf-8"))
    rows = [q for q in meta["queries"] if q["bucket"] not in DROP_BUCKETS]
    assert 4 <= len(rows) <= 6, f"expected 4-6 rows, got {len(rows)}"
    # display contract (cannot regress): rank-1 rows; siblings collapsed at the
    # strict thresholds. The job computed and verified the pairwise numbers on the
    # full-resolution images; re-assert them here from the stored contract.
    dc = meta["display_contract"]
    assert all(q["true_rank"] == 1 for q in rows), "non-rank-1 row"
    flat = [i for q in rows for i in q["top5_idx"]]
    assert len(set(flat)) == len(flat), "image reused across rows"
    assert dc["phash_threshold"] >= 20 and dc["cos_threshold"] <= 0.97,         f"weaker thresholds than the contract: {dc}"
    assert dc["min_pairwise_phash"] > dc["phash_threshold"],         f"pairwise phash {dc['min_pairwise_phash']} within threshold {dc['phash_threshold']}"
    assert dc["max_pairwise_cos_same_size"] < dc["cos_threshold"],         f"same-size cosine {dc['max_pairwise_cos_same_size']} reaches {dc['cos_threshold']}"

    n = len(rows)
    row_h, cap_h, pad = 1.55, 0.34, 0.12
    fig_w = 11.5
    fig_h = 0.62 + n * (row_h + cap_h + pad) + 0.78
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)
    fig.patch.set_facecolor("white")

    captions = []
    fig.text(0.013, 1 - 0.30 / fig_h,
             "Ember: text → thermal retrieval  (release holdout, 2,000-image gallery)",
             fontsize=12, color=INK, weight="bold", va="center")

    for r_i, q in enumerate(rows):
        top = 1 - (0.62 + r_i * (row_h + cap_h + pad)) / fig_h
        cap = f"“{q['caption_first_sentence']}”"
        t = fig.text(0.013, top - 0.5 * cap_h / fig_h, cap, fontsize=8.6, color=INK,
                     va="center")
        captions.append((q["bucket"], t))
        for c_i, idx in enumerate(q["top5_idx"]):
            im = Image.open(os.path.join(args.data_dir, f"thumb_{idx}.png"))
            ax = fig.add_axes([0.013 + c_i * 0.198,
                               top - (cap_h + row_h) / fig_h,
                               0.185, row_h / fig_h])
            ax.imshow(im, cmap=None)
            ax.set_xticks([]); ax.set_yticks([])
            exact = q["top5_is_exact"][c_i]
            for s in ax.spines.values():
                s.set_edgecolor(GREEN if exact else "#c6cdd8")
                s.set_linewidth(3.0 if exact else 1.0)
            if exact:
                ax.text(0.03, 0.06, f"exact match · rank {c_i + 1}",
                        transform=ax.transAxes, fontsize=6.8, color="white",
                        weight="bold",
                        bbox=dict(facecolor=GREEN, edgecolor="none", pad=1.6))

    foot_lines = [
        "Examples shown are rank-1 retrievals (green: the query's exact match, retrieved first). "
        "Displayed results are de-duplicated: near-identical frames from the same source sequence "
        "are collapsed to their top-ranked representative.",
        "Queries shortened to their first sentence for display; retrieval used the full captions.",
        "Images from the IR-TD dataset (IRGPT, ICCV 2025), shown for evaluation illustration "
        "under its academic license."]
    for f_i, fl in enumerate(foot_lines):
        t = fig.text(0.013, (0.56 - f_i * 0.17) / fig_h, fl, fontsize=7.2, color=GRAY,
                     va="center")
        captions.append((f"footer{f_i}", t))

    # self-check: captions must fit the canvas width
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    bad = []
    for name, t in captions:
        ext = t.get_window_extent(renderer=ren)
        if ext.x1 > fig.bbox.x1 - 6:
            bad.append(f"  caption for {name!r} overflows the canvas")
    if bad:
        raise SystemExit("figure self-check failed:\n" + "\n".join(bad))
    print(f"self-check: {len(captions)} captions OK")

    fig.savefig(args.out, facecolor="white")
    plt.close(fig)
    print("saved", args.out)


if __name__ == "__main__":
    main()
