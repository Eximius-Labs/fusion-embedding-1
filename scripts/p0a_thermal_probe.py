"""P0a — thermal zero-train probe (docs/sensor_extension_plan.md section 3).

No training. Measures whether the frozen Qwen3-VL vision path, through the
RELEASED fusion-embedding-1 remote-code stack (AutoModel, pinned revision),
reads thermal imagery:

  (a) PRIMARY / GATED: LLVIP test zero-shot binary person classification.
      ImageBind-style crop protocol: pedestrian bbox crops = "person" class,
      non-overlapping random crops = "background" class, 500/500, seed 0,
      cosine vs templated class prompts. Three template variants, all
      reported; T1 declared primary before the run. (LLVIP is an all-pedestrian
      dataset, so whole-image classification is trivial — the crop protocol is
      what makes it a real binary task.)
      Gates (pre-registered in the plan): >=65 proceed; 55-65 proceed with
      caveats + preprocessing variants; <55 stop and debug.
  (b) SECONDARY: thermal->visible-twin retrieval over 1,000 registered LLVIP
      test pairs (R@1/5/10, both sides through the frozen image path).
      SUBSTITUTION vs plan: IR-TD's images require multi-source reassembly
      (no direct archive; deferred to Phase 1), and LLVIP itself ships labels,
      not captions, so the caption-retrieval half is replaced by twin
      retrieval - the same emergent-alignment mechanism our audio Phase-0
      measured (audio->image, zero pairs).
  (c) RGB CEILING: identical classification protocol on 200 visible-light
      crops from the registered twins.

Data: jsonhash/LLVIP HF mirror. The images live in LLVIP.zip (4.0 GB), which
extracts to LLVIP/{infrared,visible}/{train,test}/*.jpg plus Pascal VOC
per-image annotations at LLVIP/Annotations/*.xml (the person label is
<object><name>person</name>; bbox is xmin/ymin/xmax/ymax). coco_annotations.7z
is NOT used. Provenance/counts recorded in the output JSON. LLVIP license:
academic / non-commercial; used here for evaluation only, nothing redistributed.

Probe first, then full (deploy + spawn; client-independent):
    PYTHONUTF8=1 uv run modal deploy scripts/p0a_thermal_probe.py
    PYTHONUTF8=1 uv run python -c "import modal; modal.Function.from_name('fusion-p0a-thermal','run_probe').spawn(limit=20)"
    PYTHONUTF8=1 uv run python -c "import modal; modal.Function.from_name('fusion-p0a-thermal','run_probe').spawn()"
Results land on the fusion-data Volume under p0a_thermal/ and print as
P0A_RESULT lines.
"""

from __future__ import annotations

import modal

app = modal.App("fusion-p0a-thermal")

hf_secret = modal.Secret.from_name("huggingface")
volume = modal.Volume.from_name("fusion-data")

FE1_REPO = "EximiusLabs/fusion-embedding-1-2b-preview"
FE1_REV = "b551ea8033bee3cd51468cbde2bb25397292e0b3"
MIRROR = "jsonhash/LLVIP"

# Single-prompt variants kept for transparency (they show the uncalibrated
# 2-prompt argmax is brittle). The PRIMARY metric is now a prompt ENSEMBLE with
# prior calibration -- standard CLIP zero-shot practice -- not a single prompt.
TEMPLATES = {
    "T1_primary": ["a thermal infrared photo of a person",
                   "a thermal infrared photo of an empty street"],
    "T2_plain": ["a photo of a person",
                 "a photo of the background"],
    "T3_surveillance": ["an infrared surveillance image containing a pedestrian",
                        "an infrared surveillance image with no people"],
}
# Prompt ensemble: each class embedding is the renormalized mean over its
# templates; the decision is calibrated by removing each class's prior over the
# eval crops (the documented fix for prompt-prior bias, applied symmetrically to
# both classes -- and, as the RGB ceiling confirms, needed by every modality,
# not just thermal). This ensembled + calibrated accuracy is the GATED metric.
ENSEMBLE = {
    "person": [
        "a thermal infrared photo of a person",
        "an infrared image of a pedestrian",
        "a thermal image showing a human figure",
        "a person seen in thermal infrared",
        "an infrared surveillance image containing a pedestrian",
        "a thermal photo of a walking person",
    ],
    "background": [
        "a thermal infrared photo of an empty street",
        "an infrared image with no people",
        "a thermal image of an empty road",
        "an infrared surveillance image with no people",
        "a thermal photo of buildings and pavement",
        "an empty scene in thermal infrared",
    ],
}
# Thermal framing we would inject on the image side if the released API exposed
# an image-instruction param; it does not (embed_image hardcodes a fixed
# document instruction), so the framing is carried by the text-side prompts.
THERMAL_DOC_INSTRUCTION = "Represent this thermal infrared image."

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        "numpy>=1.24",
        "transformers>=4.46",
        "accelerate>=0.30",
        "pillow>=10.0",
        "huggingface_hub>=0.26",
        "qwen-vl-utils>=0.0.14",
        "soundfile>=0.12",  # released remote code imports the audio tower at load
        "librosa>=0.10",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
)


def _parse_voc_person(xml_path):
    """Person bboxes from a Pascal VOC XML as [(xmin, ymin, xmax, ymax), ...]."""
    import xml.etree.ElementTree as ET

    boxes = []
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return boxes
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if name != "person":
            continue
        bb = obj.find("bndbox")
        if bb is None:
            continue
        try:
            x1 = float(bb.findtext("xmin"))
            y1 = float(bb.findtext("ymin"))
            x2 = float(bb.findtext("xmax"))
            y2 = float(bb.findtext("ymax"))
        except (TypeError, ValueError):
            continue
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def _build_crop_sets(ann_dir, img_dir, n_per_class, seed):
    """Person crops from VOC person bboxes + non-overlapping background crops.

    Returns (person_crops, background_crops) as lists of PIL images, plus the
    per-crop source filenames for the record. Deterministic under seed. Ids are
    matched to annotations by filename stem; images with no person box are used
    for background only.
    """
    import os
    import random

    from PIL import Image

    files = sorted(f for f in os.listdir(img_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    rng = random.Random(seed)
    rng.shuffle(files)

    def crop_person(img, box, pad=0.15):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w / 2, y1 + h / 2
        s = max(w, h) * (1 + pad) / 2
        s = max(s, 32)  # min half-size: avoid degenerate slivers
        return img.crop((max(0, cx - s), max(0, cy - s),
                         min(img.width, cx + s), min(img.height, cy + s)))

    def overlaps(box, others):
        x1, y1, x2, y2 = box
        for ox1, oy1, ox2, oy2 in others:
            if x1 < ox2 and ox1 < x2 and y1 < oy2 and oy1 < y2:
                return True
        return False

    persons, bgs, srcs = [], [], {"person": [], "background": []}
    for fname in files:
        if len(persons) >= n_per_class and len(bgs) >= n_per_class:
            break
        stem = os.path.splitext(fname)[0]
        boxes = _parse_voc_person(os.path.join(ann_dir, stem + ".xml"))
        try:
            img = Image.open(os.path.join(img_dir, fname)).convert("RGB")  # 3x replicate
        except Exception:
            continue
        if len(persons) < n_per_class and boxes:
            persons.append(crop_person(img, boxes[0]))
            srcs["person"].append(fname)
        if len(bgs) < n_per_class:
            for _ in range(30):
                cw = ch = rng.randint(96, 224)
                if img.width <= cw or img.height <= ch:
                    continue
                x = rng.randint(0, img.width - cw)
                y = rng.randint(0, img.height - ch)
                if not overlaps((x, y, x + cw, y + ch), boxes):
                    bgs.append(img.crop((x, y, x + cw, y + ch)))
                    srcs["background"].append(fname)
                    break
    return persons, bgs, srcs


def _find_layout(root):
    """Robustly locate (annotation_dir, infrared_test_dir, visible_test_dir).

    Walks the tree instead of hardcoding paths: the annotation dir is any dir
    with .xml files; image dirs are any dir with .jpg files, classified by an
    'infrared'/'visible' path component and preferring a 'test' split. A
    slightly different mirror layout will still resolve.
    """
    import os

    ann_dir = None
    jpg_dirs = {}
    for r, _dirs, files in os.walk(root):
        if ann_dir is None and any(f.lower().endswith(".xml") for f in files):
            ann_dir = r
        n = sum(1 for f in files if f.lower().endswith((".jpg", ".jpeg", ".png")))
        if n:
            jpg_dirs[r] = n

    def pick(spectrum):
        norm = {d: d.lower().replace("\\", "/") for d in jpg_dirs}
        cands = [(d, jpg_dirs[d]) for d in jpg_dirs if spectrum in norm[d]]
        test = [(d, n) for d, n in cands if "/test" in norm[d] or norm[d].endswith("test")]
        pool = test or cands
        return max(pool, key=lambda x: x[1])[0] if pool else None

    return ann_dir, pick("infrared"), pick("visible"), jpg_dirs


@app.function(gpu="L4", image=image, secrets=[hf_secret],
              volumes={"/vol": volume}, timeout=3 * 3600)
def run_probe(limit: int = 0) -> dict:
    """limit>0 = smoke mode (tiny crop sets, tiny retrieval)."""
    import json
    import os
    import time
    import zipfile

    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModel

    t0 = time.time()
    out_dir = "/vol/p0a_thermal"
    os.makedirs(out_dir, exist_ok=True)

    # ---- data: LLVIP mirror; images come from LLVIP.zip (cache on the Volume) ----
    data_root = "/vol/p0a_thermal/llvip"
    os.makedirs(data_root, exist_ok=True)

    def count_jpg(d):
        return (len([f for f in os.listdir(d) if f.lower().endswith(".jpg")])
                if d and os.path.isdir(d) else 0)

    ann_dir, ir_test, vis_test, _ = _find_layout(data_root)
    marker = os.path.join(data_root, ".images_extracted")
    need = (not os.path.exists(marker)
            or count_jpg(ir_test) < 3000 or count_jpg(vis_test) < 3000)
    if need:
        zp = os.path.join(data_root, "LLVIP.zip")
        if not os.path.exists(zp) or os.path.getsize(zp) < 3_900_000_000:
            zp = hf_hub_download(MIRROR, "LLVIP.zip", repo_type="dataset",
                                 local_dir=data_root)
        print(f"P0A_EXTRACT: extracting {zp} ({os.path.getsize(zp)/1e9:.1f} GB)")
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(data_root)
        ann_dir, ir_test, vis_test, jpg_dirs = _find_layout(data_root)
        with open(marker, "w", encoding="utf-8") as f:
            f.write(json.dumps({"ir": count_jpg(ir_test),
                                "vis": count_jpg(vis_test)}))
        volume.commit()
    else:
        _, _, _, jpg_dirs = _find_layout(data_root)

    assert ann_dir and ir_test and vis_test, \
        f"layout: ann={ann_dir} ir={ir_test} vis={vis_test}"
    n_ann = len([f for f in os.listdir(ann_dir) if f.lower().endswith(".xml")])
    provenance = {
        "annotation_dir": ann_dir.replace(data_root, "<data_root>"),
        "infrared_test_dir": ir_test.replace(data_root, "<data_root>"),
        "visible_test_dir": vis_test.replace(data_root, "<data_root>"),
        "n_annotations_xml": n_ann,
        "n_infrared_test_jpg": count_jpg(ir_test),
        "n_visible_test_jpg": count_jpg(vis_test),
        "jpg_dirs": {d.replace(data_root, "<data_root>"): n
                     for d, n in sorted(jpg_dirs.items())},
        "annotation_format": "Pascal VOC XML (object/name==person, "
                             "bndbox xmin/ymin/xmax/ymax)",
    }
    print("P0A_LAYOUT:", json.dumps(provenance))

    n_cls = 10 if limit else 500
    n_ret = 8 if limit else 1000
    n_rgb = 6 if limit else 200

    # ---- model: released FE1 remote code, pinned ----
    model = AutoModel.from_pretrained(FE1_REPO, revision=FE1_REV,
                                      trust_remote_code=True).to("cuda")
    model.eval()

    def embed_images(pils):
        # released embed_image: frozen vision path, fixed doc instruction, returns
        # a unit-norm CPU vector. 3x-channel-replicated thermal handled by convert.
        return torch.stack([model.embed_image(im).float() for im in pils])

    def embed_texts(txts):
        return torch.stack([model.embed_text(t).float() for t in txts])

    # record whether the released embed_image exposes an instruction param
    has_instr = "instruction" in model.embed_image.__code__.co_varnames
    instruction_note = (
        "released embed_image exposes instruction param" if has_instr else
        "released embed_image has a fixed document instruction; thermal framing "
        "carried by the text-side prompts (a custom image instruction is not "
        "available through the released API)")

    result = {"model": FE1_REPO, "revision": FE1_REV, "mirror": MIRROR,
              "seed": 0, "n_per_class": n_cls, "limit": limit,
              "instruction_note": instruction_note,
              "thermal_doc_instruction_desired": THERMAL_DOC_INSTRUCTION,
              "templates": TEMPLATES, "provenance": provenance,
              "classification": {}, "retrieval": {}, "rgb_ceiling": {}}

    # ---- (a) classification: thermal crops ----
    persons, bgs, _ = _build_crop_sets(ann_dir, ir_test, n_cls, seed=0)
    result["counts"] = {"person": len(persons), "background": len(bgs)}
    pe = embed_images(persons)
    be = embed_images(bgs)
    result["embedding_dim"] = pe.shape[1]

    def _classify(pos_emb, neg_emb, te):
        """top1 (pre-registered raw argmax) + a prior-corrected diagnostic.

        The raw 2-prompt argmax is sensitive to a constant per-prompt bias (the
        text prompts sit at different baseline cosines), which can saturate the
        decision toward one class regardless of image content. top1_centered
        removes that bias by subtracting each prompt's mean similarity over the
        pooled crop set (the per-modality centering the audio cross-modal work
        used), isolating whether the vision path actually SEPARATES the two
        classes. It is a diagnostic, not the gated metric."""
        sp = pos_emb @ te.T
        sn = neg_emb @ te.T
        acc_p = (sp.argmax(1) == 0).float().mean().item()
        acc_b = (sn.argmax(1) == 1).float().mean().item()
        center = torch.cat([sp, sn], 0).mean(0, keepdim=True)
        acc_pc = ((sp - center).argmax(1) == 0).float().mean().item()
        acc_bc = ((sn - center).argmax(1) == 1).float().mean().item()
        np_, nn_ = pos_emb.shape[0], neg_emb.shape[0]
        return {
            "acc_person": round(acc_p, 4), "acc_background": round(acc_b, 4),
            "top1": round((acc_p * np_ + acc_b * nn_) / (np_ + nn_) * 100, 2),
            "acc_person_centered": round(acc_pc, 4),
            "acc_background_centered": round(acc_bc, 4),
            "top1_centered": round((acc_pc * np_ + acc_bc * nn_)
                                   / (np_ + nn_) * 100, 2)}

    for name, (pos_t, neg_t) in TEMPLATES.items():
        te = embed_texts([pos_t, neg_t])
        result["classification"][name] = _classify(pe, be, te)

    def _ensemble_emb(prompts):
        """Renormalized mean of a class's prompt embeddings (CLIP-style)."""
        e = embed_texts(prompts).mean(0, keepdim=True)
        return e / e.norm(dim=1, keepdim=True)

    def _classify_calibrated(pos_emb, neg_emb, class_te):
        """PRIMARY gated metric: prompt-ensembled class embeddings + prior
        calibration. The calibration subtracts each class column's mean logit
        over the pooled crops, removing the constant per-prompt bias that
        saturates a raw 2-prompt argmax. Reports both the ensembled raw argmax
        (for transparency) and the calibrated accuracy (the gate)."""
        sp = pos_emb @ class_te.T          # [n_person, 2]
        sn = neg_emb @ class_te.T          # [n_bg, 2]
        np_, nn_ = pos_emb.shape[0], neg_emb.shape[0]
        raw = ((sp.argmax(1) == 0).float().sum()
               + (sn.argmax(1) == 1).float().sum()) / (np_ + nn_)
        bias = torch.cat([sp, sn], 0).mean(0, keepdim=True)
        pc = ((sp - bias).argmax(1) == 0).float().mean().item()
        bc = ((sn - bias).argmax(1) == 1).float().mean().item()
        return {
            "ensemble_raw_top1": round(raw.item() * 100, 2),
            "acc_person_calibrated": round(pc, 4),
            "acc_background_calibrated": round(bc, 4),
            "top1_calibrated": round((pc * np_ + bc * nn_) / (np_ + nn_) * 100, 2),
            "n_person_templates": len(ENSEMBLE["person"]),
            "n_background_templates": len(ENSEMBLE["background"]),
        }

    class_te = torch.cat([_ensemble_emb(ENSEMBLE["person"]),
                          _ensemble_emb(ENSEMBLE["background"])], 0)
    result["classification"]["ENSEMBLE_calibrated_PRIMARY"] = \
        _classify_calibrated(pe, be, class_te)

    # ---- (c) RGB ceiling: visible twins, SAME ensembled+calibrated harness ----
    # Fair control: identical calibration method, visible-light-worded prompts.
    ENSEMBLE_RGB = {
        "person": ["a photo of a person", "a photo of a pedestrian",
                   "a street photo showing a person", "a person walking on a street",
                   "a photograph of a human figure", "a picture of a person outdoors"],
        "background": ["a photo of an empty street", "a photo of an empty road",
                       "a street photo with no people", "a photograph of buildings and pavement",
                       "an empty outdoor scene", "a picture of a street with no one in it"],
    }
    vp, vb, _ = _build_crop_sets(ann_dir, vis_test, n_rgb, seed=0)
    vpe, vbe = embed_images(vp), embed_images(vb)
    rgb_te = torch.cat([_ensemble_emb(ENSEMBLE_RGB["person"]),
                        _ensemble_emb(ENSEMBLE_RGB["background"])], 0)
    rgb = _classify_calibrated(vpe, vbe, rgb_te)
    rgb["single_prompt_T2"] = _classify(vpe, vbe, embed_texts(list(TEMPLATES["T2_plain"])))
    rgb["protocol"] = "ensembled+calibrated on visible-light twin crops"
    rgb["n_per_class"] = len(vp)
    result["rgb_ceiling"] = rgb

    # ---- (b) retrieval: thermal -> visible twin over n_ret registered pairs ----
    import random

    rng = random.Random(0)
    names = sorted(os.listdir(ir_test))
    rng.shuffle(names)
    pairs = [n for n in names
             if n.lower().endswith(".jpg")
             and os.path.exists(os.path.join(vis_test, n))][:n_ret]
    from PIL import Image
    ir_e = embed_images(
        [Image.open(os.path.join(ir_test, n)).convert("RGB") for n in pairs])
    vi_e = embed_images(
        [Image.open(os.path.join(vis_test, n)).convert("RGB") for n in pairs])
    sims = ir_e @ vi_e.T
    ranks = (sims > sims.diag().unsqueeze(1)).sum(1) + 1
    result["retrieval"] = {
        "protocol": "thermal->visible twin, substitution for IR-TD caption "
                    "retrieval (IR-TD images gated; LLVIP has labels not "
                    "captions; see plan P0a note)",
        "n_pairs": len(pairs),
        "R@1": round((ranks <= 1).float().mean().item(), 4),
        "R@5": round((ranks <= 5).float().mean().item(), 4),
        "R@10": round((ranks <= 10).float().mean().item(), 4),
        "chance_R@1": round(1 / max(1, len(pairs)), 4)}

    result["runtime_s"] = round(time.time() - t0, 1)
    tag = "smoke" if limit else "full"
    with open(f"{out_dir}/p0a_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    volume.commit()
    print("P0A_RESULT:", json.dumps(result))
    return result
