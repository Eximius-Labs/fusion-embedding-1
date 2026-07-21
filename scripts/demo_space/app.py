"""Fusion Embedding demo Space: one embedding space for text, images, and audio.

Runs the released EximiusLabs/fusion-embedding-2-2b-preview checkpoint at a
pinned revision through its public remote-code path. Galleries are
pre-embedded; only the query is encoded live. ZeroGPU when available, CPU
fallback otherwise (slow but functional).
"""
import json
import os

import numpy as np
import torch
import gradio as gr

try:  # ZeroGPU decorator; no-op off Spaces
    import spaces

    def gpu_wrap(fn):
        return spaces.GPU(duration=90)(fn)
except Exception:  # pragma: no cover
    def gpu_wrap(fn):
        return fn

MODEL_ID = "EximiusLabs/fusion-embedding-2-2b-preview"
REVISION = "ad45028fbeb638fcc333690bd6119e02a66dc7d7"  # v0.2-preview
AUDIO_INSTR = "Retrieve audio by sound description."
UNIFIED_INSTR = "Retrieve audio or images relevant to the user's query."

NAVY, TERRA, CREAM = "#1F3A5F", "#C46A4A", "#F5EFE6"

_here = os.path.dirname(os.path.abspath(__file__))


def _load_index(name):
    d = torch.load(os.path.join(_here, name), map_location="cpu", weights_only=False)
    emb = d["emb"].float()
    emb = torch.nn.functional.normalize(emb, dim=-1)
    center = emb.mean(0, keepdim=True)
    centered = torch.nn.functional.normalize(emb - center, dim=-1)
    return {"emb": emb, "centered": centered, "center": center, "meta": d["meta"]}


AUDIO_IDX = _load_index("audio_index.pt")
IMAGE_IDX = _load_index("image_index.pt")
EXAMPLES = json.load(open(os.path.join(_here, "examples.json"), encoding="utf-8"))

_model = None


def _get_model():
    global _model
    if _model is None:
        from transformers import AutoModel
        m = AutoModel.from_pretrained(MODEL_ID, revision=REVISION,
                                      trust_remote_code=True,
                                      torch_dtype=torch.float32)
        m.eval()
        _model = m
    if torch.cuda.is_available() and next(_model.parameters()).device.type != "cuda":
        _model = _model.to("cuda")
    return _model


@gpu_wrap
def _embed_text(text, instruction):
    m = _get_model()
    with torch.no_grad():
        e = m.embed_text(text, instruction=instruction)
    return e.detach().float().cpu()


@gpu_wrap
def _embed_audio_file(path):
    import librosa
    wav, _ = librosa.load(path, sr=16000, mono=True)
    if len(wav) > 16000 * 30:
        wav = wav[: 16000 * 30]
    m = _get_model()
    with torch.no_grad():
        e = m.embed_audio(wav, sr=16000)
    return e.detach().float().cpu()


def _topk(q, idx, k, centered=False, q_center=None):
    q = q.reshape(1, -1)
    if centered:
        qc = q_center if q_center is not None else q.mean(0, keepdim=True) * 0
        q = torch.nn.functional.normalize(q - qc, dim=-1)
        g = idx["centered"]
    else:
        q = torch.nn.functional.normalize(q, dim=-1)
        g = idx["emb"]
    scores = (g @ q.T).squeeze(1)
    top = torch.topk(scores, min(k, g.shape[0]))
    return [(idx["meta"][i], float(s)) for i, s in zip(top.indices.tolist(),
                                                      top.values.tolist())]


def text_to_sound(query):
    if not (query or "").strip():
        return [gr.update(value=None, visible=False)] * 5 + [gr.update(value="", visible=False)] * 5
    q = _embed_text(query.strip(), AUDIO_INSTR)
    hits = _topk(q, AUDIO_IDX, 5)
    audios, caps = [], []
    for meta, score in hits:
        audios.append(gr.update(value=os.path.join(_here, meta["file"]), visible=True))
        caps.append(gr.update(
            value=f"**{meta['caption']}** · cosine {score:.3f}  \n"
                  f"<sub>[source]({meta['source']}) · {meta['license'].split('/')[-3] if '/' in meta['license'] else meta['license']}</sub>",
            visible=True))
    while len(audios) < 5:
        audios.append(gr.update(value=None, visible=False))
        caps.append(gr.update(value="", visible=False))
    return audios + caps


def sound_to_images(audio_path, centered):
    if not audio_path:
        return []
    q = _embed_audio_file(audio_path)
    hits = _topk(q, IMAGE_IDX, 12, centered=bool(centered),
                 q_center=AUDIO_IDX["center"])
    return [(os.path.join(_here, m["file"]), f"{m['caption']} · {s:.3f}")
            for m, s in hits]


def one_space(query_text, audio_path):
    if audio_path:
        q = _embed_audio_file(audio_path)
        qname = "your audio clip"
    elif (query_text or "").strip():
        q = _embed_text(query_text.strip(), UNIFIED_INSTR)
        qname = f"“{query_text.strip()}”"
    else:
        return "Enter a text query or provide an audio clip.", [], *(
            [gr.update(value=None, visible=False)] * 3)
    ah = [(m, s, "audio") for m, s in _topk(q, AUDIO_IDX, 10)]
    ih = [(m, s, "image") for m, s in _topk(q, IMAGE_IDX, 10)]
    merged = sorted(ah + ih, key=lambda x: -x[1])[:10]
    lines = [f"Top matches for {qname} across both galleries, one similarity metric:"]
    for r, (m, s, kind) in enumerate(merged, 1):
        icon = "\U0001F50A" if kind == "audio" else "\U0001F5BC"
        lines.append(f"{r}. {icon} {m['caption']} — {s:.3f}")
    gallery = [(os.path.join(_here, m["file"]), f"#{r+1} · {m['caption']} · {s:.3f}")
               for r, (m, s, k) in enumerate(merged) if k == "image"]
    audio_hits = [(m, s) for m, s, k in merged if k == "audio"][:3]
    audio_out = []
    for m, s in audio_hits:
        audio_out.append(gr.update(value=os.path.join(_here, m["file"]),
                                   label=f"{m['caption']} · {s:.3f}", visible=True))
    while len(audio_out) < 3:
        audio_out.append(gr.update(value=None, visible=False))
    return "\n".join(lines), gallery, *audio_out


CSS = f"""
.fe-header {{background:{NAVY}; color:{CREAM}; padding:18px 22px; border-radius:14px;}}
.fe-header h1 {{margin:0 0 6px 0; font-size:1.5rem;}}
.fe-header p {{margin:0; opacity:.92;}}
.fe-header a {{color:{CREAM}; text-decoration:underline;}}
.fe-accent {{color:{TERRA}; font-weight:600;}}
footer {{visibility:hidden}}
"""

with gr.Blocks(css=CSS, theme=gr.themes.Soft(), title="Fusion Embedding demo") as demo:
    gr.HTML(f"""
    <div class="fe-header">
      <h1>Fusion Embedding</h1>
      <p>One embedding space for text, images, video, and audio — from frozen models.
      This demo runs the released
      <a href="https://huggingface.co/{MODEL_ID}">fusion-embedding-2-2b-preview</a>
      checkpoint, revision-pinned. Galleries are pre-embedded; your query is encoded live.
      First query loads the model and takes longest.</p>
    </div>""")

    with gr.Tab("Sound → Images"):
        gr.Markdown(
            "Upload or record a sound and retrieve matching **images**. "
            "<span class='fe-accent'>The model was never trained on a single "
            "audio–image pair</span> — audio is aligned to text only; because the "
            "frozen base already binds text and images, retrieval between audio and "
            "images emerges.")
        with gr.Row():
            with gr.Column(scale=1):
                aud_in = gr.Audio(type="filepath", sources=["upload", "microphone"],
                                  label="Query sound")
                centered = gr.Checkbox(True, label="Centered readout (recommended)")
                btn2 = gr.Button("Retrieve images", variant="primary")
                gr.Examples([[os.path.join(_here, e["file"])] for e in EXAMPLES],
                            inputs=[aud_in],
                            label="Example sounds (" + " / ".join(e["caption"] for e in EXAMPLES[:3]) + " …)")
            with gr.Column(scale=2):
                gal = gr.Gallery(label="Retrieved images", columns=4, height=520)
        btn2.click(sound_to_images, [aud_in, centered], [gal], api_name="sound_to_images")

    with gr.Tab("Text → Sound"):
        gr.Markdown("Describe a sound; retrieve real recordings from a "
                    f"{AUDIO_IDX['emb'].shape[0]}-clip gallery (CC0/CC-BY, attributed).")
        q_in = gr.Textbox(label="Describe a sound",
                          placeholder="heavy rain on a tin roof")
        btn1 = gr.Button("Retrieve sounds", variant="primary")
        gr.Examples([["dog barking in the distance"], ["heavy rain on a tin roof"],
                     ["someone playing acoustic guitar"], ["church bells ringing"],
                     ["a helicopter passing overhead"]], inputs=[q_in])
        auds, caps = [], []
        for i in range(5):
            with gr.Row():
                with gr.Column(scale=1):
                    a = gr.Audio(visible=False, label=f"match {i+1}")
                with gr.Column(scale=1):
                    c = gr.Markdown(visible=False)
            auds.append(a); caps.append(c)
        btn1.click(text_to_sound, [q_in], auds + caps, api_name="text_to_sound")

    with gr.Tab("One Space"):
        gr.Markdown("One query, both galleries, **one similarity metric** — "
                    "text, audio, and images live in the same vector space.")
        with gr.Row():
            uq = gr.Textbox(label="Text query (or provide audio)",
                            placeholder="waves crashing on a beach")
            ua = gr.Audio(type="filepath", sources=["upload"],
                          label="…or an audio query")
        btn3 = gr.Button("Search the space", variant="primary")
        ranking = gr.Markdown()
        ugal = gr.Gallery(label="Image matches", columns=5, height=300)
        uauds = [gr.Audio(visible=False) for _ in range(3)]
        btn3.click(one_space, [uq, ua], [ranking, ugal, *uauds], api_name="one_space")

    gr.Markdown(
        f"---\nModels: [fusion-embedding-1]"
        "(https://huggingface.co/EximiusLabs/fusion-embedding-1-2b-preview) · "
        "[fusion-embedding-2](https://huggingface.co/EximiusLabs/fusion-embedding-2-2b-preview) "
        "· [code](https://github.com/Eximius-Labs/fusion-embedding) · both "
        "generations are on the public MTEB audio (MAEB) and video (MVEB) "
        "leaderboards via the official `mteb` harness.  \n"
        "Gallery media are CC0/CC-BY with per-item attribution in "
        "`audio_attribution.json` and `image_attribution.json`. Scores are cosine "
        "similarities in the shared space.")

if __name__ == "__main__":
    demo.launch()
