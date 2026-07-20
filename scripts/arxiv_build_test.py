"""Build-test the arXiv bundle with pdflatex (arXiv's toolchain), on Modal CPU.

arXiv runs TeX Live pdflatex, while local development compiles with tectonic
(XeTeX). This job compiles the exact submission bundle the way arXiv will:
pdflatex -> (bibtex skipped: main.bbl shipped) -> pdflatex x2, and reports
undefined references, errors, and the page count.

Run:
    PYTHONUTF8=1 uv run modal run scripts/arxiv_build_test.py
"""
import modal

app = modal.App("fusion-arxiv-build-test")

image = (
    modal.Image.from_registry("texlive/texlive:latest", add_python="3.11")
    .pip_install("pymupdf")
    .add_local_dir("docs/paper/arxiv_bundle", "/root/bundle")
)


@app.function(image=image, timeout=1200)
def build() -> dict:
    import re
    import subprocess

    def run(cmd):
        # pdflatex emits non-UTF8 bytes on stdout; capture raw, judge via main.log
        return subprocess.run(cmd, cwd="/root/bundle", capture_output=True,
                              timeout=600)

    results = []
    for i in range(3):
        r = run(["pdflatex", "-interaction=nonstopmode", "main.tex"])
        results.append(r.returncode)

    log = open("/root/bundle/main.log", encoding="utf-8", errors="replace").read()
    errors = re.findall(r"^! .*", log, re.M)
    undef = re.findall(r"Warning: (Reference|Citation) `[^']+' .*undefined", log)
    overfull = re.findall(r"Overfull \\hbox \((\d+\.?\d*)pt", log)
    big_overfull = [f for f in overfull if float(f) > 20]

    import fitz
    d = fitz.open("/root/bundle/main.pdf")
    text_p1 = d[0].get_text()

    out = {
        "returncodes": results,
        "errors": errors[:10],
        "undefined": undef[:10],
        "n_overfull_gt20pt": len(big_overfull),
        "pages": d.page_count,
        "title_ok": "Fusion Embedding" in text_p1,
        "authors_ok": all(n in text_p1 for n in
                          ("Tonmoy", "Hoque", "Arham", "Luthra")),
        "todo_in_pdf": any("TODO" in d[i].get_text() for i in range(d.page_count)),
    }
    print("ARXIV_BUILD_TEST:", out)
    return out


@app.local_entrypoint()
def main():
    r = build.remote()
    assert not r["errors"] and not r["undefined"] and r["title_ok"], r
    print("BUILD OK:", r["pages"], "pages")
