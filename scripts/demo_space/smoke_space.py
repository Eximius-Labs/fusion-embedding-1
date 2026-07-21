"""Headless smoke of the private demo Space via gradio_client.

Exercises one query per tab against the running Space runtime and prints
result shapes. Long timeouts: on cpu-basic the first query loads a 2B model.

Run:  PYTHONUTF8=1 uv run --with gradio_client python scripts/demo_space/smoke_space.py
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def token() -> str:
    for line in open(os.path.join(HERE, "..", "..", ".env"), encoding="utf-8"):
        if line.startswith("HF_TOKEN="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("no token")


def main() -> None:
    from gradio_client import Client, handle_file

    c = Client("EximiusLabs/fusion-embedding-demo", hf_token=token())
    print("connected; endpoints:", len(c.endpoints))

    r1 = c.predict("church bells ringing", api_name="/text_to_sound")
    n_aud = sum(1 for x in r1[:5] if x)
    print("tab1 text->sound: returned", len(r1), "outputs;", n_aud, "audio hits")

    ex = [f for f in os.listdir(os.path.join(HERE, "_staging", "examples"))
          if f.endswith(".ogg")]
    if ex:
        p = os.path.join(HERE, "_staging", "examples", ex[0])
        r2 = c.predict(handle_file(p), True, api_name="/sound_to_images")
        print("tab2 sound->images: gallery size", len(r2) if r2 else 0)

    r3 = c.predict("waves crashing on a beach", None, api_name="/one_space")
    print("tab3 one-space: ranking chars", len(r3[0]) if r3 else 0)


if __name__ == "__main__":
    main()
