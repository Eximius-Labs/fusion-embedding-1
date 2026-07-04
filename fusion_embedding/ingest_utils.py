"""Archive-extraction helpers for the WavCaps-style multi-part zip ingests.

Extracted from ``modal_app`` so the strategy logic is unit-testable WITHOUT a Modal deploy —
the 2026-07-01 ASL/BBC incidents (unzip warning-exit treated as fatal; 7z silently extracting
nothing from a ``zip -FF`` output) all lived in untested inline code. The rules learned:

  * exit codes of ``unzip``/``zip``/``7z`` are NOT ground truth (warnings are nonzero; 7z can
    "succeed" extracting 0 files) — the extracted-file count is the only thing to trust;
  * spanned pkzip sets (``.zip`` + ``.z01..``) have three viable extraction routes, none of
    which works for every source: 7z reads spans natively; ``zip -s 0`` is the canonical
    unsplit; ``zip -FF`` is a salvage that happens to fit some archives (BBC).

``runner`` is injected so tests can simulate any tool behaviour (including the pathological
success-with-no-output) without the real binaries.
"""

from __future__ import annotations

import glob
import os
import subprocess
import time
from typing import Callable, Sequence


def retry(fn: Callable, *, attempts: int = 3, wait_s: float = 30.0,
          log: Callable = print, sleep: Callable = time.sleep):
    """Call ``fn`` with linear-backoff retries — for multi-GB HF downloads that die on transient
    server disconnects (BBC 2026-07-01: httpx.RemoteProtocolError mid-snapshot). Re-raises the
    last error. ``snapshot_download`` resumes finished files, so a retry costs only the
    in-flight part, not the whole download."""
    for a in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:                                   # noqa: BLE001
            log(f"  attempt {a}/{attempts} failed: {type(e).__name__}: {str(e)[:140]}")
            if a == attempts:
                raise
            sleep(wait_s * a)


def count_files(root: str, pattern: str = "*.flac") -> int:
    return sum(1 for _ in glob.glob(os.path.join(root, "**", pattern), recursive=True))


def extract_split_zip(
    zdir: str,
    source: str,
    audio_dir: str,
    *,
    merged_path: str | None = None,
    pattern: str = "*.flac",
    runner: Callable = subprocess.run,
    log: Callable = print,
) -> tuple[str, int]:
    """Extract ``{zdir}/{source}.zip`` (single or spanned) into ``audio_dir``.

    Tries strategies in order until the extracted-file count is > 0 (exit codes are logged but
    never trusted). Returns ``(strategy_name, n_files)``; raises ``RuntimeError`` if every
    strategy yields zero files. The zip parts in ``zdir`` are left in place — the caller frees
    them AFTER success (a failed strategy must not destroy the next one's input).
    """
    parts = sorted(os.listdir(zdir))
    main_zip = os.path.join(zdir, f"{source}.zip")
    merged = merged_path or os.path.join(os.path.dirname(zdir.rstrip("/\\")) or ".",
                                         f"{source}_merged.zip")
    for p in parts:
        try:
            sz = os.path.getsize(os.path.join(zdir, p))
            log(f"  part {p}: {sz / 1e9:.2f} GB")
        except OSError:
            pass

    def _try(label: str, cmds: Sequence[tuple[list, bytes | None]]) -> int:
        for cmd, inp in cmds:
            rc = runner(cmd, input=inp, stdout=subprocess.DEVNULL,
                        stderr=subprocess.STDOUT).returncode
            log(f"  [{label}] {' '.join(cmd[:2])}... rc={rc}")
        n = count_files(audio_dir, pattern)
        log(f"  [{label}] -> {n} files")
        return n

    spanned = any(p.endswith(".z01") for p in parts)
    strategies: list[tuple[str, list]] = []
    if not spanned:
        strategies.append(("plain", [(["unzip", "-o", "-q", main_zip, "-d", audio_dir], None)]))
    else:
        strategies.append(("7z-span", [(["7z", "x", "-y", f"-o{audio_dir}", main_zip], None)]))
        strategies.append(("unsplit", [(["zip", "-q", "-s", "0", main_zip, "--out", merged], None),
                                       (["unzip", "-o", "-q", merged, "-d", audio_dir], None)]))
        strategies.append(("FF", [(["zip", "-q", "-FF", main_zip, "--out", merged], b"y\ny\n"),
                                  (["unzip", "-o", "-q", merged, "-d", audio_dir], None)]))
    for label, cmds in strategies:
        n = _try(label, cmds)
        if n > 0:
            return label, n
    raise RuntimeError(f"extraction produced 0 files for {source} after "
                       f"{[s for s, _ in strategies]}")
