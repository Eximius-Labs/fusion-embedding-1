"""Extraction-strategy gate: the multi-part zip logic that broke in prod (2026-07-01) —
now tested WITHOUT Modal: fake runners simulate every observed tool pathology, plus a
real-binaries integration test (skipped where zip/unzip aren't installed)."""

import os
import shutil
import subprocess
import tempfile
import types

import pytest

from fusion_embedding.ingest_utils import count_files, extract_split_zip, retry


def test_retry_recovers_after_transient_failures():
    calls = {"n": 0}
    sleeps = []

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("server disconnected")
        return "ok"

    out = retry(flaky, attempts=3, wait_s=10, log=lambda *_: None, sleep=sleeps.append)
    assert out == "ok" and calls["n"] == 3
    assert sleeps == [10, 20]                                     # linear backoff, no sleep after success


def test_retry_reraises_after_exhaustion():
    def always_fails():
        raise ConnectionError("nope")

    sleeps = []
    with pytest.raises(ConnectionError):
        retry(always_fails, attempts=3, wait_s=1, log=lambda *_: None, sleep=sleeps.append)
    assert len(sleeps) == 2                                       # attempts-1 waits, then re-raise


def _setup(spanned=True):
    zdir = tempfile.mkdtemp()
    audio = tempfile.mkdtemp()
    open(os.path.join(zdir, "SRC.zip"), "wb").write(b"x" * 10)
    if spanned:
        open(os.path.join(zdir, "SRC.z01"), "wb").write(b"x" * 10)
    return zdir, audio


def _fake_runner(succeed_on, audio_dir, n_files=3, rc=0):
    """Stateful runner producing files only when strategy `succeed_on`'s EXTRACTION command runs
    (an unzip belongs to whichever zip-rewrite preceded it — that's how the real chain works)."""
    calls = []
    state = {"unsplit": False, "FF": False}

    def _write():
        for i in range(n_files):
            open(os.path.join(audio_dir, f"f{i}.flac"), "wb").write(b"a")

    def run(cmd, input=None, stdout=None, stderr=None):
        calls.append(cmd)
        tool = cmd[0]
        if tool == "zip" and "-s" in cmd:
            state["unsplit"] = True
        elif tool == "zip" and "-FF" in cmd:
            state["FF"] = True
        elif tool == "7z" and succeed_on == "7z-span":
            _write()
        elif tool == "unzip":
            if succeed_on == "plain":
                _write()
            elif succeed_on == "FF" and state["FF"]:
                _write()
            elif succeed_on == "unsplit" and state["unsplit"] and not state["FF"]:
                _write()
        return types.SimpleNamespace(returncode=rc)

    run.calls = calls
    return run


def test_first_strategy_success_stops_chain():
    zdir, audio = _setup(spanned=True)
    run = _fake_runner("7z-span", audio)
    label, n = extract_split_zip(zdir, "SRC", audio, runner=run, log=lambda *_: None)
    assert (label, n) == ("7z-span", 3)
    assert all(c[0] == "7z" for c in run.calls)                   # never touched zip/unzip


def test_falls_through_to_unsplit_then_ff():
    zdir, audio = _setup(spanned=True)
    run = _fake_runner("FF", audio)
    label, n = extract_split_zip(zdir, "SRC", audio, runner=run, log=lambda *_: None)
    assert label == "FF" and n == 3
    tools = [(c[0], "-FF" in c or "-s" in c) for c in run.calls]
    assert tools[0][0] == "7z"                                    # tried span-read first
    assert any(t == ("zip", True) for t in tools)                 # then a zip rewrite


def test_nonzero_exit_codes_are_not_fatal_when_files_appear():
    """The BBC failure mode: unzip rc=1 (warnings) but files extracted fine."""
    zdir, audio = _setup(spanned=False)
    run = _fake_runner("plain", audio, rc=1)                      # rc=1 on every command
    label, n = extract_split_zip(zdir, "SRC", audio, runner=run, log=lambda *_: None)
    assert (label, n) == ("plain", 3)


def test_zero_files_with_success_exit_is_failure():
    """The ASL failure mode: 7z rc=0 but extracts nothing -> must fall through, then raise."""
    zdir, audio = _setup(spanned=True)
    run = _fake_runner("nothing-succeeds", audio, rc=0)           # rc=0 everywhere, no files ever
    with pytest.raises(RuntimeError, match="0 files"):
        extract_split_zip(zdir, "SRC", audio, runner=run, log=lambda *_: None)
    assert len(run.calls) == 5                                    # all 3 strategies exhausted (1+2+2 cmds)


def test_parts_left_in_place_for_caller():
    zdir, audio = _setup(spanned=True)
    run = _fake_runner("7z-span", audio)
    extract_split_zip(zdir, "SRC", audio, runner=run, log=lambda *_: None)
    assert sorted(os.listdir(zdir)) == ["SRC.z01", "SRC.zip"]     # cleanup is the caller's job


@pytest.mark.skipif(shutil.which("zip") is None or shutil.which("unzip") is None,
                    reason="zip/unzip not installed")
def test_real_spanned_zip_roundtrip():
    """Integration: build a REAL split archive with `zip -s`, extract via the strategy chain."""
    src = tempfile.mkdtemp(); zdir = tempfile.mkdtemp(); audio = tempfile.mkdtemp()
    for i in range(8):
        open(os.path.join(src, f"clip{i}.flac"), "wb").write(os.urandom(64 * 1024))
    # split into ~100KB spans -> SRC.zip + SRC.z01..; zip needs cwd for relative naming
    subprocess.run(["zip", "-q", "-s", "100k", "-r", os.path.join(zdir, "SRC.zip"), "."],
                   cwd=src, check=True)
    assert any(f.endswith(".z01") for f in os.listdir(zdir))      # genuinely spanned
    label, n = extract_split_zip(zdir, "SRC", audio, log=lambda *_: None)
    assert n == 8, (label, n)
    assert count_files(audio) == 8
