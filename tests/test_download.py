"""Download-helper tests. No network: everything is monkeypatched or local."""

from __future__ import annotations

import io
import zipfile

import pytest

from enginepulse.data import download as dl


def test_expected_files_lists_three_per_dataset():
    files = dl._expected_files("FD001")
    assert files == ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"]


def test_prepare_data_dir_raises_when_files_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="enginepulse download"):
        dl.prepare_data_dir(tmp_path / "nothing_here")


def test_prepare_data_dir_accepts_complete_layout(tmp_path):
    for f in dl._expected_files("FD001") + dl._expected_files("FD002") + \
             dl._expected_files("FD003") + dl._expected_files("FD004"):
        (tmp_path / f).write_text("x")
    assert dl.prepare_data_dir(tmp_path) == tmp_path


def test_fetch_returns_false_without_network(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise OSError("no network in tests")

    monkeypatch.setattr(dl.urllib.request, "urlopen", boom)
    ok = dl._fetch("https://example.invalid/x.zip", tmp_path / "x.zip")
    assert ok is False


def test_fetch_rejects_tiny_payload(monkeypatch, tmp_path):
    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(dl.urllib.request, "urlopen", lambda req, timeout=60: Resp(b"tiny"))
    assert dl._fetch("https://example.invalid/x.zip", tmp_path / "x.zip") is False


def test_extract_zip_flattens_inner_dirs(tmp_path):
    zpath = tmp_path / "a.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("nested/dir/train_FD001.txt", "1 1 100 500 0.02\n")
        zf.writestr("nested/readme.txt", "ignore me")
    dest = tmp_path / "out"
    dest.mkdir()
    dl._extract_zip(zpath, dest)
    assert (dest / "train_FD001.txt").exists()
    assert not (dest / "readme.txt").exists()


def test_verify_rejects_truncated_file(tmp_path):
    (tmp_path / "train_FD001.txt").write_text("one line only")
    with pytest.raises(dl.DownloadError, match="expected 20631"):
        dl._verify(tmp_path, ("FD001",))


def test_download_all_raises_clear_error_when_no_source(tmp_path, monkeypatch):
    """With every URL unreachable the user must get actionable guidance."""
    monkeypatch.setattr(dl, "_fetch", lambda url, target, timeout=60: False)
    with pytest.raises(dl.DownloadError, match="NASA Prognostics Data Repository"):
        dl.download_all(tmp_path / "data", datasets=("FD001",))


def test_download_all_skips_files_already_present(tmp_path, monkeypatch):
    calls: list[str] = []

    def spy(url, target, timeout=60):
        calls.append(url)
        return False

    monkeypatch.setattr(dl, "_fetch", spy)
    # Pretend FD001 exists; verification will still fail on row counts, so we
    # only assert the "already present" branch avoided any fetch attempt.
    for f in dl._expected_files("FD001"):
        (tmp_path / f).write_text("placeholder")
    with pytest.raises(dl.DownloadError):
        dl.download_all(tmp_path, datasets=("FD001",))
    assert calls == []
