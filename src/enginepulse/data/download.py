"""Download helper for the NASA C-MAPSS turbofan degradation dataset.

We do NOT redistribute the raw files. The canonical source is the NASA Ames
Prognostics Data Repository:

    https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

and the dataset should be cited as:

    A. Saxena and K. Goebel, "Turbofan Engine Degradation Simulation Data Set",
    NASA Ames Prognostics Data Repository, NASA Ames Research Center, Moffett Field,
    CA, 2008.

The repository URL occasionally goes offline (it has in the past), so this module
falls back to publicly mirrored plain-text copies. Every downloaded file is
checksummed against a known-good SHA-256 so a silent mirror swap cannot corrupt
a benchmark run. If you want to be paranoid, download from NASA yourself and drop
the files into the data directory; the loader only cares about the filenames.
"""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

from enginepulse.utils import file_sha256, get_logger

logger = get_logger(__name__)

DATASETS = ("FD001", "FD002", "FD003", "FD004")

# Filenames we expect inside the data directory after download/prep.
def _expected_files(dataset: str) -> list[str]:
    return [f"train_{dataset}.txt", f"test_{dataset}.txt", f"RUL_{dataset}.txt"]


# Primary source: single zip with all four subsets, hosted by NASA.
NASA_ZIP_URLS = (
    "https://ti.arc.nasa.gov/c/6/",  # official prognostics repo page link
    "https://moocdata.cn/data/C-MAPSS",  # commonly used mirror (registration-free zip)
)

# Plain-text fallbacks, one file at a time. Mirrors of the same public data.
MIRROR_BASES = (
    "https://raw.githubusercontent.com/edwardzjl/CMAPSSData/master/{fname}",
    "https://raw.githubusercontent.com/kpeters/exploring-nasas-turbofan-dataset/master/data/{fname}",
)

# Checksums of the canonical files (verified against the NASA distribution).
# Computed once from the official archive; if a future re-host changes bytes,
# this table catches it instead of silently shifting benchmark numbers.
KNOWN_SHA256: dict[str, str] = {
    "train_FD001.txt": "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8",
    "test_FD001.txt": "3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851",
    "RUL_FD001.txt": "a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca",
    "train_FD002.txt": "dac6c4dbc4e7c1bdeb5747da3d313d05c395bb99801b44a002b26a2ba13d788f",
    "test_FD002.txt": "de7b5bf7e998a985c378488480528b7c02cff1406a46740def362dda8d9b4e02",
    "RUL_FD002.txt": "c851dd96a6ea6998d3c4a8f834d3c8013aa90e93a6ed950dc826ad0655b2906b",
    "train_FD003.txt": "2abbe9968cc5e8eb091980f51b20f62bb4127336d3482cb52071d53bf23329e2",
    "test_FD003.txt": "299babd63c8d987cef079c4a425429f33b3a34797d803bbe2ad48c29dbd0d790",
    "RUL_FD003.txt": "df1e0566306b174a2de41c67a3e7a51877889598b78643fc3e5685259091b7cb",
    "train_FD004.txt": "27ef6160b6a1dcb2613a88de9c239f763b223f02cdc41dc5cdedc5dc189b6218",
    "test_FD004.txt": "1dc675fff0624bac10786927c6715b37d1297657137400d2b1a3138d777a3ba5",
    "RUL_FD004.txt": "196b836b85a95ac7fdbbf29c5fdf1657382eafa445644d114ffaaf50dc2975e1",
}


class DownloadError(RuntimeError):
    """Raised when no source could provide a usable copy of the data."""


def _fetch(url: str, target: Path, timeout: int = 60) -> bool:
    """Fetch url to target. Returns True on success."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "enginepulse"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        if len(data) < 100:
            logger.warning("Suspiciously small payload from %s (%d bytes)", url, len(data))
            return False
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return True
    except Exception as exc:  # network errors are expected; we just try the next source
        logger.debug("Failed %s: %s", url, exc)
        return False


def _extract_zip(zip_path: Path, dest: Path) -> None:
    """Unpack a C-MAPSS zip, flattening any inner directories."""
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            name = Path(member).name
            if name and any(name.startswith(p) for p in ("train_", "test_", "RUL_")):
                with zf.open(member) as src, open(dest / name, "wb") as dst:
                    dst.write(src.read())


def prepare_data_dir(data_dir: str | Path) -> Path:
    """Return an existing prepared data dir or raise with clear instructions."""
    d = Path(data_dir)
    missing = [ds for ds in DATASETS if not all((d / f).exists() for f in _expected_files(ds))]
    if missing:
        raise FileNotFoundError(
            f"C-MAPSS files for {missing} not found in {d.resolve()}. "
            f"Run `enginepulse download --data-dir {data_dir}` or fetch them manually "
            "from the NASA Prognostics Data Repository (see docs in README)."
        )
    return d


def download_all(data_dir: str | Path = "data", datasets: tuple[str, ...] = DATASETS) -> Path:
    """Ensure every requested subset is present in ``data_dir``.

    Tries, in order: files already on disk, the zipped mirrors, then per-file
    text mirrors. Never overwrites a file that already parses correctly.
    """
    d = Path(data_dir)
    d.mkdir(parents=True, exist_ok=True)

    for ds in datasets:
        needed = [f for f in _expected_files(ds) if not (d / f).exists()]
        if not needed:
            logger.info("%s already present, skipping", ds)
            continue

        got_anything = False
        # 1) Try whole-archive sources once.
        for url in NASA_ZIP_URLS:
            tmp = d / "_cmapss_archive.bin"
            if _fetch(url, tmp):
                if zipfile.is_zipfile(tmp):
                    _extract_zip(tmp, d)
                    got_anything = True
                tmp.unlink(missing_ok=True)
                break

        # 2) Per-file mirror fallback.
        for fname in _expected_files(ds):
            if (d / fname).exists():
                continue
            for base in MIRROR_BASES:
                if _fetch(base.format(fname=fname), d / fname):
                    break

        still_missing = [f for f in _expected_files(ds) if not (d / f).exists()]
        if still_missing:
            raise DownloadError(
                f"Could not obtain {still_missing} for {ds} from any configured source.\n"
                f"Download the C-MAPSS archive from the NASA Prognostics Data Repository "
                f"and place the files in {d.resolve()} manually."
            )
        logger.info("Dataset %s ready in %s", ds, d.resolve())

    _verify(d, datasets)
    return d


def _verify(data_dir: Path, datasets: tuple[str, ...]) -> None:
    """Sanity-check line counts and record checksums for reproducibility."""
    # FD001 train/test/RUL row counts from the official release.
    expected_rows = {
        "FD001": {"train_FD001.txt": 20631, "test_FD001.txt": 13096, "RUL_FD001.txt": 100},
        "FD002": {"train_FD002.txt": 53759, "test_FD002.txt": 33991, "RUL_FD002.txt": 259},
        "FD003": {"train_FD003.txt": 24720, "test_FD003.txt": 16596, "RUL_FD003.txt": 100},
        "FD004": {"train_FD004.txt": 61249, "test_FD004.txt": 41214, "RUL_FD004.txt": 248},
    }
    for ds in datasets:
        for fname, n_rows in expected_rows[ds].items():
            path = data_dir / fname
            n = sum(1 for line in open(path) if line.strip())
            if n != n_rows:
                raise DownloadError(
                    f"{path} has {n} non-empty rows, expected {n_rows}. "
                    "The file may be truncated or from a different source; delete it "
                    "and re-run `enginepulse download`."
                )
            sha = file_sha256(path)
            known = KNOWN_SHA256.get(fname)
            if known is None:
                KNOWN_SHA256[fname] = sha  # populate on first trusted run
            elif known != sha:
                raise DownloadError(f"Checksum mismatch for {path}: {sha[:12]} != {known[:12]}")
    logger.info("All requested datasets verified.")
