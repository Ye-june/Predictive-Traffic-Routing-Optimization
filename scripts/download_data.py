"""Download METR-LA traffic speeds and DCRNN sensor metadata.

Raw files are written only to ``data/raw/`` and are never overwritten if a
checksum-matching copy already exists.

Sources
-------
* Traffic matrix: Hugging Face mirrors of ``metr-la.h5``
* Sensor IDs and coordinates: DCRNN repository
  https://github.com/liyaguang/DCRNN
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import requests
from tqdm import tqdm

from trafficflow.utils.config import load_config
from trafficflow.utils.logging import get_logger
from trafficflow.utils.paths import get_project_root, resolve_path

logger = get_logger(__name__)

_CHUNK = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ssl_verify() -> str | bool:
    try:
        import certifi

        return certifi.where()
    except ImportError:
        return True


def _download_with_requests(url: str, destination: Path, timeout: int) -> None:
    verify = _ssl_verify()
    with requests.get(url, stream=True, timeout=timeout, verify=verify) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length") or 0)
        tmp_path = destination.with_suffix(destination.suffix + ".part")
        with tmp_path.open("wb") as handle, tqdm(
            total=total or None,
            unit="B",
            unit_scale=True,
            desc=destination.name,
        ) as progress:
            for chunk in response.iter_content(chunk_size=_CHUNK):
                if not chunk:
                    continue
                handle.write(chunk)
                progress.update(len(chunk))
        tmp_path.replace(destination)


def _download_with_curl(url: str, destination: Path) -> None:
    curl = shutil.which("curl") or shutil.which("curl.exe")
    if curl is None:
        raise RuntimeError("curl is not available for TLS fallback.")
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    logger.info("Using curl TLS fallback for %s", url)
    subprocess.run(
        [curl, "-L", "--fail", "--retry", "3", "-o", str(tmp_path), url],
        check=True,
    )
    tmp_path.replace(destination)


def download_file(url: str, destination: Path, *, timeout: int = 120) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s", url, destination)
    try:
        _download_with_requests(url, destination, timeout)
    except Exception as exc:
        message = str(exc)
        if "SSL" not in message and "CERTIFICATE" not in message:
            raise
        logger.warning("Python SSL verification failed (%s); retrying with curl.", exc)
        _download_with_curl(url, destination)


def _ensure_file(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        logger.info("Already present: %s", destination)
        return
    download_file(url, destination)


def download_traffic(config: dict) -> Path:
    paths = config["paths"]
    download_cfg = config["download"]
    destination = resolve_path(paths["traffic_h5"])
    expected_sha = (download_cfg.get("traffic_sha256") or "").lower()

    if destination.exists() and destination.stat().st_size > 0:
        if expected_sha:
            actual = sha256_file(destination)
            if actual == expected_sha:
                logger.info("Traffic file checksum matches: %s", destination)
                return destination
            logger.warning(
                "Existing file checksum mismatch (got %s, expected %s). Re-downloading.",
                actual,
                expected_sha,
            )
        else:
            logger.info("Using existing traffic file: %s", destination)
            return destination

    errors: list[str] = []
    for url in download_cfg["traffic_urls"]:
        try:
            download_file(url, destination)
            if expected_sha:
                actual = sha256_file(destination)
                if actual != expected_sha:
                    raise ValueError(
                        f"Checksum mismatch for {url}: got {actual}, expected {expected_sha}"
                    )
            return destination
        except Exception as exc:  # noqa: BLE001 - try each mirror
            errors.append(f"{url}: {exc}")
            logger.warning("Download failed from %s (%s)", url, exc)

    raise RuntimeError(
        "Could not download metr-la.h5 from any configured URL:\n" + "\n".join(errors)
    )


def main() -> int:
    config = load_config("data")
    raw_dir = resolve_path(config["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Project root: %s", get_project_root())
    traffic_path = download_traffic(config)
    logger.info("Traffic file: %s (%s bytes)", traffic_path, traffic_path.stat().st_size)

    download_cfg = config["download"]
    _ensure_file(
        download_cfg["sensor_locations_url"],
        resolve_path(config["paths"]["sensor_locations"]),
    )
    _ensure_file(
        download_cfg["sensor_ids_url"],
        resolve_path(config["paths"]["sensor_ids"]),
    )
    _ensure_file(
        download_cfg["distances_url"],
        resolve_path(config["paths"]["distances"]),
    )
    logger.info("Download complete. Raw files are in %s", raw_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
