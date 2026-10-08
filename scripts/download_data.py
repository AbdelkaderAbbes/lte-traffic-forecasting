#!/usr/bin/env python
"""Download Dataset_02.zip (LTE 1800 MHz) from Zenodo, verify its checksum and extract the CSV into data/.

Zenodo record: https://doi.org/10.5281/zenodo.17815388   (CC BY 4.0)
Paper: Lehoczky et al., Scientific Data (2026), doi:10.1038/s41597-026-07723-0
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://zenodo.org/records/17815388/files/Dataset_02.zip?download=1"
MD5 = "7d9c1c019e5d18bf318b4e4ad9393d91"   # as listed on the Zenodo record page
DEST = Path(__file__).resolve().parents[1] / "data"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    DEST.mkdir(exist_ok=True)
    z = DEST / "Dataset_02.zip"
    if not z.exists():
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, z)
    got = md5(z)
    if got != MD5:
        sys.exit(f"checksum mismatch: expected {MD5}, got {got}. Delete {z} and retry, or download manually (data/README.md).")
    with zipfile.ZipFile(z) as zf:
        zf.extractall(DEST)
        print("extracted:", *zf.namelist(), sep="\n  ")
    print("\nIf the CSV name differs from data/Dataset_02_LTE_1800.csv, set [data].path in configs/default.toml or pass --data.")


if __name__ == "__main__":
    main()
