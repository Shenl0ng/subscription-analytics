"""Download the public datasets once, verify them, and load them with explicit types.

The raw files are not committed: they are fetched from their public sources and checked
against a SHA-256, so an analysis can never silently run on a different file.
"""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Source:
    filename: str
    url: str
    sha256: str


COOKIE_CATS = Source(
    "cookie_cats.csv",
    "https://raw.githubusercontent.com/robguilarr/ab_testing_cookie_cats/main/datasets/cookie_cats.csv",
    "5ab54d761fbddcd50de7b88e4eaf7837cba4569474f50c043a4d17ee342c46bd",
)
TELCO = Source(
    "telco_churn.csv",
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91",
)


class DataError(Exception):
    """The source file is not the one the analysis was written for."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(source: Source, data_dir: Path) -> Path:
    path = data_dir / source.filename
    if not path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        urllib.request.urlretrieve(source.url, tmp)
        tmp.replace(path)
    digest = sha256(path)
    if digest != source.sha256:
        raise DataError(f"{path} has sha256 {digest}, expected {source.sha256}")
    return path


def load_cookie_cats(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(fetch(COOKIE_CATS, data_dir))
    return df.astype({"retention_1": bool, "retention_7": bool})


def load_telco(data_dir: Path) -> pd.DataFrame:
    """Telco churn with typed columns.

    ``TotalCharges`` is blank for customers with 0 months of tenure (they have not been
    billed yet). It is filled with 0 rather than dropped: those customers exist.
    """
    df = pd.read_csv(fetch(TELCO, data_dir))
    total = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce")
    blank = total.isna()
    if not (df.loc[blank, "tenure"] == 0).all():
        raise DataError("TotalCharges is blank for customers with tenure > 0")
    return df.assign(TotalCharges=total.fillna(0.0), churned=df["Churn"].eq("Yes"))
