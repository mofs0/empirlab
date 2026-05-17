"""Unified data read/write helpers with optional disk caching."""
from __future__ import annotations
import hashlib
import pathlib
import pandas as pd

_CACHE_DIR = pathlib.Path.home() / ".empirlab_cache"


def read_panel(path: str | pathlib.Path, **kwargs) -> pd.DataFrame:
    """Read csv / parquet / xlsx with automatic format detection."""
    p = pathlib.Path(path)
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(p, **kwargs)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(p, **kwargs)
    return pd.read_csv(p, **kwargs)


def write_panel(df: pd.DataFrame, path: str | pathlib.Path, **kwargs) -> None:
    """Write a DataFrame to csv / parquet / xlsx."""
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(p, **kwargs)
    elif suffix in (".xlsx", ".xls"):
        df.to_excel(p, **kwargs)
    else:
        df.to_csv(p, **kwargs)


def cache_csv(key: str, fetch_fn, force: bool = False) -> pd.DataFrame:
    """Cache the result of fetch_fn() to ~/.empirlab_cache/<hash>.csv.

    Parameters
    ----------
    key      : str          Unique string describing the data pull.
    fetch_fn : callable     Zero-arg function returning a DataFrame.
    force    : bool         Ignore existing cache and re-fetch.
    """
    _CACHE_DIR.mkdir(exist_ok=True)
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    path = _CACHE_DIR / f"{h}.csv"
    if path.exists() and not force:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    df = fetch_fn()
    df.to_csv(path)
    return df
