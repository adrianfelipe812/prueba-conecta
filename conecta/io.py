"""Lectura/escritura de capas del lake (parquet)."""
from pathlib import Path
import pandas as pd


def path(lake_dir, capa: str, tabla: str) -> Path:
    return Path(lake_dir) / capa / f"{tabla}.parquet"


def write(df: pd.DataFrame, lake_dir, capa: str, tabla: str) -> Path:
    p = path(lake_dir, capa, tabla)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return p


def read(lake_dir, capa: str, tabla: str) -> pd.DataFrame:
    return pd.read_parquet(path(lake_dir, capa, tabla))
