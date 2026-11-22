"""Capa BRONZE: copia fiel de cada fuente, sin limpiar, con metadatos de ingesta.

Regla: aquí no se corrige nada. Todo valor se guarda como texto tal cual llegó
(salvo que la fuente ya traiga tipos, como SQLite), para poder auditar después
qué cambió en silver.
Metadatos: _fuente, _fila (posición en la fuente), _sha256_fuente, _ingestado_en.
"""
import gzip
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import io

TABLAS_SQL = ["clientes", "suscripciones", "facturas", "etiquetas_entrenamiento", "clientes_a_puntuar"]


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _meta(df: pd.DataFrame, fuente: str, sha: str, ts: str) -> pd.DataFrame:
    df = df.copy()
    df.insert(0, "_fila", range(len(df)))
    df["_fuente"] = fuente
    df["_sha256_fuente"] = sha
    df["_ingestado_en"] = ts
    return df


def _leer_jsonl(p: Path) -> pd.DataFrame:
    """Guarda cada línea como JSON crudo + versión de esquema detectada.

    tickets.jsonl mezcla dos esquemas (plataforma antigua en inglés y nueva en
    español con campos anidados); bronze no los unifica, solo los etiqueta.
    """
    opener = gzip.open if p.suffix == ".gz" else open
    filas = []
    with opener(p, "rt", encoding="utf-8") as f:
        for linea in f:
            linea = linea.rstrip("\n")
            if not linea.strip():
                continue
            try:
                claves = tuple(sorted(json.loads(linea).keys()))
                error = None
            except json.JSONDecodeError as e:
                claves, error = (), str(e)
            filas.append({"raw_json": linea, "esquema": "|".join(claves), "error_parseo": error})
    return pd.DataFrame(filas)


def run(data_dir, lake_dir) -> dict:
    data_dir = Path(data_dir)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    salidas = {}

    db = data_dir / "conecta.db"
    sha = _sha256(db)
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)   # solo lectura
    try:
        for t in TABLAS_SQL:
            df = pd.read_sql(f'SELECT * FROM "{t}"', con)
            salidas[t] = io.write(_meta(df, f"conecta.db/{t}", sha, ts), lake_dir, "bronze", t)
    finally:
        con.close()

    for nombre, archivo in [("tickets", "tickets.jsonl"), ("eventos", "eventos.jsonl.gz")]:
        p = data_dir / archivo
        df = _leer_jsonl(p)
        salidas[nombre] = io.write(_meta(df, archivo, _sha256(p), ts), lake_dir, "bronze", nombre)
    return salidas
