"""Orquestador: bronze -> silver -> gold -> reporte de calidad."""
import time

from . import bronze, config, gold, quality, silver


def run(data_dir=None, lake_dir=None, verbose=True):
    data_dir = data_dir or config.DATA_DIR
    lake_dir = lake_dir or config.LAKE_DIR
    t0 = time.time()
    for nombre, paso in [("bronze", lambda: bronze.run(data_dir, lake_dir)),
                         ("silver", lambda: silver.run(lake_dir)),
                         ("gold", lambda: gold.run(lake_dir))]:
        out = paso()
        if verbose:
            print(f"[{nombre}] {len(out)} tablas  ({time.time() - t0:.1f}s)")
    rep = quality.run(lake_dir)
    fallas = rep[rep["estado"] == "FALLA"]
    if verbose:
        print(f"[calidad] {len(rep)} checks · {len(fallas)} fallas")
    if len(fallas):
        raise AssertionError("Checks de calidad en silver fallaron:\n" + fallas.to_string())
    return rep


if __name__ == "__main__":
    run()
