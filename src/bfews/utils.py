"""Shared helpers: configuration, paths, timing, peak-memory measurement, logging."""
from __future__ import annotations

import gc
import json
import logging
import multiprocessing as mp
import os
import resource
import sys
import time
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import yaml

# Keep Polars thread pool modest to minimize peak memory on laptops
os.environ.setdefault("POLARS_MAX_THREADS", "4")

ROOT = Path(__file__).resolve().parents[2]  # final/


@lru_cache
def cfg() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text())


def path(key: str) -> Path:
    p = Path(cfg()["paths"][key])
    return p if p.is_absolute() else (ROOT / p).resolve()


def raw_dir() -> Path:
    return path("raw")


def data_dir(layer: str = "") -> Path:
    p = path("data") / layer
    p.mkdir(parents=True, exist_ok=True)
    return p


def out_dir(sub: str = "") -> Path:
    p = path("outputs") / sub
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_logger(name: str) -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        log.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(name)s | %(message)s", "%H:%M:%S")
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(fmt)
        log.addHandler(h)
        fh = logging.FileHandler(out_dir("logs") / f"{name}.log", mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)
    return log


@contextmanager
def timer(label: str, log: logging.Logger | None = None):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    (log.info if log else print)(f"{label}: {dt:.2f}s")


def save_json(obj, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=str))


def df_to_markdown(df, floatfmt: str = ".4g") -> str:
    """Render a small Polars/pandas frame as a GitHub markdown table (no tabulate dependency)."""
    try:
        import polars as pl
        if isinstance(df, pl.DataFrame):
            cols, rows = df.columns, df.rows()
        else:
            cols, rows = list(df.columns), df.itertuples(index=False, name=None)
    except ImportError:  # pragma: no cover
        cols, rows = list(df.columns), df.itertuples(index=False, name=None)

    def f(v):
        if isinstance(v, float):
            return format(v, floatfmt) if v == v else ""
        return "" if v is None else str(v)

    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(f(v) for v in r) + " |" for r in rows]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fair benchmarking: every measurement runs in a fresh *spawned* process so that
# peak RSS (ru_maxrss) and caches of one run cannot leak into the next.
# IMPORTANT: Do NOT pass or return large DataFrames across mp.Queue; only pass
# lightweight arguments (paths/scalars) and return scalar summaries.
# ---------------------------------------------------------------------------
def _child(q, fn, args, kwargs):
    os.environ["POLARS_MAX_THREADS"] = "4"
    base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()
    try:
        res = fn(*args, **kwargs)
        # Only retain scalar summary if a frame was returned
        scalar_res = len(res) if hasattr(res, "__len__") and not isinstance(res, (dict, str)) else res
        del res
        gc.collect()
        err = None
    except Exception as exc:  # noqa: BLE001
        scalar_res, err = None, repr(exc)
    dt = time.perf_counter() - t0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    q.put({"seconds": dt, "peak_rss_mb": peak / 1024, "delta_rss_mb": (peak - base) / 1024,
           "result": scalar_res, "error": err})


def measure(fn, *args, repeat: int = 1, **kwargs) -> dict:
    """Run fn(*args) `repeat` times in fresh processes; return median time and max peak RSS."""
    ctx = mp.get_context("spawn")
    runs = []
    for _ in range(repeat):
        q = ctx.Queue()
        p = ctx.Process(target=_child, args=(q, fn, args, kwargs))
        p.start()
        r = q.get()
        p.join()
        if r["error"]:
            raise RuntimeError(r["error"])
        runs.append(r)
    secs = sorted(r["seconds"] for r in runs)
    return {"median_s": secs[len(secs) // 2], "min_s": secs[0], "max_s": secs[-1],
            "peak_rss_mb": max(r["peak_rss_mb"] for r in runs),
            "delta_rss_mb": max(r["delta_rss_mb"] for r in runs),
            "result": runs[-1]["result"]}


def available_memory_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except ImportError:  # pragma: no cover
        return float(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")) / 1e9
