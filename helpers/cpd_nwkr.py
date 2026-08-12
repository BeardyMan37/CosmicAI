"""Fast NWKR-background change-point detection: detrend once, then segment.

The naive version (helpers/cpd_nwkr.py) refits NWKR inside every candidate
segment, which Dynp calls O(n^2) times -- roughly O(n^4) overall, intractable
past n ~ 500.

This version uses the standard nonparametric-regression change-point
construction (Loader 1996): fit the nonparametric background *once*, then detect
change points in the residuals. The background family is still NWKR with the
same bandwidth as the scan statistic, so the comparison the reviewers asked for
is preserved -- but the segmentation runs entirely in Ruptures' compiled
KernelCPD, and the NWKR fit costs O(n*r) once via the truncated kernel.

Complexity: O(n*r) for the fit + O(n^2) compiled DP, versus O(n^4) in Python.

Registry keys:  ``cpdf_nwkr_gaussian``, ``cpdf_nwkr_laplace``
Ablation rungs: ``cpdf_mean``, ``cpdf_poly2``
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

__all__ = ["nwkr_residuals", "scan_row_cpd_fast", "CPDF_METHODS"]


# ---------------------------------------------------------------------------
# Background fits (once per row)
# ---------------------------------------------------------------------------

def _trunc_kernel(w: float, kind: str, C: float = 2.0) -> np.ndarray:
    """One-sided-to-symmetric truncated kernel vector, radius C*w."""
    r = max(1, int(round(C * float(w))))
    d = np.arange(-r, r + 1, dtype=np.float64)
    if kind == "gaussian":
        k = np.exp(-(d * d) / (float(w) * float(w)))
    elif kind == "laplace":
        k = np.exp(-np.abs(d) / max(float(w), 1e-12))
    else:
        raise ValueError(f"Unknown kernel kind {kind!r}")
    return k


def nwkr_residuals(x: np.ndarray, w: float, kind: str = "gaussian",
                   C: float = 3.0) -> np.ndarray:
    """x minus its truncated Nadaraya-Watson fit. O(n*r) via convolution.

    Same estimator as the scan statistic, same bandwidth, same truncation.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    k = _trunc_kernel(w, kind, C)
    num = np.convolve(x, k, mode="same")
    den = np.convolve(np.ones_like(x), k, mode="same")
    fit = num / np.maximum(den, 1e-12)
    return x - fit


def _poly_residuals(x: np.ndarray, degree: int = 2) -> np.ndarray:
    n = x.size
    t = np.linspace(-1.0, 1.0, n)
    V = np.vander(t, degree + 1, increasing=True)
    beta, *_ = np.linalg.lstsq(V, x, rcond=None)
    return x - V @ beta


def _residuals(x: np.ndarray, family: str, fkw: dict) -> np.ndarray:
    if family == "nwkr_gaussian":
        return nwkr_residuals(x, fkw["w"], "gaussian", fkw.get("C", 3.0))
    if family == "nwkr_laplace":
        return nwkr_residuals(x, fkw["w"], "laplace", fkw.get("C", 3.0))
    if family == "poly":
        return _poly_residuals(x, int(fkw.get("degree", 2)))
    if family == "mean":
        return x - x.mean()
    raise ValueError(f"Unknown family {family!r}")


# ---------------------------------------------------------------------------
# Row-level entry point
# ---------------------------------------------------------------------------

def scan_row_cpd_fast(
    x: np.ndarray,
    W: int,
    *,
    family: str = "nwkr_gaussian",
    family_kwargs: Optional[dict] = None,
    algo: str = "kernelcpd",
    min_size: int = 3,
    n_bkps: int = 2,
    width_cap: Optional[int] = None,
    jump: int = 1,
) -> Tuple[float, Tuple[int, int]]:
    """Detrend with ``family``, segment the residuals, return the best segment.

    ``algo``
        ``"kernelcpd"``  Ruptures KernelCPD, compiled exact DP (recommended)
        ``"binseg"``     greedy binary segmentation, O(n log n), fastest
        ``"window"``     sliding window, O(n), approximate

    Returns ``(score, (a, b))`` on the repo's standard contract, with the score
    being the paper's normalised statistic so it is comparable against tau_S.
    """
    from benchmark_regressor_on_syth import _epidemic_score

    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    fkw = dict(family_kwargs or {})
    cap = int(width_cap if width_cap is not None else 3 * W)

    if n < 4 * min_size:
        return 0.0, (0, -1)

    try:
        resid = _residuals(x, family, fkw)
        sig = resid.reshape(-1, 1)

        import ruptures as rpt
        if algo == "kernelcpd":
            # Compiled dynamic programming; linear kernel == L2 cost
            algo_obj = rpt.KernelCPD(kernel="linear", min_size=int(min_size),
                                     jump=int(jump)).fit(sig)
        elif algo == "binseg":
            algo_obj = rpt.Binseg(model="l2", min_size=int(min_size),
                                  jump=int(jump)).fit(sig)
        elif algo == "window":
            algo_obj = rpt.Window(width=max(4, min(cap, n // 4)), model="l2",
                                  min_size=int(min_size), jump=int(jump)).fit(sig)
        else:
            raise ValueError(f"Unknown algo {algo!r}")
        bkps = algo_obj.predict(n_bkps=int(n_bkps))
    except Exception as exc:
        log.warning("cpdf(%s/%s) failed on a row (n=%d): %s",
                    family, algo, n, exc)
        return 0.0, (0, -1)

    pts = sorted({0, n} | {int(p) for p in bkps if 0 < int(p) < n})
    if len(pts) < 3:
        return 0.0, (0, -1)

    # Pick the segment whose residual variance departs most from the whole row.
    # This is the segmentation's own objective, not a constant-mean rescore, so
    # the background family is not silently swapped back to F_0.
    total_ss = float(resid @ resid)
    best_sel, best_ab = -np.inf, (0, -1)
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1] - 1
        if b <= a:
            continue
        if b - a + 1 > cap:
            c = (a + b) // 2
            a = max(0, c - cap // 2)
            b = min(n - 1, a + cap - 1)
        inside = resid[a:b + 1]
        outside = np.concatenate([resid[:a], resid[b + 1:]])
        ss = float(np.sum((inside - inside.mean()) ** 2))
        if outside.size:
            ss += float(np.sum((outside - outside.mean()) ** 2))
        sel = (total_ss - ss) / total_ss if total_ss > 1e-18 else 0.0
        if sel > best_sel:
            best_sel, best_ab = sel, (a, b)

    a, b = best_ab
    if b <= a:
        return 0.0, (0, -1)
    return float(_epidemic_score(x, a, b)), (int(a), int(b))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CPDF_METHODS: Dict[str, dict] = {
    "cpdf_nwkr_gaussian": {"mode": "cpdf", "family": "nwkr_gaussian",
                           "algo": "kernelcpd", "min_size": 3, "n_bkps": 2},
    "cpdf_nwkr_laplace":  {"mode": "cpdf", "family": "nwkr_laplace",
                           "algo": "kernelcpd", "min_size": 3, "n_bkps": 2},
    "cpdf_mean":          {"mode": "cpdf", "family": "mean",
                           "algo": "kernelcpd", "min_size": 3, "n_bkps": 2},
    "cpdf_poly2":         {"mode": "cpdf", "family": "poly",
                           "family_kwargs": {"degree": 2},
                           "algo": "kernelcpd", "min_size": 3, "n_bkps": 2},
    # Faster variants if even KernelCPD is too slow on the full 38k set
    "cpdf_nwkr_gaussian_binseg": {"mode": "cpdf", "family": "nwkr_gaussian",
                                  "algo": "binseg", "min_size": 3, "n_bkps": 2},
}