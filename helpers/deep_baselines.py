"""
Deep / OOD time-series anomaly detection baselines, adapted to the
interval-localisation contract used by the rest of this repo.

Every detector in this module exposes the same signature as the existing
baselines in ``benchmark_regressor_on_syth``::

    scan_row_deep(x, W, ...) -> (score: float, (a: int, b: int))

Two backends are supported:

``tsbad``   TSB-AD (Liu & Paparrizos, NeurIPS 2024 D&B).  Fully *unsupervised*
            and per-series: the detector is fit and applied to the same signal,
            so it slots directly into the existing per-row harness with no
            training corpus.  ``pip install TSB-AD``.

``deepod``  DeepOD (Xu et al.).  *Semi-supervised / one-class*: the model is fit
            once on a corpus of known-normal signals and then applied to every
            row.  This is the setting that actually matches the ALMA problem
            (38,650 negatives available as in-distribution training data) and is
            what "OOD detection" means here.  ``pip install deepod``.

Both backends return a per-timestamp anomaly *score vector*.  The repo's
evaluation is IoU over a single predicted *interval*, so the score vector is
converted to an interval by ``scores_to_interval`` below.  That conversion
deliberately mirrors the candidate-and-rescore strategy already used by
``scan_row_stumpy``: peaks in the detector's own score define candidate
locations, and the returned score is always ``_epidemic_score`` so that it is
directly comparable to tau_S across every method in the paper.

NOTE ON EVALUATION PROTOCOL
---------------------------
Do *not* score these detectors with point-adjusted F1.  Kim et al. (AAAI 2022)
showed PA inflates results so severely that a random score vector can look
state-of-the-art.  The IoU / tau_S protocol already used in this repo is the
right one and should be stated as such in the paper.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)

__all__ = [
    "ensure_numpy2_compat",
    "scores_to_interval",
    "set_normal_corpus",
    "load_normal_corpus_npy",
    "tsbad_available",
    "TSBAD_UNSUP_MODELS",
    "TSBAD_ZERO_SHOT",
    "scan_row_deep",
    "DeepModelStore",
    "MODEL_STORE",
    "TSBAD_MODELS",
    "DEEPOD_MODELS",
    "DEEP_METHODS",
]


# ---------------------------------------------------------------------------
# NumPy 2.0 compatibility shim
# ---------------------------------------------------------------------------
# NumPy 2.0 removed a set of long-deprecated aliases (np.Inf, np.NaN, np.float_,
# ...).  Several third-party detectors still use them and fail at runtime with
# "`np.Inf` was removed in the NumPy 2.0 release".  Each removed name was an
# exact synonym for a name that still exists, so restoring them is safe and
# changes no numerical behaviour.  Call this before importing TSB-AD / DeepOD.

_NUMPY2_ALIASES = {
    "Inf": "inf", "Infinity": "inf", "infty": "inf", "PINF": "inf",
    "NaN": "nan", "NAN": "nan",
    "float_": "float64", "complex_": "complex128",
    "unicode_": "str_", "bool8": "bool_",
    "int0": "intp", "uint0": "uintp",
    "round_": "round", "product": "prod", "cumproduct": "cumprod",
    "sometrue": "any", "alltrue": "all",
}

_numpy2_patched = False


def ensure_numpy2_compat(verbose: bool = False) -> List[str]:
    """Restore NumPy 1.x aliases removed in NumPy 2.0.  Idempotent.

    Returns the list of names that had to be restored (empty on NumPy 1.x).
    """
    global _numpy2_patched
    restored: List[str] = []
    if _numpy2_patched:
        return restored
    if int(np.__version__.split(".")[0]) < 2:
        _numpy2_patched = True
        return restored

    for missing, present in _NUMPY2_ALIASES.items():
        if not hasattr(np, missing) and hasattr(np, present):
            setattr(np, missing, getattr(np, present))
            restored.append(missing)
    # NINF / NZERO have no surviving synonym; they were plain float constants.
    if not hasattr(np, "NINF"):
        np.NINF = float("-inf"); restored.append("NINF")
    if not hasattr(np, "NZERO"):
        np.NZERO = -0.0; restored.append("NZERO")
    if not hasattr(np, "PZERO"):
        np.PZERO = 0.0; restored.append("PZERO")

    _numpy2_patched = True
    if restored and verbose:
        log.info("NumPy %s: restored removed aliases %s", np.__version__, restored)
    return restored


# ---------------------------------------------------------------------------
# Model catalogues
# ---------------------------------------------------------------------------

# Unsupervised, per-series.  Names must match TSB_AD.model_wrapper.
TSBAD_MODELS: Tuple[str, ...] = (
    "USAD",                 # ranks in the TSB-AD-U top-12; adversarial AE
    "CNN",                  # forecasting-based; closest in spirit to F_d
    "AnomalyTransformer",   # ICLR'22, the one reviewers will name
    "TranAD",               # VLDB'22, the other one
    "M2N2",                 # distribution-shift aware -> the "OOD" baseline
    "TimesNet",
    "OmniAnomaly",
    "Donut",
    "AutoEncoder",
    "LSTMAD",
    "FITS",
)

#: Unsupervised pool: no training array needed at all.
TSBAD_UNSUP_MODELS: Tuple[str, ...] = (
    "Chronos", "TimesFM", "MOMENT_ZS", "Lag_Llama",   # deep, zero-shot
    "Sub_PCA", "IForest", "MatrixProfile", "KMeansAD", "KShapeAD", "SR", "CBLOF", "EIF",
)

# One-class / semi-supervised, fit once on normals.  Names must match deepod.models.
DEEPOD_MODELS: Tuple[str, ...] = (
    "TranAD",
    "AnomalyTransformer",
    "COUTA",
    "NCAD",
    "TcnED",
    "DeepSVDDTS",
    "DeepIsolationForestTS",
)


# ---------------------------------------------------------------------------
# Score-vector -> interval
# ---------------------------------------------------------------------------

def _epidemic_score_local(x: np.ndarray, a: int, b: int) -> float:
    """Standalone copy of benchmark_regressor_on_syth._epidemic_score.

    Only used if the main module cannot be imported (e.g. unit tests).  The
    real implementation is preferred so that the two can never drift.
    """
    n = x.size
    if a > b or n == 0:
        return -np.inf
    mu_all = x.mean()
    sra = float(np.sum((x - mu_all) ** 2))
    if sra < 1e-18:
        return 0.0
    ins = x[a:b + 1]
    out = np.concatenate([x[:a], x[b + 1:]])
    sse_in = float(np.sum((ins - ins.mean()) ** 2)) if ins.size > 0 else 0.0
    sse_out = float(np.sum((out - out.mean()) ** 2)) if out.size > 0 else 0.0
    return 1.0 - (sse_in + sse_out) / sra


def _get_epidemic_score():
    """Bind to the canonical scorer, falling back to the local copy."""
    try:
        from benchmark_regressor_on_syth import _epidemic_score  # type: ignore
        return _epidemic_score
    except Exception:  # pragma: no cover - only in standalone use
        return _epidemic_score_local


def _moving_average(s: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return s
    kernel = np.ones(int(k), dtype=np.float64) / float(k)
    return np.convolve(s, kernel, mode="same")


def scores_to_interval(
    x: np.ndarray,
    s: np.ndarray,
    W: int,
    *,
    min_width: int = 2,
    top_k: int = 3,
    n_widths: int = 24,
    smooth: int = 0,
    select: str = "deep",
    refine: bool = True,
    refine_iters: int = 400,
    baseline: str = "mean",
) -> Tuple[float, Tuple[int, int]]:
    """Collapse a per-timestamp anomaly score vector into one interval [a, b].

    Mirrors ``scan_row_stumpy``: take the ``top_k`` highest-scoring
    non-overlapping locations as candidate centres, expand each over a ladder
    of widths up to ``W``, and pick the best window.

    Parameters
    ----------
    x
        The raw signal (used for the returned epidemic score).
    s
        Per-timestamp anomaly score from the deep detector, same length as x.
        Shorter vectors (some detectors drop the first ``seq_len`` points) are
        right-aligned and edge-padded.
    W
        Scan-window cap, as everywhere else in the repo.
    select
        ``"deep"`` (default)
                        choose the window maximising the detector's own excess
                        score mass, then *report* its epidemic score.  This is
                        the fair setting: the deep model picks its own interval
                        and is only converted to a comparable scale afterwards.
        ``"epidemic"``  choose the window maximising ``_epidemic_score``, i.e.
                        the same candidate-and-rescore rule currently applied to
                        STUMPY/BOCPD.  Included for consistency with the existing
                        baselines -- but note this rule selects with a *constant
                        mean* model, which the paper itself shows is a poor fit
                        for drifting backgrounds, so it can discard a correct
                        detection.  Report both if a reviewer asks.

    Returns
    -------
    (score, (a, b))
        ``score`` is always ``_epidemic_score(x, a, b)`` so that it is on the
        same [0, 1] scale as Phi(x) and directly comparable against tau_S.
        Returns ``(0.0, (0, -1))`` when no valid interval exists.
    """
    epidemic_score = _get_epidemic_score()

    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 10:
        return 0.0, (0, -1)

    s = np.asarray(s, dtype=np.float64).ravel()
    if s.size == 0:
        return 0.0, (0, -1)
    if s.size != n:
        # Right-align (detectors that use a look-back window emit fewer scores)
        pad = n - s.size
        if pad > 0:
            s = np.concatenate([np.full(pad, s[0], dtype=np.float64), s])
        else:
            s = s[-n:]
    s = np.nan_to_num(s, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    if not np.any(np.isfinite(s)):
        return 0.0, (0, -1)

    if smooth and smooth > 1:
        finite = np.isfinite(s)
        s_f = np.where(finite, s, np.nanmedian(s[finite]))
        s = _moving_average(s_f, int(smooth))

    W = int(max(min_width, min(W, n - 1)))
    widths = np.unique(
        np.clip(
            np.round(np.linspace(min_width, W, num=max(2, int(n_widths)))).astype(int),
            min_width, W,
        )
    )

    # Prefix sums of *excess* score above a robust baseline.  Excess mass, not
    # the window mean, is the right criterion: a mean is maximised by the
    # narrowest window sitting on the single highest point, whereas excess mass
    # keeps growing while the detector stays above baseline and therefore
    # recovers the full extent of the anomalous block.  This is the same
    # max-segment-sum logic the scan statistic itself uses.
    finite_mask = np.isfinite(s)
    if np.any(finite_mask):
        sf = s[finite_mask]
        if baseline == "mean":
            base = float(sf.mean())
        elif baseline == "median":
            base = float(np.median(sf))
        else:  # numeric quantile in [0, 1]
            base = float(np.quantile(sf, float(baseline)))
    else:
        base = 0.0
    # NOTE: the offset must make background points *negative*, otherwise adding
    # background to a window never lowers the sum and the widest admissible
    # window wins.  The mean does this even for sparse anomalies; the median
    # does not (background sits exactly at the median, contributing zero).
    excess = np.where(finite_mask, s - base, 0.0)
    csum = np.concatenate([[0.0], np.cumsum(excess)])

    def _deep_excess(a: int, b: int) -> float:
        return float(csum[b + 1] - csum[a])

    work = s.copy()
    best_sel = -np.inf
    best_ab: Tuple[int, int] = (0, -1)

    for _ in range(max(1, int(top_k))):
        if not np.any(np.isfinite(work)):
            break
        centre = int(np.argmax(work))
        # Suppress this peak (and its neighbourhood) before the next iteration
        half = max(min_width, W // 2)
        work[max(0, centre - half): centre + half + 1] = -np.inf

        for m in widths:
            m = int(m)
            # Slide a width-m window across the neighbourhood of the peak so the
            # candidate does not have to be perfectly centred.
            lo = max(0, centre - m + 1)
            hi = min(n - m, centre)
            if hi < lo:
                continue
            for a in range(lo, hi + 1):
                b = a + m - 1
                if b >= n:
                    continue
                sel = _deep_excess(a, b) if select == "deep" else epidemic_score(x, a, b)
                if sel > best_sel:
                    best_sel = sel
                    best_ab = (a, b)

    # ---- exact local refinement ------------------------------------------
    # The width ladder above is coarse, so the coarse winner is usually a few
    # channels off.  Greedily adjust each endpoint one channel at a time until
    # no single move improves the criterion.  This removes the discretisation
    # ceiling (oracle IoU goes to ~1.0) at negligible cost, since it touches
    # only O(refine_iters) windows rather than the full O(nW) grid.
    if best_ab[1] > best_ab[0] and refine:
        def _sel(a: int, b: int) -> float:
            if a < 0 or b >= n or b <= a or (b - a + 1) > W or (b - a + 1) < min_width:
                return -np.inf
            return _deep_excess(a, b) if select == "deep" else epidemic_score(x, a, b)

        a, b = best_ab
        cur = _sel(a, b)
        for _ in range(int(refine_iters)):
            moves = ((a - 1, b), (a + 1, b), (a, b - 1), (a, b + 1))
            vals = [_sel(*m) for m in moves]
            k = int(np.argmax(vals))
            if vals[k] > cur:
                cur = vals[k]
                a, b = moves[k]
            else:
                break
        best_ab = (a, b)

    a, b = best_ab
    if b <= a:
        return 0.0, (0, -1)
    final = float(epidemic_score(x, a, b))
    if not np.isfinite(final):
        return 0.0, (0, -1)
    return final, (int(a), int(b))


# ---------------------------------------------------------------------------
# Backend: TSB-AD (unsupervised, per-series)
# ---------------------------------------------------------------------------

# TSB-AD splits its catalogue into two dispatch pools.  Unsupervised detectors
# are fit and applied to the same series (perfect for the per-row harness).
# Semi-supervised detectors need a separate training array and are dispatched
# through a different function -- calling the wrong one yields the confusing
# error "Model function 'run_X' is not defined".
_TSBAD_UNSUP_FALLBACK = frozenset({
    "FFT", "SR", "NORMA", "Series2Graph", "Sub_IForest", "IForest", "LOF", "Sub_LOF",
    "POLY", "MatrixProfile", "Sub_PCA", "PCA", "HBOS", "Sub_HBOS", "KNN", "Sub_KNN",
    "KMeansAD", "KMeansAD_U", "KShapeAD", "COPOD", "CBLOF", "COF", "EIF", "RobustPCA",
    "MMPAD", "Lag_Llama", "TimesFM", "Chronos", "MOMENT_ZS", "TSPulse_ZS",
})
_TSBAD_SEMI_FALLBACK = frozenset({
    "Left_STAMPi", "SAND", "MCD", "Sub_MCD", "OCSVM", "Sub_OCSVM", "AutoEncoder",
    "CNN", "LSTMAD", "TranAD", "USAD", "OmniAnomaly", "PatchTST", "AnomalyTransformer",
    "TimesNet", "FITS", "Donut", "OFA", "MOMENT_FT", "M2N2", "TSPulse_FT", "xLSTMAD",
})

#: Deep foundation models that are *zero-shot* -- unsupervised pool, no training
#: array required, so they drop straight into the per-row harness.
TSBAD_ZERO_SHOT: Tuple[str, ...] = ("Chronos", "TimesFM", "MOMENT_ZS", "Lag_Llama")

# Module-level corpus of known-normal signals, used to build the training array
# for semi-supervised TSB-AD models.  Set once per process.
_NORMAL_CORPUS: Optional[np.ndarray] = None


def set_normal_corpus(signals: Sequence[np.ndarray], *, normalise: bool = True,
                      max_points: int = 200_000) -> int:
    """Register anomaly-free signals as the training array for TSB-AD models.

    Without this, semi-supervised detectors fall back to training on a prefix of
    the row being scored (TSB-AD's own protocol), which is unsafe here because a
    platforming anomaly may sit inside that prefix.

    Returns the number of timestamps registered.
    """
    global _NORMAL_CORPUS
    chunks = []
    total = 0
    for sig in signals:
        v = np.asarray(sig, dtype=np.float64).ravel()
        if v.size < 8:
            continue
        if normalise:
            sd = v.std()
            v = (v - v.mean()) / (sd if sd > 1e-12 else 1.0)
        chunks.append(v)
        total += v.size
        if total >= max_points:
            break
    if not chunks:
        raise ValueError("No usable signals in corpus")
    _NORMAL_CORPUS = np.concatenate(chunks).reshape(-1, 1)
    log.info("Registered normal corpus: %d timestamps from %d signals",
             _NORMAL_CORPUS.shape[0], len(chunks))
    return _NORMAL_CORPUS.shape[0]


def load_normal_corpus_npy(path: str) -> int:
    """Load a corpus previously saved with ``np.save`` (list of 1-D arrays)."""
    arr = np.load(path, allow_pickle=True)
    return set_normal_corpus(list(arr))


def tsbad_available() -> dict:
    """Report what the *installed* TSB-AD version actually supports."""
    ensure_numpy2_compat()
    from TSB_AD import model_wrapper as mw  # type: ignore

    uns = set(getattr(mw, "Unsupervise_AD_Pool", _TSBAD_UNSUP_FALLBACK))
    semi = set(getattr(mw, "Semisupervise_AD_Pool", _TSBAD_SEMI_FALLBACK))
    defined = {n[4:] for n in dir(mw) if n.startswith("run_")}
    import TSB_AD  # type: ignore

    version = getattr(TSB_AD, "__version__", None)
    if not version:
        try:
            from importlib.metadata import version as _dist_version
            version = _dist_version("TSB-AD")
        except Exception:
            version = "unknown"
    return {
        "unsupervised": sorted(uns & defined),
        "semisupervised": sorted(semi & defined),
        "declared_but_missing": sorted((uns | semi) - defined),
        "version": version,
        "package_path": os.path.dirname(getattr(TSB_AD, "__file__", "") or ""),
        "wrapper_path": getattr(mw, "__file__", "unknown"),
        "n_declared_unsup": len(uns),
        "n_declared_semi": len(semi),
    }


def _tsbad_scores(
    x: np.ndarray,
    model: str,
    seed: int = 42,
    train_frac: float = 0.3,
    train: str = "auto",
    **kw,
) -> np.ndarray:
    """Per-timestamp anomaly scores from a TSB-AD detector.

    Routes to ``run_Unsupervise_AD`` or ``run_Semisupervise_AD`` depending on
    which pool ``model`` belongs to in the *installed* version of TSB-AD.

    ``train``
        ``"auto"``    use the registered normal corpus if one exists, else prefix
        ``"corpus"``  require a corpus registered via ``set_normal_corpus``
        ``"prefix"``  train on the first ``train_frac`` of this row (TSB-AD's own
                      protocol; unsafe if the anomaly can fall in the prefix)
    """
    import random

    ensure_numpy2_compat()
    from TSB_AD import model_wrapper as mw  # type: ignore

    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass

    if not hasattr(mw, f"run_{model}"):
        avail = tsbad_available()
        raise RuntimeError(
            f"Installed TSB-AD has no run_{model}. "
            f"Unsupervised available: {avail['unsupervised']}. "
            f"Semi-supervised available: {avail['semisupervised']}. "
            f"Try `pip install -U TSB-AD` or install from the GitHub main branch."
        )

    uns = set(getattr(mw, "Unsupervise_AD_Pool", _TSBAD_UNSUP_FALLBACK))
    semi = set(getattr(mw, "Semisupervise_AD_Pool", _TSBAD_SEMI_FALLBACK))

    data = np.asarray(x, dtype=np.float64).reshape(-1, 1)

    if model in uns and model not in semi:
        out = mw.run_Unsupervise_AD(model, data, **kw)
    else:
        if train == "corpus" and _NORMAL_CORPUS is None:
            raise RuntimeError(
                f"{model} is semi-supervised and train='corpus', but no corpus is "
                f"registered. Call set_normal_corpus(...) or use train='prefix'."
            )
        if train in ("auto", "corpus") and _NORMAL_CORPUS is not None:
            data_train = _NORMAL_CORPUS
        else:
            cut = max(16, int(round(data.shape[0] * float(train_frac))))
            cut = min(cut, data.shape[0] - 1)
            data_train = data[:cut]
        out = mw.run_Semisupervise_AD(model, data_train, data, **kw)

    if isinstance(out, str):  # TSB-AD returns an error string rather than raising
        raise RuntimeError(f"TSB-AD {model} failed: {out}")
    return np.asarray(out, dtype=np.float64).ravel()


# ---------------------------------------------------------------------------
# Backend: DeepOD (one-class, fit once on normals)
# ---------------------------------------------------------------------------

@dataclass
class DeepModelStore:
    """Process-local cache of DeepOD models fit on known-normal signals.

    Fitting happens once per process.  Under multiprocessing each worker
    re-loads the pickled model from ``artifact_dir`` rather than re-fitting.
    """

    artifact_dir: str = field(
        default_factory=lambda: os.environ.get("RSS_DEEP_ARTIFACTS", "data/deep_models")
    )
    _cache: Dict[str, object] = field(default_factory=dict, repr=False)

    def path_for(self, model: str) -> str:
        return os.path.join(self.artifact_dir, f"deepod_{model}.joblib")

    def get(self, model: str):
        if model in self._cache:
            return self._cache[model]
        import joblib

        path = self.path_for(model)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No fitted DeepOD model at {path}. Run fit_deep_models.py first, "
                f"or set RSS_DEEP_ARTIFACTS to the artifact directory."
            )
        clf = joblib.load(path)
        self._cache[model] = clf
        return clf

    def fit(
        self,
        model: str,
        normals: Sequence[np.ndarray],
        *,
        seq_len: int = 100,
        epochs: int = 10,
        device: str = "cpu",
        seed: int = 42,
        normalise: bool = True,
        save: bool = True,
        **model_kw,
    ):
        """Fit ``model`` on a corpus of anomaly-free signals.

        Signals are z-normalised individually (so amplitude offsets between
        spectral windows do not dominate) and concatenated into one long
        series, which is the input shape DeepOD expects.
        """
        ensure_numpy2_compat()
        import deepod.models as dm  # type: ignore

        if not hasattr(dm, model):
            raise ValueError(
                f"Unknown DeepOD model {model!r}. Available: {DEEPOD_MODELS}"
            )
        cls = getattr(dm, model)

        chunks = []
        for sig in normals:
            v = np.asarray(sig, dtype=np.float64).ravel()
            if v.size < seq_len + 1:
                continue
            if normalise:
                sd = v.std()
                v = (v - v.mean()) / (sd if sd > 1e-12 else 1.0)
            chunks.append(v)
        if not chunks:
            raise ValueError("No training signals long enough for seq_len=%d" % seq_len)
        train = np.concatenate(chunks).reshape(-1, 1)
        log.info(
            "Fitting DeepOD %s on %d signals (%d timestamps)",
            model, len(chunks), train.shape[0],
        )

        clf = cls(
            seq_len=seq_len,
            epochs=epochs,
            device=device,
            random_state=seed,
            **model_kw,
        )
        t0 = time.perf_counter()
        clf.fit(train)
        log.info("  fit took %.1f s", time.perf_counter() - t0)

        self._cache[model] = clf
        if save:
            import joblib

            os.makedirs(self.artifact_dir, exist_ok=True)
            joblib.dump(clf, self.path_for(model))
            log.info("  saved -> %s", self.path_for(model))
        return clf


# Auto-load a normal corpus if the environment points at one.  This makes
# semi-supervised TSB-AD models work under score_experiment.py without editing
# the runner (each serial worker imports this module).
_ENV_CORPUS = os.environ.get("RSS_NORMAL_CORPUS")
if _ENV_CORPUS and os.path.exists(_ENV_CORPUS):
    try:
        load_normal_corpus_npy(_ENV_CORPUS)
    except Exception as _corpus_exc:  # pragma: no cover
        log.warning("Could not load RSS_NORMAL_CORPUS=%s: %s", _ENV_CORPUS, _corpus_exc)


MODEL_STORE = DeepModelStore()


def _deepod_scores(x: np.ndarray, model: str, normalise: bool = True, **_) -> np.ndarray:
    clf = MODEL_STORE.get(model)
    v = np.asarray(x, dtype=np.float64).ravel()
    if normalise:
        sd = v.std()
        v = (v - v.mean()) / (sd if sd > 1e-12 else 1.0)
    return np.asarray(clf.decision_function(v.reshape(-1, 1)), dtype=np.float64).ravel()


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

_BACKENDS = {
    "tsbad": _tsbad_scores,
    "deepod": _deepod_scores,
}


def scan_row_deep(
    x: np.ndarray,
    W: int,
    *,
    backend: str = "tsbad",
    model: str = "USAD",
    seed: int = 42,
    min_width: int = 2,
    top_k: int = 3,
    n_widths: int = 24,
    smooth: int = 0,
    select: str = "deep",
    refine: bool = True,
    refine_iters: int = 400,
    baseline: str = "mean",
    backend_kwargs: Optional[dict] = None,
) -> Tuple[float, Tuple[int, int]]:
    """Deep anomaly detector -> (epidemic score, interval), repo-standard contract."""
    if backend not in _BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}; expected one of {sorted(_BACKENDS)}")

    x = np.asarray(x, dtype=np.float64)
    if x.size < 10:
        return 0.0, (0, -1)

    try:
        s = _BACKENDS[backend](x, model, seed=seed, **(backend_kwargs or {}))
    except Exception as exc:
        log.warning("%s:%s failed on a row (n=%d): %s", backend, model, x.size, exc)
        return 0.0, (0, -1)

    return scores_to_interval(
        x, s, W,
        min_width=min_width,
        top_k=top_k,
        n_widths=n_widths,
        smooth=smooth,
        select=select,
        refine=refine,
        refine_iters=refine_iters,
        baseline=baseline,
    )


# ---------------------------------------------------------------------------
# Registry entries to merge into ALL_METHODS
# ---------------------------------------------------------------------------

def _entry(backend: str, model: str, **over) -> dict:
    cfg = {"mode": "deep", "backend": backend, "model": model, "seed": 42}
    cfg.update(over)
    return cfg


#: Merge this into ``ALL_METHODS`` in benchmark_regressor_on_syth.py.
DEEP_METHODS: Dict[str, dict] = {
    **{f"tsbad_{m.lower()}": _entry("tsbad", m) for m in TSBAD_MODELS},
    **{f"tsbad_{m.lower()}": _entry("tsbad", m) for m in TSBAD_UNSUP_MODELS},
    **{f"deepod_{m.lower()}": _entry("deepod", m) for m in DEEPOD_MODELS},
    # Ablation: same detector, window chosen by the constant-mean epidemic
    # score instead (the rule currently used for STUMPY/BOCPD candidates).
    "tsbad_usad_epsel": _entry("tsbad", "USAD", select="epidemic"),
    "tsbad_tranad_epsel": _entry("tsbad", "TranAD", select="epidemic"),
}

#: Suggested five for the paper (see also the runtime column argument).
PAPER_DEEP_METHODS: List[str] = [
    "tsbad_usad",
    "tsbad_cnn",
    "tsbad_anomalytransformer",
    "tsbad_tranad",
    "tsbad_m2n2",
]
