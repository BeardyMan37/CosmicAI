"""Self-tests for helpers/deep_baselines.py.

Runs without TSB-AD / DeepOD installed: the backends are mocked so that only
the score-vector -> interval logic is exercised.

    python -m pytest tests/test_deep_baselines.py -q
    python tests/test_deep_baselines.py          # also works standalone
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers import deep_baselines as db  # noqa: E402


def _ar2_signal(n=500, seed=0):
    """Slowly drifting AR(2) background matching the paper's synthetic setup."""
    rng = np.random.default_rng(seed)
    phi1, phi2 = 1.985, -0.985056
    x = np.zeros(n)
    x[0], x[1] = rng.normal(scale=0.1), rng.normal(scale=0.1)
    for i in range(2, n):
        x[i] = phi1 * x[i - 1] + phi2 * x[i - 2] + rng.normal(scale=0.01)
    x = x / (x.std() + 1e-12)
    return x + rng.normal(scale=0.05, size=n)


def _plant(x, a, b, snr=2.5):
    y = x.copy()
    y[a:b + 1] -= snr * x.std()
    return y


def test_recovers_planted_interval():
    n, a, b = 500, 220, 269
    x = _plant(_ar2_signal(n), a, b)

    # Mock detector: elevated score inside the anomaly, noisy elsewhere
    rng = np.random.default_rng(1)
    s = rng.normal(scale=0.1, size=n)
    s[a:b + 1] += 1.0

    score, (pa, pb) = db.scores_to_interval(x, s, W=60)
    inter = max(0, min(b, pb) - max(a, pa) + 1)
    union = max(b, pb) - min(a, pa) + 1
    iou = inter / union
    assert iou > 0.75, f"IoU too low: {iou:.3f} for ({pa},{pb}) vs ({a},{b})"
    assert 0.0 <= score <= 1.0, score
    print(f"  recovered ({pa},{pb}) vs truth ({a},{b}) IoU={iou:.3f} score={score:.4f}")


def test_epidemic_selection_mode_runs():
    n, a, b = 500, 100, 149
    x = _plant(_ar2_signal(n, seed=3), a, b)
    rng = np.random.default_rng(2)
    s = rng.normal(scale=0.1, size=n)
    s[a:b + 1] += 1.0

    score, (pa, pb) = db.scores_to_interval(x, s, W=60, select="epidemic")
    inter = max(0, min(b, pb) - max(a, pa) + 1)
    union = max(b, pb) - min(a, pa) + 1
    print(f"  select='epidemic' -> ({pa},{pb}) IoU={inter/union:.3f} score={score:.4f}")
    print("    (may be worse than select='deep': constant-mean selection on a")
    print("     drifting AR(2) background can discard a correct detection)")


def test_handles_short_score_vector():
    """Detectors with a look-back window emit fewer scores than timestamps."""
    n, a, b = 400, 200, 239
    x = _plant(_ar2_signal(n, seed=5), a, b)
    s_full = np.zeros(n)
    s_full[a:b + 1] = 1.0
    s_short = s_full[100:]  # simulate seq_len=100 truncation

    score, (pa, pb) = db.scores_to_interval(x, s_short, W=60)
    assert pb > pa
    assert abs(pa - a) < 25, (pa, a)
    print(f"  short vector ({s_short.size}/{n}) -> ({pa},{pb}) score={score:.4f}")


def test_degenerate_inputs():
    assert db.scores_to_interval(np.zeros(5), np.zeros(5), W=3) == (0.0, (0, -1))
    x = _ar2_signal(200)
    assert db.scores_to_interval(x, np.array([]), W=30) == (0.0, (0, -1))
    assert db.scores_to_interval(x, np.full(200, np.nan), W=30) == (0.0, (0, -1))
    flat = np.ones(200)
    score, ab = db.scores_to_interval(flat, np.ones(200), W=30)
    assert 0.0 <= score <= 1.0
    print("  degenerate inputs handled")


def test_scan_row_deep_failure_is_soft():
    """A missing backend must degrade to (0.0, (0,-1)), never raise."""
    x = _ar2_signal(300)
    score, ab = db.scan_row_deep(x, W=40, backend="deepod", model="TranAD")
    assert score == 0.0 and ab == (0, -1)
    print("  missing artifact -> soft failure OK")


def test_registry_shape():
    for key, cfg in db.DEEP_METHODS.items():
        assert cfg["mode"] == "deep"
        assert cfg["backend"] in ("tsbad", "deepod")
        assert isinstance(cfg["model"], str)
    for key in db.PAPER_DEEP_METHODS:
        assert key in db.DEEP_METHODS, key
    print(f"  registry OK ({len(db.DEEP_METHODS)} methods)")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"{name}:")
            fn()
    print("\nall tests passed")
