#!/usr/bin/env python
"""Register the NWKR-background change-point baseline in the harness.

Anchors on unique strings rather than line numbers, so it applies regardless of
what else has changed in benchmark_regressor_on_syth.py. Safe to run twice.

    python install_cpd_nwkr.py
    python install_cpd_nwkr.py --revert
"""

from __future__ import annotations

import argparse
import shutil
import sys

TARGET = "benchmark_regressor_on_syth.py"

REGISTRY_ANCHOR = '# Methods that are known to be very slow and need a hard timeout guard'
REGISTRY_BLOCK = '''# --- NWKR-background change-point baseline (reviewer head-to-head) ----------
try:
    from helpers.cpd_nwkr import CPD_METHODS as _CPD_METHODS
    ALL_METHODS.update(_CPD_METHODS)
except Exception as _cpd_exc:  # pragma: no cover
    logging.getLogger(__name__).debug("cpd baselines unavailable: %s", _cpd_exc)


'''

DISPATCH_ANCHOR = '    raise ValueError(f"Unknown mode {mode!r}")'
DISPATCH_BLOCK = '''    if mode == "cpd":
        from helpers.cpd_nwkr import scan_row_cpd_family
        fkw = dict(cfg.get("family_kwargs", {}))
        if str(cfg.get("family", "")).startswith("nwkr"):
            fkw.setdefault("w", float(cfg.get("w", W)))
        return scan_row_cpd_family(
            x, W,
            family        = cfg.get("family", "nwkr_gaussian"),
            family_kwargs = fkw,
            min_size      = int(cfg.get("min_size", 5)),
            n_bkps        = int(cfg.get("n_bkps", 2)),
            width_cap     = cfg.get("width_cap", None),
            select        = cfg.get("select", "native"),
        )

'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=TARGET)
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    try:
        src = open(args.target).read()
    except FileNotFoundError:
        raise SystemExit(f"{args.target} not found — run from the repo root.")

    if args.revert:
        out = src.replace(REGISTRY_BLOCK, "").replace(DISPATCH_BLOCK, "")
        if out == src:
            print("Nothing to revert.")
            return
        shutil.copy(args.target, args.target + ".bak")
        open(args.target, "w").write(out)
        print(f"Reverted. Backup at {args.target}.bak")
        return

    changed = []

    if "from helpers.cpd_nwkr import CPD_METHODS" in src:
        print("  registry block: already present, skipping")
    elif src.count(REGISTRY_ANCHOR) == 1:
        src = src.replace(REGISTRY_ANCHOR, REGISTRY_BLOCK + REGISTRY_ANCHOR)
        changed.append("registry block")
    else:
        raise SystemExit(
            f"Could not find a unique anchor for the registry block "
            f"(found {src.count(REGISTRY_ANCHOR)} matches). "
            f"Add REGISTRY_BLOCK by hand just above the line:\n  {REGISTRY_ANCHOR}"
        )

    if 'if mode == "cpd":' in src:
        print("  dispatch branch: already present, skipping")
    elif src.count(DISPATCH_ANCHOR) == 1:
        src = src.replace(DISPATCH_ANCHOR, DISPATCH_BLOCK + DISPATCH_ANCHOR)
        changed.append("dispatch branch")
    else:
        raise SystemExit(
            f"Could not find a unique anchor for the dispatch branch "
            f"(found {src.count(DISPATCH_ANCHOR)} matches). "
            f"Add DISPATCH_BLOCK by hand just above the final "
            f'raise ValueError("Unknown mode ...") in _run_one().'
        )

    if not changed:
        print("Already installed — nothing to do.")
        return

    shutil.copy(args.target, args.target + ".bak")
    open(args.target, "w").write(src)
    print(f"Installed: {', '.join(changed)}")
    print(f"Backup at {args.target}.bak")

    import py_compile
    try:
        py_compile.compile(args.target, doraise=True)
        print("Syntax OK")
    except Exception as exc:
        raise SystemExit(f"Syntax error after patching: {exc}\n"
                         f"Restore with: cp {args.target}.bak {args.target}")

    print("\nVerify with:")
    print('  python -c "import benchmark_regressor_on_syth as B; '
          "print([k for k in B.ALL_METHODS if k.startswith('cpd_')])\"")


if __name__ == "__main__":
    main()
