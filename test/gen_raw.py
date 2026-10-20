#!/usr/bin/env python3
"""Generate test/raw40.txt — a fixed 1-factor dataset (n=40, p=4) used by the
cfa_fit raw-data smoke test in test/test_factor.sql. Deterministic (seed 42).

n=40 is large enough for the FIML EM to converge; smaller n (e.g. 12) leaves
the sample covariance near-singular and cfa_fit returns NULL.

Usage: /path/to/python gen_raw.py > test/raw40.txt
"""
import sys

try:
    import numpy as np
except ImportError:  # fallback: pure-python PRNG (matches numpy only conceptually)
    raise SystemExit("numpy required")

rng = np.random.default_rng(42)
n, p, q = 40, 4, 1
L = np.array([[0.9], [0.75], [0.6], [0.5]])
Psi = np.array([0.3, 0.3, 0.3, 0.3])
f = rng.normal(0, 1, (n, q))
X = f @ L.T + rng.normal(0, 1, (n, p)) * np.sqrt(Psi)
rows = [" ".join(f"{x:.4f}" for x in row) for row in X]
sys.stdout.write(" | ".join(rows) + "\n")
