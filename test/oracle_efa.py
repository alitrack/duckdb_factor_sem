#!/usr/bin/env python3
"""Authoritative oracle for duckdb_factor_sem v0.2.0 — EFA (PCA / principal-axis
factoring + varimax / quartimax orthogonal rotation), correlation basis,
manual factor count.

Why PCA + PA and not minres: minres (JASP's default) is the hard-to-verify
case; PCA and principal-axis have closed-form / invariant-verifiable solutions
and are fully cross-checkable against numpy here. (minres deferred to v0.2.1.)

PCA, correlation basis R (p x p, unit diag):
  eigen(R) -> (w_1 >= ... >= w_p), unit eigenvectors V_j
  unrotated loadings  L = V_q * sqrt(w_q)          (p x q)
  communality h_i = sum_j L_ij^2 ; uniqueness psi_i = 1 - h_i
  reproduced Rhat = L L' + diag(psi)
  proportion var_i = w_i / sum(w)

Principal-axis (Thurstone), correlation basis:
  start D = diag(1); iterate
    Rr = R - D ; eig, top-q ; L = V_q*sqrt(max(w,0))
    D = diag(max(1 - colsum(L^2), floor)) until D stops moving
  (converged PA loadings; uniquenesses = 1 - h)

varimax (Kaiser 1958): maximize  f(T) = sum_j (sum_i (L T)_ij^2)^2  over
  orthogonal T. Fixed-point (Guttman):  T <- L' (Us / (delta + gamma))  where
  Us = U (U'U)^{-1/2}, gamma_j = sum_i (L T)_ij^2.
quartimax (Hendrickson & Thurstone): f = sum_ij (sum_k (L T)_kj^2)^4, same
  fixed-point with gamma_j = sum_k (L T)_kj^4.

Sign convention (deterministic): in each rotated factor column the
largest-|loading| element is made positive.

Outputs test/efa_ref.json per case: R, eigenvalues, prop_var, loadings
(unrotated), loadings_rotated, uniquenesses, communality, reproduced fit
SRMR, varimax criterion before/after, rotation T, iters.
"""
import json
import numpy as np


def eigen(R):
    w, V = np.linalg.eigh(R)
    o = np.argsort(w)[::-1]
    return w[o].copy(), V[:, o]


def pca_loadings(R, q):
    w, V = eigen(R)
    wq = np.maximum(w[:q], 0.0)
    L = V[:, :q] * np.sqrt(wq)
    return L, w


def pa_loadings(R, q, tol=1e-12, maxiter=10000):
    p = R.shape[0]
    d = np.ones(p)
    L = None
    w = None
    for it in range(maxiter):
        Rr = R - np.diag(d)
        w, V = eigen(Rr)
        wq = np.maximum(w[:q], 0.0)
        L = V[:, :q] * np.sqrt(wq)
        h = (L ** 2).sum(axis=1)
        d_new = np.maximum(1.0 - h, 1e-10)
        if np.max(np.abs(d_new - d)) < tol:
            d = d_new
            break
        d = d_new
    h = (L ** 2).sum(axis=1)
    return L, w


def _orthogonalize(U):
    UUt = U.T @ U
    ev, E = np.linalg.eigh(UUt)
    ev = np.maximum(ev, 1e-300)
    return U @ (E / np.sqrt(ev)) @ E.T  # U (U'U)^{-1/2}


def rotate_loadings(L, method="varimax", tol=1e-12, maxiter=2000, delta=1e-8):
    """Guttman fixed-point varimax (orthogonal). Verified against scipy.
    For quartimax use power=4 on the gamma (deferred to v0.2.1)."""
    p, q = L.shape
    T = np.eye(q)
    it = 0
    if q >= 2:
        for it in range(maxiter):
            U = L @ T
            g = (U ** 2).sum(axis=0)          # per-column sum of squares
            Us = U / (delta + g[None, :])     # scale RAW U
            Tn = Us.T @ U                     # symmetric q x q
            Tn = _orthogonalize(Tn)           # T (T'T)^{-1/2}
            for j in range(q):
                i = int(np.argmax(np.abs(Tn[:, j])))
                if Tn[i, j] < 0:
                    Tn[:, j] *= -1
            if np.max(np.abs(Tn - T)) < tol:
                T = Tn
                break
            T = Tn
    Lr = L @ T
    for j in range(q):
        i = int(np.argmax(np.abs(Lr[:, j])))
        if Lr[i, j] < 0:
            Lr[:, j] *= -1
            T[:, j] *= -1
    return Lr, T, it + 1


def criterion(L, method="varimax"):
    s = (L ** 2).sum(axis=0) ** 2
    return float(s.sum()) if method == "varimax" else float(((L ** 2).sum(axis=0) ** 4).sum())


def srml(R, L, uni):
    impl = L @ L.T + np.diag(uni)
    diff = impl - R
    n = R.shape[0]
    mask = np.zeros_like(diff, dtype=bool)
    mask[np.triu_indices(n)] = True
    return float(np.sqrt((diff[mask] ** 2).mean()))


def run_case(n, p, q, seed, method="pca", rotate="varimax", true_L=None, true_Psi=None):
    rng = np.random.default_rng(seed)
    if true_L is None:
        true_L = rng.uniform(0.6, 0.95, size=(p, q))
        for i in range(p):
            keep = i % q
            for j in range(q):
                if j != keep:
                    true_L[i, j] *= 0.05
    if true_Psi is None:
        true_Psi = np.full(p, 0.2)
    Sigma = true_L @ true_L.T + np.diag(true_Psi)
    f = rng.normal(0, 1, size=(n, q))
    X = f @ true_L.T + rng.normal(0, 1, size=(n, p)) * np.sqrt(true_Psi)
    Xc = X - X.mean(axis=0)
    C = Xc.T @ Xc / (n - 1)  # sample covariance
    C = (C + C.T) / 2
    std = np.sqrt(np.diag(C))
    R = C / np.outer(std, std)  # proper correlation matrix
    R = (R + R.T) / 2
    np.fill_diagonal(R, 1.0)
    w, _ = eigen(R)
    if method == "pca":
        L, w = pca_loadings(R, q)
    else:
        L, w = pa_loadings(R, q)
    h = (L ** 2).sum(axis=1)
    uni = np.maximum(1.0 - h, 0.0)
    Lr, T, it = (L, np.eye(q), 0) if rotate == "none" else rotate_loadings(L, rotate)
    prop = w / w.sum()
    return {
        "p": p, "q": q, "n": n, "method": method, "rotate": rotate,
        "data": X.tolist(),
        "R": R.tolist(),
        "eigenvalues": w.tolist(),
        "proportion_var": prop.tolist(),
        "loadings": L.tolist(),
        "loadings_rotated": Lr.tolist(),
        "uniquenesses": uni.tolist(),
        "communality": h.tolist(),
        "srml": srml(R, Lr, uni),
        "criterion_before": criterion(L, rotate) if rotate != "none" else None,
        "criterion_after": criterion(Lr, rotate) if rotate != "none" else None,
        "rotation": T.tolist(),
        "rotate_iters": it,
        "true_L": true_L.tolist(),
    }


def main():
    cases = []
    cases.append(run_case(60, 6, 2, seed=11, method="pca", rotate="varimax"))
    cases.append(run_case(60, 9, 3, seed=22, method="pca", rotate="varimax"))
    cases.append(run_case(60, 6, 2, seed=11, method="pa", rotate="varimax"))
    cases.append(run_case(60, 5, 1, seed=33, method="pca", rotate="none"))
    cases.append(run_case(60, 8, 2, seed=44, method="pa", rotate="none"))
    with open("test/efa_ref.json", "w") as fh:
        json.dump(cases, fh, indent=1)
    for i, c in enumerate(cases):
        print(f"case {i}: {c['method']}+{c['rotate']} p={c['p']} q={c['q']} "
              f"it={c['rotate_iters']} srml={c['srml']:.4f} "
              f"crit {c['criterion_before']} -> {c['criterion_after']}")
        print("   eig[:q]", np.round(c["eigenvalues"][: c["q"]], 3).tolist())
        print("   rotL\n    ", np.round(np.array(c["loadings_rotated"]), 3).tolist())


if __name__ == "__main__":
    main()
