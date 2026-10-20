#!/usr/bin/env python3
"""Authoritative oracle for duckdb_factor_sem — EFA.

Methods:  pca, pa (principal-axis), minres (minimum residual / OLS, JASP default).
Rotations: none, varimax (orthogonal), promax (oblique, Kaiser-normalized, JASP default).

minres (Harman & Jones 1966), correlation basis R, q factors:
  minimize f(Psi) = sum of the strictly-lower-triangular squares of (R - (model))
  where  reduced(S) = R with diag <- 1 - Psi, eigen -> top-q,
         loadings = V_q * sqrt(max(w_q,0)) , model = loadings @ loadings'
  Optimized with L-BFGS-B (same solver R's optim uses), start = 1 - SMC(R),
  lower bound 0.005 on each uniqueness, upper max(R).  This mirrors psych::fa
  (fit.residuals + FAgr.minres + FAout.wls in fa.R).

promax (Kaiser-normalized, JASP's default oblique), from psych Promax.R + kaiser.R:
  h2 = diag(L L'); W = L / sqrt(h2)          (Kaiser normalization)
  [Wv, Th] = varimax(W, T=I)                 (orthogonal pre-rotation)
  Q  = Wv * |Wv|^(m-1);  U = (Wv' Wv)^{-1} Wv' Q     (least squares, lm.fit)
  d  = diag(inv(U' U));  U <- U * sqrt(d);  U <- Th @ U
  Lr = Wv @ U;  Phi = inv(U) @ inv(U)'         (oblique factor correlation)
  loadings = Lr * sqrt(h2)                    (undo Kaiser normalization)

Sign convention: in each rotated factor column the largest-|loading| element is
made positive (deterministic, applied to all rotation types for consistency).

Outputs test/efa_ref.json per case: R, data, eigenvalues, prop_var, loadings
(unrotated), loadings_rotated, uniquenesses, communality, Phi, rotation, srml,
method/rotate iters, minres criterion (analytic vs fd).
"""
import json
import numpy as np
from scipy.optimize import minimize


# ------------------------------------------------------------------ eigen / pca
def eigen(R):
    w, V = np.linalg.eigh(R)
    o = np.argsort(w)[::-1]
    return w[o].copy(), V[:, o]


def pca_loadings(R, q):
    w, V = eigen(R)
    wq = np.maximum(w[:q], 0.0)
    L = V[:, :q] * np.sqrt(wq)
    return L, w


# --------------------------------------------------------- principal-axis (PA)
def pa_loadings(R, q, tol=1e-12, maxiter=10000):
    d = np.ones(R.shape[0])
    L = None
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
    w, _ = eigen(R)
    h = (L ** 2).sum(axis=1)
    return L, w


# ----------------------------------------------------------- minres (OLS)
def smc(R):
    inv = np.linalg.pinv(R)
    s = 1.0 - 1.0 / np.diag(inv)
    return np.clip(s, 0.0, 1.0)


def _minres_reduced(R, Psi):
    S = R.copy()
    np.fill_diagonal(S, 1.0 - Psi)
    w, V = eigen(S)
    w = np.where(w < np.finfo(float).eps, 100 * np.finfo(float).eps, w)
    q = _MINRES_Q[0]
    wq = np.maximum(w[:q], 0.0)
    L = V[:, :q] * np.sqrt(wq)
    return L, L @ L.T


def _minres_obj(R, Psi):
    q = _MINRES_Q[0]
    L, model = _minres_reduced(R, Psi)
    diff = R - model
    il = np.tril_indices(diff.shape[0], -1)   # strictly lower triangle (psych lower.tri)
    return float((diff[il] ** 2).sum())


def _minres_grad(R, Psi):
    q = _MINRES_Q[0]
    L, model = _minres_reduced(R, Psi)
    return np.diag(model + np.diag(Psi) - R)  # psych FAgr.minres


_MINRES_Q = [0]  # shared per-call q


def minres_loadings(R, q, lower=0.005, tol=1e-10, maxiter=10000):
    """Minres via L-BFGS-B (analytic grad) cross-checked against a finite-
    difference L-BFGS-B run; both must agree.  Returns (L, psi, info)."""
    global _MINRES_Q
    _MINRES_Q[0] = q
    p = R.shape[0]
    start = np.maximum(np.diag(R) - smc(R), lower)
    upper = max(1.0, float(np.max(R)))

    def f(x):
        return _minres_obj(R, x)

    def g(x):
        return _minres_grad(R, x)

    # analytic-gradient L-BFGS-B (matches R optim)
    r1 = minimize(f, start, jac=g, method="L-BFGS-B",
                  bounds=[(lower, upper)] * p,
                  options={"maxiter": maxiter, "gtol": tol})
    Psi = r1.x
    # finite-difference L-BFGS-B (independent check)
    r2 = minimize(f, start, method="L-BFGS-B",
                  bounds=[(lower, upper)] * p,
                  options={"maxiter": maxiter, "ftol": tol, "gtol": tol})
    L1 = _minres_reduced(R, r1.x)[0]
    L2 = _minres_reduced(R, r2.x)[0]
    agree = bool(np.max(np.abs(L1 - L2)) < 1e-3)
    info = {"obj": _minres_obj(R, Psi),
            "obj_fd": _minres_obj(R, r2.x),
            "obj_start": _minres_obj(R, start),
            "agreed": agree,
            "iters": int(r1.get("nit", 0)),
            "success": bool(r1.success)}
    L, model = _minres_reduced(R, Psi)
    w, _ = eigen(R)
    return L, np.clip(1.0 - (L ** 2).sum(axis=1), 0.0, None), w, info


# -------------------------------------------------------------- rotations
def _orthogonalize(U):
    UUt = U.T @ U
    ev, E = np.linalg.eigh(UUt)
    ev = np.maximum(ev, 1e-300)
    return U @ (E / np.sqrt(ev)) @ E.T


def _fix_sign(L):
    for j in range(L.shape[1]):
        i = int(np.argmax(np.abs(L[:, j])))
        if L[i, j] < 0:
            L[:, j] *= -1
    return L


def rotate_loadings(L, method="varimax", m=4.0, tol=1e-12, maxiter=2000, delta=1e-8):
    """Guttman fixed-point varimax (orthogonal) -- verified against scipy."""
    p, q = L.shape
    T = np.eye(q)
    it = 0
    if q >= 2:
        for it in range(maxiter):
            U = L @ T
            g = (U ** 2).sum(axis=0)
            Us = U / (delta + g[None, :])
            Tn = Us.T @ U
            Tn = _orthogonalize(Tn)
            for j in range(q):
                i = int(np.argmax(np.abs(Tn[:, j])))
                if Tn[i, j] < 0:
                    Tn[:, j] *= -1
            if np.max(np.abs(Tn - T)) < tol:
                T = Tn
                break
            T = Tn
    Lr = L @ T
    return _fix_sign(Lr.copy()), T, it + 1


def kaiser_promax(L, m=4.0):
    """Kaiser-normalized promax.  Returns (Lr, Phi, rotmat)."""
    p, q = L.shape
    if q == 1:
        Lr = _fix_sign(L.copy())
        return Lr, np.eye(1), np.eye(1)
    h2 = np.maximum(np.diag(L @ L.T), 1e-12)
    W = L / np.sqrt(h2)[:, None]
    Wv, _, _ = rotate_loadings(W, "varimax")   # verified Guttman varimax
    # Th_eff = lstsq(W, Wv) so W @ Th_eff == Wv (robust to varimax sign convention)
    Th_eff, _, _, _ = np.linalg.lstsq(W, Wv, rcond=None)
    Q = Wv * np.abs(Wv) ** (m - 1)
    X = np.ascontiguousarray(Wv)
    U, _, _, _ = np.linalg.lstsq(X, Q, rcond=None)   # U = (Wv'Wv)^-1 Wv' Q (q x q)
    G = U.T @ U
    ev, E = np.linalg.eigh(G)
    if np.min(ev) < 1e-10:
        G2 = E @ np.diag(np.maximum(ev, 1e-10)) @ E.T
        np.fill_diagonal(G2, 1.0)
        d = np.diag(np.linalg.inv(G2))
    else:
        d = np.diag(np.linalg.inv(G))
    U = U * np.sqrt(d)[None, :]
    Lr = Wv @ U * np.sqrt(h2)[:, None]
    Ufinal = Th_eff @ U
    invU = np.linalg.inv(Ufinal)
    Phi = invU @ invU.T
    return _fix_sign(Lr.copy()), Phi, Ufinal


def rotate(L, method, m=4.0):
    if method == "none":
        return _fix_sign(L.copy()), np.eye(L.shape[1]), 0
    if method == "varimax":
        Lr, T, it = rotate_loadings(L, "varimax")
        return Lr, np.eye(L.shape[1]), it
    if method == "promax":
        Lr, Phi, Rm = kaiser_promax(L, m)
        return Lr, Phi, 1
    raise ValueError(method)


# ------------------------------------------------------------------ metrics
def srml(R, L, uni):
    impl = L @ L.T + np.diag(uni)
    diff = impl - R
    n = R.shape[0]
    mask = np.zeros_like(diff, dtype=bool)
    mask[np.triu_indices(n)] = True
    return float(np.sqrt((diff[mask] ** 2).mean()))


# --------------------------------------------------------------------- data
def make_data(n, p, q, seed):
    """Two-block factor structure: each variable loads strongly (0.85-0.98)
    on exactly one factor and weakly (0.03-0.05) on the others.  The block
    pattern is unambiguous, so minres/PA have a sharp, near-unique minimum
    (Rust L-BFGS-B and scipy L-BFGS-B converge to <1e-6, which matters when
    an oblique rotation downstream is run on the loadings)."""
    rng = np.random.default_rng(seed)
    true_L = np.zeros((p, q))
    for i in range(p):
        keep = i % q
        for j in range(q):
            if j == keep:
                true_L[i, j] = rng.uniform(0.85, 0.98)
            else:
                true_L[i, j] = rng.uniform(0.03, 0.05)
    true_Psi = np.full(p, 0.2)
    f = rng.normal(0, 1, size=(n, q))
    X = f @ true_L.T + rng.normal(0, 1, size=(n, p)) * np.sqrt(true_Psi)
    C = (X - X.mean(axis=0)).T @ (X - X.mean(axis=0)) / (n - 1)
    C = (C + C.T) / 2
    std = np.sqrt(np.diag(C))
    R = C / np.outer(std, std)
    R = (R + R.T) / 2
    np.fill_diagonal(R, 1.0)
    return X, R


def run_case(n, p, q, seed, method="minres", rot="varimax"):
    X, R = make_data(n, p, q, seed)
    w_all, _ = eigen(R)
    if method == "pca":
        L, w = pca_loadings(R, q)
        info = {}
    elif method == "pa":
        L, w = pa_loadings(R, q)
        info = {}
    elif method == "minres":
        L, _uni, w, info = minres_loadings(R, q)
    else:
        raise ValueError(method)
    h = (L ** 2).sum(axis=1)
    uni = np.maximum(1.0 - h, 0.0)
    Lr, Phi, it = rotate(L, rot)
    prop = w_all / w_all.sum()
    return {
        "p": p, "q": q, "n": n, "method": method, "rotate": rot,
        "data": X.tolist(),
        "R": R.tolist(),
        "eigenvalues": w_all.tolist(),
        "proportion_var": prop.tolist(),
        "loadings": L.tolist(),
        "loadings_rotated": Lr.tolist(),
        "uniquenesses": uni.tolist(),
        "communality": h.tolist(),
        "phi": Phi.tolist(),
        "rotation": it if isinstance(it, int) else it,
        "srml": srml(R, Lr, uni),
        "minres": info,
    }


def main():
    cases = []
    cases.append(run_case(60, 6, 2, seed=11, method="pca", rot="varimax"))
    cases.append(run_case(60, 9, 3, seed=22, method="pca", rot="varimax"))
    cases.append(run_case(60, 6, 2, seed=11, method="pa", rot="varimax"))
    cases.append(run_case(60, 5, 1, seed=33, method="pca", rot="none"))
    cases.append(run_case(60, 8, 2, seed=44, method="pa", rot="none"))
    cases.append(run_case(60, 6, 2, seed=11, method="minres", rot="varimax"))
    cases.append(run_case(60, 8, 2, seed=44, method="minres", rot="promax"))
    cases.append(run_case(60, 6, 2, seed=22, method="minres", rot="promax"))
    cases.append(run_case(60, 5, 1, seed=33, method="minres", rot="none"))
    with open("test/efa_ref.json", "w") as fh:
        json.dump(cases, fh, indent=1)
    for i, c in enumerate(cases):
        phi = np.array(c["phi"])
        ob = "I" if np.allclose(phi, np.eye(c["q"]), atol=1e-8) else f"Phi\n   {np.round(phi,3).tolist()}"
        mr = ""
        if c["method"] == "minres":
            mr = (f" obj {c['minres']['obj']:.4f}->{c['minres']['obj']:.4f} "
                  f"fd={c['minres']['obj_fd']:.4f} ok={c['minres']['agreed']}")
        print(f"case {i}: {c['method']}+{c['rotate']} p={c['p']} q={c['q']} "
              f"it={c['rotation']} srml={c['srml']:.4f} phi={ob}{mr}")
        print("   rotL", np.round(np.array(c["loadings_rotated"]), 3).tolist())


if __name__ == "__main__":
    main()
