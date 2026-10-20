#!/usr/bin/env python3
"""Authoritative oracle for duckdb_factor_sem v0.1.0 (CFA, FIML/ML).

Model: X (n x p) = f Lambda^T + e, f ~ N(0, I_q) [q uncorrelated standard
factors], e ~ N(0, diag(Psi)).  Sigma = Lambda Lambda^T + diag(Psi), Phi = I.

The ML objective depends only on the sample covariance S = (1/n) sum x x^T
(centered).  Complete-data (FIML) log-likelihood:
    LL = -n/2 [ p ln2pi + ln|Sigma| + tr(Sigma^-1 S) ]

Reference = scipy L-BFGS-B minimizing -LL (INDEPENDENT of the Rust EM), run
to tight tolerance => the MLE.  We ALSO run a numpy covariance-EM that uses
the SAME update as the Rust code, to give the Rust test a cross-language
1e-6 anchor:  |LL_em - LL_mle| and |Lambda_em - Lambda_mle|.

We store per case: S (sample cov), S-derived saturated LL inputs, the MLE
(Lambda_ref, Psi_ref) and fit indices.  The Rust unit test recomputes
Sigma_hat = L L^T + diag(psi), re-derives chi2/df/AIC/BIC/RMSEA/CFI/TLI and
compares against these stored values.

Dumps factor_ref.json.
"""
import json
import numpy as np
from scipy.optimize import minimize

def logdet_sym(S):
    s, lg = np.linalg.slogdet(S)
    if s <= 0:
        return -np.inf
    return lg

def fiml_ll(Lam, Psi, S, n):
    p = S.shape[0]
    Sigma = Lam @ Lam.T + np.diag(Psi)
    ev = np.linalg.eigvalsh(Sigma)
    if ev.min() <= 1e-9:
        return -1e18
    Sigma_inv = np.linalg.inv(Sigma)
    return -0.5 * n * (p * np.log(2*np.pi) + logdet_sym(Sigma) + np.trace(Sigma_inv @ S))

def saturated_ll(S, n):
    p = S.shape[0]
    return -0.5 * n * (p * np.log(2*np.pi) + logdet_sym(S) + p)

def pca_init(S, p, q):
    w, v = np.linalg.eigh(S)
    idx = np.argsort(w)[::-1][:q]
    vq = v[:, idx]
    Lam = vq * np.sqrt(np.clip(w[idx], 1e-6, None))
    resid = np.diag(S) - np.sum(Lam**2, axis=1)
    return Lam, np.clip(resid, 1e-4, None)

def cov_em(S, n, Lam0, Psi0, iters=8000, tol=1e-13):
    """FIML covariance EM for ML factor analysis (Phi=I) — mirrors math.rs.
    E[ff^T|x] = I - Lam' Sig^-1 Lam + Lam' Sig^-1 x x' Sig^-1 Lam"""
    p = S.shape[0]; q = Lam0.shape[1]
    Iq = np.eye(q)
    Lam = Lam0.copy(); Psi = Psi0.copy()
    for it in range(iters):
        Sig = Lam @ Lam.T + np.diag(Psi)
        Sig_inv = np.linalg.inv(Sig)
        R = Sig_inv @ Lam                 # p x q
        Q = Sig_inv @ S @ Sig_inv         # p x p  (1/n sum S_i S_i')
        B = Lam.T @ Q @ Lam + Iq - Lam.T @ R   # E[ff^T]  (q x q)
        Lbar = S @ R                       # p x q  E[xf^T]
        B_inv = np.linalg.inv(B)
        Lam_new = Lbar @ B_inv
        M = Lbar @ B_inv @ Lbar.T          # = Lam_new B Lam_new'
        Psi_new = np.clip(np.diag(S) - np.diag(M), 1e-9, None)
        if np.max(np.abs(Psi_new - Psi)) < tol and np.max(np.abs(Lam_new - Lam)) < tol:
            Lam, Psi = Lam_new, Psi_new
            break
        Lam, Psi = Lam_new, Psi_new
    return Lam, Psi, it + 1

def indep_ll(S, n):
    """Independence model: Sigma = diag(s_jj)."""
    p = S.shape[0]
    D = np.diag(np.diag(S))
    return -0.5 * n * (p * np.log(2*np.pi) + logdet_sym(D) + np.trace(np.linalg.inv(D) @ S))

def fit_indices(Lam, Psi, S, n, p, q):
    llm = fiml_ll(Lam, Psi, S, n)
    lls = saturated_ll(S, n)
    chi2 = -2.0 * (llm - lls)
    df = p * (p + 1) // 2 - (p * q + p)
    aic = chi2 - 2.0 * df
    bic = chi2 + np.log(n) * df
    rmsea = float(np.sqrt(max(0.0, (chi2 / df - 1.0) / (n - 1))) if df > 0 else 0.0)
    # baseline = independence model (p uncorrelated vars)
    chi2_base = -2.0 * (indep_ll(S, n) - lls)
    cfi = 1.0 - max(0.0, chi2 - chi2_base) / max(1e-12, chi2_base) if chi2_base > 0 else 1.0
    tli = (chi2_base - chi2) / chi2_base if (chi2_base > 0) else 1.0
    return dict(ll_model=float(llm), ll_saturated=float(lls), chi2=float(chi2),
                df=int(df), aic=float(aic), bic=float(bic), rmsea=rmsea,
                cfi=float(cfi), tli=float(tli))

def fit_case(n, p, q, seed, true_Lam, true_Psi, maxiter=8000):
    rng = np.random.default_rng(seed)
    f = rng.normal(0, 1, size=(n, q))
    e = rng.normal(0, 1, size=(n, p)) * np.sqrt(true_Psi)
    X = f @ true_Lam.T + e
    X = X - X.mean(axis=0)
    S = X.T @ X / n
    L0, Ps0 = pca_init(S, p, q)
    def obj(z):
        Lam = z[:p*q].reshape(p, q)
        Psi = np.exp(z[p*q:])
        return -fiml_ll(Lam, Psi, S, n)
    # multi-start: PCA init + a few random perturbations; keep best LL
    starts = [L0]
    for k in range(4):
        starts.append(L0 * (1.0 + 0.15 * rng.normal(0, 1, size=L0.shape)))
    best_z = None; best_ll = -np.inf
    for st in starts:
        z0 = np.concatenate([st.ravel(), np.log(np.clip(Ps0, 1e-4, None))])
        res = minimize(obj, z0, method='L-BFGS-B',
                       options=dict(maxiter=maxiter, ftol=1e-16, gtol=1e-13))
        ll = fiml_ll(res.x[:p*q].reshape(p, q), np.exp(res.x[p*q:]), S, n)
        if ll > best_ll:
            best_ll = ll; best_z = res.x
    Lam_ref = best_z[:p*q].reshape(p, q)
    Psi_ref = np.exp(best_z[p*q:])
    ind = fit_indices(Lam_ref, Psi_ref, S, n, p, q)
    Lam_em, Psi_em, n_it = cov_em(S, n, L0, Ps0)
    ll_em = fiml_ll(Lam_em, Psi_em, S, n)
    return dict(p=p, q=q, n=int(n),
                S=S.tolist(),
                lambda_ref=Lam_ref.tolist(), psi_ref=Psi_ref.tolist(),
                lambda_em=Lam_em.tolist(), psi_em=Psi_em.tolist(),
                em_iters=int(n_it), ll_em=float(ll_em),
                **ind)

def main():
    cases = []
    cases.append(fit_case(400, 4, 1, 11,
                          np.array([[0.9],[0.7],[0.5],[0.6]]),
                          np.array([0.25, 0.35, 0.45, 0.30])))
    cases.append(fit_case(200, 3, 1, 22,
                          np.array([[0.8],[0.6],[0.7]]),
                          np.array([0.4, 0.45, 0.35])))
    cases.append(fit_case(600, 6, 2, 33,
                          np.array([[0.9,0.0],[0.8,0.1],[0.1,0.85],[0.0,0.75],[0.6,0.4],[0.3,0.6]]),
                          np.array([0.25,0.30,0.30,0.25,0.45,0.40])))
    with open("/mnt/d/wsl2/duckdb_factor_sem/test/factor_ref.json", "w") as fh:
        json.dump(cases, fh, indent=2)
    for c in cases:
        print(f"p={c['p']} q={c['q']} n={c['n']} df={c['df']} chi2={c['chi2']:.4f} "
              f"CFI={c['cfi']:.5f} TLI={c['tli']:.5f} RMSEA={c['rmsea']:.5f} "
              f"em_iters={c['em_iters']}")
        le = np.array(c['lambda_em']); lr = np.array(c['lambda_ref'])
        dlam = np.abs(le - lr).max()
        # allow sign flip per factor column (factor rotation)
        for fc in range(c['q']):
            if abs(abs(lr[0, fc]) - abs(le[0, fc])) < 0.2 and le[0, fc]*lr[0, fc] < 0:
                le[:, fc] = -le[:, fc]
        dlam = min(dlam, np.abs(le - lr).max())
        print(f"   |LL_em-LL_mle|={abs(c['ll_em']-c['ll_model']):.2e}  "
              f"|Lam_em-Lam_mle|_max={dlam:.2e}  LL_mle={c['ll_model']:.4f}")
    print(f"wrote {len(cases)} cases")

if __name__ == "__main__":
    main()
