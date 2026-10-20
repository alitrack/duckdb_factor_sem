//! CFA (confirmatory factor analysis) core — FIML / ML fit via EM.
//!
//! Model (JASP Factor / lavaan / Mplus ML): X_i = Λ f_i + e_i,
//!     f_i ~ N(0, I_q)   (uncorrelated standard factors, Φ = I)
//!     e_i ~ N(0, diag(Psi))
//!     Sigma = Lambda Lambda^T + diag(Psi)
//!
//! Objective (FIML complete-data, identical to the likelihood of the sample
//! covariance S):  LL = -n/2 [ p ln 2π + ln|Sigma| + tr(Sigma^-1 S) ]
//!
//! Fit = MLE via Olsen et al. (2002) EM:
//!   E-step  m_i = Lambda^T S_i  where S_i = Sigma^-1 (X_i - mean),  fhat = m_i
//!   M-step  B   = (1/n) Σ m_i m_i^T          (q x q)
//!           Lbar = (1/n) Σ X_i m_i^T          (p x q)
//!           Lnew = Lbar B^-1                  (Φ = I)
//!           Psi_new_j = (1/n) Σ (X_ij - (Lnew m_i)_j)^2
//!
//! Reference (test/oracle_factor.py): scipy L-BFGS-B on the same objective,
//! an INDEPENDENT optimizer. The two must agree to ~1e-6 in LL and in |Λ|,
//! |ψ| (factor loadings are sign/rotation-invariant up to sign, so tests
//! compare absolute values).
//!
//! Fit indices (p items, q factors, n observations):
//!   df    = p(p+1)/2 - (pq + p)
//!   chi2  = -2 (LL_model - LL_saturated)
//!   AIC   = chi2 - 2 df            BIC = chi2 + ln n * df
//!   RMSEA = sqrt(max(0, (chi2/df - 1) / (n-1)))
//!   CFI   = 1 - max(0, chi2 - chi_min)/max(1, 0 - chi_min)
//!   TLI   = (chi_min - chi2)/chi_min
//!   where chi_min = (df-1)^2/df, LL_saturated = -n/2 (p ln 2π + ln|S| + p)

/// ln(2π) = 1.8378770664093453
pub const LOG_2PI: f64 = 1.8378770664093453;

// ─── small linear algebra (dense, no external deps) ─────────────────────

/// Invert a square matrix via Gauss-Jordan with partial pivoting.
/// Returns None on (near-)singularity.
pub fn invert(a: &[f64], n: usize) -> Option<Vec<f64>> {
    if n == 0 {
        return Some(vec![]);
    }
    let w = 2 * n; // augmented row width [A | I]
    // aug = [A | I]
    let mut aug = vec![0.0f64; n * w];
    for i in 0..n {
        for j in 0..n {
            aug[i * w + j] = a[i * n + j];
        }
        aug[i * w + (n + i)] = 1.0;
    }
    for col in 0..n {
        // pivot
        let mut piv = col;
        let mut best = aug[col * w + col].abs();
        for r in col + 1..n {
            let v = aug[r * w + col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best < 1e-12 {
            return None;
        }
        if piv != col {
            for j in 0..w {
                let t = aug[col * w + j];
                aug[col * w + j] = aug[piv * w + j];
                aug[piv * w + j] = t;
            }
        }
        let d = aug[col * w + col];
        for j in 0..w {
            aug[col * w + j] /= d;
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = aug[r * w + col];
            if f.abs() < 1e-18 {
                continue;
            }
            for j in 0..w {
                aug[r * w + j] -= f * aug[col * w + j];
            }
        }
    }
    let mut out = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = aug[i * w + (n + j)];
        }
    }
    Some(out)
}

/// Matrix multiply out = a (m x k) * b (k x n).
pub fn matmul(a: &[f64], m: usize, k: usize, b: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; m * n];
    for i in 0..m {
        for jj in 0..k {
            let aij = a[i * k + jj];
            if aij.abs() < 1e-30 {
                continue;
            }
            for j in 0..n {
                out[i * n + j] += aij * b[jj * n + j];
            }
        }
    }
    out
}

/// Symmetric positive-definite check + log|C| + the true inverse C^-1.
/// Uses Cholesky for log|C|, Gauss-Jordan for the inverse.
pub fn logdet_invert(c: &[f64], n: usize) -> Option<(f64, Vec<f64>)> {
    // Cholesky for logdet
    let mut ld = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = c[i * n + j];
            for k in 0..j {
                sum -= ld[i * n + k] * ld[j * n + k];
            }
            if j == i {
                if sum <= 1e-12 {
                    return None;
                }
                ld[i * n + i] = sum.sqrt();
            } else {
                ld[i * n + j] = sum / ld[j * n + j];
            }
        }
    }
    let mut lg = 0.0f64;
    for i in 0..n {
        lg += 2.0 * ld[i * n + i].ln();
    }
    let c_inv = invert(c, n)?;
    Some((lg, c_inv))
}

// ─── data parse ─────────────────────────────────────────────────────────

pub fn parse_numbers(s: &str) -> Result<Vec<f64>, ()> {
    let mut out: Vec<f64> = Vec::with_capacity(s.len() / 4);
    let mut buf = String::new();
    let flush = |buf: &mut String, out: &mut Vec<f64>| -> Result<(), ()> {
        if !buf.is_empty() {
            buf.parse::<f64>().map(|v| out.push(v)).map_err(|_| ())?;
            buf.clear();
        }
        Ok(())
    };
    for c in s.chars() {
        if c.is_ascii_digit() || c == '.' || c == '-' || c == '+' || c == 'e' || c == 'E' {
            buf.push(c);
        } else if c.is_whitespace() || c == ',' {
            flush(&mut buf, &mut out)?;
        } else {
            return Err(());
        }
    }
    flush(&mut buf, &mut out)?;
    Ok(out)
}

// ─── FIML-EM CFA ────────────────────────────────────────────────────────

pub struct CfaResult {
    pub lambda: Vec<f64>,  // p*q, row-major
    pub psi: Vec<f64>,     // p
    pub ll_model: f64,
    pub ll_saturated: f64,
    pub iterations: usize,
}

/// Jacobi eigen-decomposition for symmetric p x p; returns
/// (eigenvalues, eigenvectors-as-columns), both full.
fn jacobi_eig(a: &[f64], p: usize) -> (Vec<f64>, Vec<f64>) {
    let mut d = vec![0.0f64; p * p];
    for i in 0..p {
        for j in 0..p {
            d[i * p + j] = 0.5 * (a[i * p + j] + a[j * p + i]);
        }
    }
    let mut v = vec![0.0f64; p * p];
    for i in 0..p {
        v[i * p + i] = 1.0;
    }
    for _sweep in 0..60 {
        let mut off = 0.0f64;
        for i in 0..p {
            for j in (i + 1)..p {
                off += d[i * p + j] * d[i * p + j];
            }
        }
        if off < 1e-24 {
            break;
        }
        for i in 0..p {
            for j in (i + 1)..p {
                if d[i * p + j].abs() < 1e-16 {
                    continue;
                }
                let th = 0.5 * d[i * p + i] - 0.5 * d[j * p + j];
                let t = d[i * p + j] / (th.abs() + d[i * p + j].abs());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                let d_ii = d[i * p + i];
                let d_jj = d[j * p + j];
                d[i * p + i] = d_ii - t * d[i * p + j];
                d[j * p + j] = d_jj + t * d[i * p + j];
                d[i * p + j] = 0.0;
                d[j * p + i] = 0.0;
                for k in 0..p {
                    if k != i && k != j {
                        let tmp = d[k * p + i] * c + d[k * p + j] * s;
                        d[k * p + j] = -d[k * p + i] * s + d[k * p + j] * c;
                        d[k * p + i] = tmp;
                        d[i * p + k] = d[k * p + i];
                        d[j * p + k] = d[k * p + j];
                    }
                }
                for k in 0..p {
                    let vk_i = v[k * p + i];
                    let vk_j = v[k * p + j];
                    v[k * p + i] = vk_i * c - vk_j * s;
                    v[k * p + j] = vk_i * s + vk_j * c;
                }
            }
        }
    }
    let ev: Vec<f64> = (0..p).map(|i| d[i * p + i]).collect();
    (ev, v)
}

/// Initialize (Lambda, Psi) for the FIML EM. `mode` chooses the basin:
///   0 = top-q eigenvectors (largest variance),
///   1 = bottom-q eigenvectors (smallest variance),
///   2 = diagonal (sqrt(variance) on first q items).
/// Factor analysis has many local maxima; multi-start over modes keeps the best.
pub fn pca_init(s: &[f64], p: usize, q: usize, mode: usize) -> (Vec<f64>, Vec<f64>) {
    let (ev, v) = jacobi_eig(s, p);
    let mut order: Vec<usize> = (0..p).collect();
    order.sort_by(|a, b| ev[*b].partial_cmp(&ev[*a]).unwrap_or(std::cmp::Ordering::Equal));
    let top: Vec<usize> = if mode == 1 {
        order[p - q..].to_vec() // smallest
    } else {
        order[..q].to_vec() // largest
    };
    let mut lam = vec![0.0f64; p * q];
    for (qq, col) in top.iter().enumerate() {
        let ev = ev[*col].max(0.0).sqrt();
        for i in 0..p {
            lam[i * q + qq] = v[i * p + col] * ev;
        }
    }
    let mut psi = vec![0.0f64; p];
    for i in 0..p {
        let mut ssum = 0.0;
        for k in 0..q {
            ssum += lam[i * q + k] * lam[i * q + k];
        }
        psi[i] = (s[i * p + i] - ssum).max(1e-6);
    }
    (lam, psi)
}

pub fn sample_cov(data: &[f64], n: usize, p: usize) -> Vec<f64> {
    let mut colmean = vec![0.0f64; p];
    for i in 0..n {
        for j in 0..p {
            colmean[j] += data[i * p + j];
        }
    }
    for j in 0..p {
        colmean[j] /= n as f64;
    }
    let mut s = vec![0.0f64; p * p];
    for i in 0..n {
        for j in 0..p {
            for k in 0..p {
                s[j * p + k] += (data[i * p + j] - colmean[j]) * (data[i * p + k] - colmean[k]);
            }
        }
    }
    for j in 0..p {
        for k in 0..p {
            s[j * p + k] /= n as f64;
        }
    }
    s
}

/// Fit CFA from raw data: build the sample covariance, then delegate to
/// cfa_cov_fit.
pub fn cfa_fit(
    data: &[f64],
    n: usize,
    p: usize,
    q: usize,
    max_iter: usize,
    tol: f64,
) -> Option<CfaResult> {
    let s = sample_cov(data, n, p);
    cfa_cov_fit(&s, p, q, n, max_iter, tol)
}

/// FIML covariance-based EM for ML factor analysis (Phi = I), operating
/// directly on the sample covariance S. Mirrors test/oracle_factor.py.
///
/// E-step:  E[ff^T|x] = I - Lambda' Sigma^-1 Lambda + Lambda' Sigma^-1 xx' Sigma^-1 Lambda
///          so  B    = Lambda' Q Lambda + I - Lambda' R,  R = Sigma^-1 Lambda,
///               Q    = Sigma^-1 S Sigma^-1   (= (1/n) sum S_i S_i')
///               Lbar = S R                  (= (1/n) sum x_i m_i')
/// M-step:  Lambda_new = Lbar B^-1,  Psi_j = S_jj - (Lambda_new B Lambda_new')_jj
pub fn cfa_em(
    s: &[f64],
    p: usize,
    q: usize,
    n: usize,
    mode: usize,
    max_iter: usize,
    tol: f64,
    jitter: f64,
) -> Option<CfaResult> {
    if n < p || p < 2 || q < 1 || q > p - 1 {
        return None;
    }
    let (mut lam, mut psi) = pca_init(s, p, q, mode);
    // Deterministic per-start perturbation of the loadings to escape local
    // maxima (different basin per jitter value). jitter=0 => pure PCA init.
    if jitter > 0.0 {
        for i in 0..p * q {
            let h = (i as f64 * 1.93 + jitter * 12.9898).sin() * 43758.5453;
            let frac = h - h.floor(); // in [0,1)
            lam[i] += (frac - 0.5) * 2.0 * jitter * 0.3;
        }
    }
    let nf = n as f64;
    let mut prev_ll = f64::NEG_INFINITY;
    let mut iters: usize = 0;
    for it in 0..max_iter {
        iters = it + 1;
        // Sigma = Lambda Lambda^T + diag(Psi)
        let mut sigma = vec![0.0f64; p * p];
        for j in 0..p {
            sigma[j * p + j] = psi[j];
        }
        for a in 0..p {
            for b in 0..p {
                let mut v = 0.0f64;
                for k in 0..q {
                    v += lam[a * q + k] * lam[b * q + k];
                }
                sigma[a * p + b] += v;
            }
        }
        let (logdet_sigma, sigma_inv) = logdet_invert(&sigma, p)?;
        let ll_model =
            -0.5 * nf * (p as f64 * LOG_2PI + logdet_sigma + trace_mul(&sigma_inv, s, p));
        if ll_model - prev_ll < tol && it > 0 {
            break;
        }
        prev_ll = ll_model;
        // R = Sigma^-1 Lambda        (p x q)
        let r = matmul(&sigma_inv, p, p, &lam, q);
        // Q = Sigma^-1 S Sigma^-1    (p x p)
        let s_siginv = matmul(s, p, p, &sigma_inv, p);
        let q_mat = matmul(&sigma_inv, p, p, &s_siginv, p);
        // Lbar = S R                 (p x q)
        let lbar = matmul(s, p, p, &r, q);
        // B = Lambda' Q Lambda + I - Lambda' R   (q x q)
        let lamt_q = matmul(&trans(&lam, p, q), q, p, &q_mat, p); // Lambda' Q (q x p)
        let lamt_q_lam = matmul(&lamt_q, q, p, &lam, q); // Lambda' Q Lambda (q x q)
        let lamt_r = matmul(&trans(&lam, p, q), q, p, &r, q); // Lambda' R (q x q)
        let mut b = vec![0.0f64; q * q];
        for k in 0..q {
            for l in 0..q {
                b[k * q + l] = lamt_q_lam[k * q + l] - lamt_r[k * q + l];
                if k == l {
                    b[k * q + l] += 1.0;
                }
            }
        }
        let b_inv = invert(&b, q)?;
        let lam_new = matmul(&lbar, p, q, &b_inv, q);
        // M = Lbar B^-1 Lbar' ; Psi_j = S_jj - M_jj  (use B^-1 directly,
        // since B is non-symmetric -> lam_new B lam_new' != M)
        let mut psi_new = vec![0.0f64; p];
        for j in 0..p {
            let mut m_jj = 0.0f64;
            for k in 0..q {
                for l in 0..q {
                    m_jj += lbar[j * q + k] * b_inv[k * q + l] * lbar[j * q + l];
                }
            }
            psi_new[j] = (s[j * p + j] - m_jj).max(1e-9);
        }
        if lam_new.iter().all(|&v| v.is_finite()) && psi_new.iter().all(|&v| v.is_finite()) {
            lam = lam_new;
            psi = psi_new;
        } else {
            break;
        }
    }
    // final LL
    let mut sigma = vec![0.0f64; p * p];
    for j in 0..p {
        sigma[j * p + j] = psi[j];
    }
    for a in 0..p {
        for b in 0..p {
            let mut v = 0.0f64;
            for k in 0..q {
                v += lam[a * q + k] * lam[b * q + k];
            }
            sigma[a * p + b] += v;
        }
    }
    let (logdet_sigma, sigma_inv) = logdet_invert(&sigma, p)?;
    let ll_model =
        -0.5 * nf * (p as f64 * LOG_2PI + logdet_sigma + trace_mul(&sigma_inv, s, p));
    let (logdet_s, _) = logdet_invert(s, p)?;
    let ll_saturated = -0.5 * nf * (p as f64 * LOG_2PI + logdet_s + p as f64);
    Some(CfaResult {
        lambda: lam,
        psi,
        ll_model,
        ll_saturated,
        iterations: iters,
    })
}

/// FIML/ML CFA via covariance EM with **multi-start** (PCA-largest,
/// PCA-smallest, diagonal). Factor analysis has many local maxima; each
/// start lands in a different basin, so we run the EM from each and keep the
/// solution with the highest log-likelihood.
pub fn cfa_cov_fit(s: &[f64], p: usize, q: usize, n: usize, max_iter: usize, tol: f64) -> Option<CfaResult> {
    if n < p || p < 2 || q < 1 || q > p - 1 {
        return None;
    }
    let nf = n as f64;
    let (logdet_s, _) = logdet_invert(s, p)?;
    let ll_sat = -0.5 * nf * (p as f64 * LOG_2PI + logdet_s + p as f64);

    let mut best: Option<CfaResult> = None;
    // 3 structural starts + 6 deterministic perturbation starts. Each lands in
    // a different basin; keep the global-best LL (standard for FA local maxima).
    let mut starts: Vec<(usize, f64)> = Vec::new();
    for mode in 0..3usize {
        starts.push((mode, 0.0));
    }
    for k in 0..6usize {
        starts.push((0, (k + 1) as f64 * 0.17));
    }
    for (mode, jitter) in starts {
        if let Some(mut r) = cfa_em(s, p, q, n, mode, max_iter, tol, jitter) {
            r.ll_saturated = ll_sat;
            match &mut best {
                None => best = Some(r),
                Some(b) if r.ll_model > b.ll_model => best = Some(r),
                _ => {}
            }
        }
    }
    best
}

// ─── fit indices ────────────────────────────────────────────────────────

fn trans(a: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut t = vec![0.0f64; m * n];
    for i in 0..n {
        for j in 0..m {
            t[i * m + j] = a[j * n + i];
        }
    }
    t
}

fn trace_mul(a: &[f64], b: &[f64], n: usize) -> f64 {
    let mut t = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            t += a[i * n + j] * b[j * n + i];
        }
    }
    t
}

// ─── fit indices ────────────────────────────────────────────────────────

pub struct FitIndices {
    pub chi2: f64,
    pub df: i64,
    pub aic: f64,
    pub bic: f64,
    pub rmsea: f64,
    pub cfi: f64,
    pub tli: f64,
}

/// Fit indices. chi2 from the model vs saturated; CFI/TLI against the
/// independence (null) model Sigma = diag(S).
pub fn fit_indices(
    ll_model: f64,
    ll_saturated: f64,
    s: &[f64],
    n: usize,
    p: usize,
    q: usize,
) -> FitIndices {
    let df = (p * (p + 1) / 2 - (p * q + p)) as i64;
    let chi2 = -2.0 * (ll_model - ll_saturated);
    let nf = n as f64;
    let aic = chi2 - 2.0 * df as f64;
    let bic = chi2 + nf.ln() * df as f64;
    let rmsea = if df > 0 {
        let x = chi2 / df as f64 - 1.0;
        if x > 0.0 {
            (x / (nf - 1.0)).sqrt()
        } else {
            0.0
        }
    } else {
        0.0
    };
    // independence model: Sigma0 = diag(S)
    let mut diag_s = 0.0f64;
    for j in 0..p {
        if s[j * p + j] <= 1e-12 {
            return FitIndices {
                chi2,
                df,
                aic,
                bic,
                rmsea,
                cfi: 1.0,
                tli: 1.0,
            };
        }
        diag_s += s[j * p + j].ln();
    }
    let mut tr_inv_s = 0.0f64;
    for j in 0..p {
        tr_inv_s += s[j * p + j] / s[j * p + j]; // = 1 each => p
    }
    let ll_indep =
        -0.5 * nf * (p as f64 * LOG_2PI + diag_s + tr_inv_s);
    let chi2_base = -2.0 * (ll_indep - ll_saturated);
    let cfi = if chi2_base > 1e-9 {
        1.0 - (chi2 - chi2_base).max(0.0) / chi2_base
    } else {
        1.0
    };
    let tli = if chi2_base > 1e-9 {
        (chi2_base - chi2) / chi2_base
    } else {
        1.0
    };
    FitIndices {
        chi2,
        df,
        aic,
        bic,
        rmsea,
        cfi,
        tli,
    }
}
