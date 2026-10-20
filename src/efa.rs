// ─── EFA (v0.2.0) ─────────────────────────────────────────────────────
//! Exploratory Factor Analysis on the correlation matrix, manual factor
//! count. Factoring: pca (eigen decomposition of R) or pa (principal-axis,
//! Jöreskog fixed point on the uniqueness diagonal). Rotation: none or
//! varimax (Guttman fixed-point, verified against scipy in
//! test/oracle_efa.py). quartimax / minres / promax / parallel analysis
//! deferred to v0.2.1.
//!
//! Sign convention: each factor column's largest-|loading| is positive.

use crate::math::jacobi_eig;
use crate::math::{invert, matmul, trans};

pub struct EfaResult {
    pub p: usize,
    pub q: usize,
    pub n: usize,
    pub method: String,
    pub rotate: String,
    pub r_: Vec<f64>, // correlation matrix p x p (row-major, full)
    pub eigenvalues: Vec<f64>, // all p, descending
    pub proportion_var: Vec<f64>, // eigenvalues / sum
    pub lambda: Vec<f64>, // unrotated loadings p x q row-major
    pub lambda_rot: Vec<f64>, // rotated loadings p x q row-major
    pub uniquenesses: Vec<f64>, // p
    pub communality: Vec<f64>, // p
    pub phi: Vec<f64>, // factor correlation q x q (identity for orthogonal)
    pub srml: f64, // SRMR over upper triangle incl. diagonal
    pub rotate_iters: usize,
}

/// Symmetric covariance from raw data, full matrix (all i,j).
pub fn sample_cov_full(data: &[f64], n: usize, p: usize) -> Vec<f64> {
    let mut colmean = vec![0.0f64; p];
    for i in 0..n {
        for j in 0..p {
            colmean[j] += data[i * p + j];
        }
    }
    for j in 0..p {
        colmean[j] /= n as f64;
    }
    let mut cov = vec![0.0f64; p * p];
    for i in 0..n {
        for a in 0..p {
            let xa = data[i * p + a] - colmean[a];
            for b in 0..p {
                cov[a * p + b] += xa * (data[i * p + b] - colmean[b]);
            }
        }
    }
    let d = 1.0 / (n as f64 - 1.0);
    for v in cov.iter_mut() {
        *v *= d;
    }
    // symmetrize (floating-point)
    for i in 0..p {
        for j in (i + 1)..p {
            let m = 0.5 * (cov[i * p + j] + cov[j * p + i]);
            cov[i * p + j] = m;
            cov[j * p + i] = m;
        }
    }
    cov
}

/// Correlation matrix from a covariance (unit diagonal).
pub fn cov2cor(cov: &[f64], p: usize) -> Option<Vec<f64>> {
    let mut diag = vec![0.0f64; p];
    for j in 0..p {
        let v = cov[j * p + j];
        if v <= 0.0 {
            return None;
        }
        diag[j] = 1.0 / v.sqrt();
    }
    let mut r = vec![0.0f64; p * p];
    for i in 0..p {
        for j in 0..p {
            r[i * p + j] = cov[i * p + j] * diag[i] * diag[j];
        }
    }
    for j in 0..p {
        r[j * p + j] = 1.0;
    }
    Some(r)
}

/// PCA loadings from R: top-q eigenvectors scaled by sqrt(eigenvalue).
fn pca_loadings(r: &[f64], p: usize, q: usize) -> (Vec<f64>, Vec<f64>) {
    let (w, v) = jacobi_eig(r, p);
    let mut order: Vec<usize> = (0..p).collect();
    order.sort_by(|a, b| {
        w[*b]
            .partial_cmp(&w[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let w_desc: Vec<f64> = order.iter().map(|&i| w[i]).collect();
    let mut lam = vec![0.0f64; p * q];
    for j in 0..q {
        let wi = w_desc[j].max(0.0).sqrt();
        let col = order[j]; // eigenvector column index in V
        for i in 0..p {
            lam[i * q + j] = v[i * p + col] * wi;
        }
    }
    (lam, w_desc)
}

/// Principal-axis (Thurstone) loadings: iterate on the uniqueness diagonal.
fn pa_loadings(r: &[f64], p: usize, q: usize, max_iter: usize, tol: f64) -> Option<(Vec<f64>, Vec<f64>, usize)> {
    let mut d = vec![1.0f64; p];
    let mut lam = vec![0.0f64; p * q];
    let mut w_desc = vec![0.0f64; p];
    let mut iters = 0;
    for it in 0..max_iter {
        iters = it + 1;
        // Rr = R - diag(d)
        let mut rr = r.to_vec();
        for j in 0..p {
            rr[j * p + j] -= d[j];
        }
        let (w, v) = jacobi_eig(&rr, p);
        let mut order: Vec<usize> = (0..p).collect();
        order.sort_by(|a, b| {
            w[*b]
                .partial_cmp(&w[*a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        w_desc = order.iter().map(|&i| w[i]).collect();
        for j in 0..q {
            let wj = w_desc[j].max(0.0);
            let col = order[j];
            for i in 0..p {
                lam[i * q + j] = v[i * p + col] * wj.sqrt();
            }
        }
        let mut changed = false;
        for i in 0..p {
            let h: f64 = (0..q).map(|j| lam[i * q + j] * lam[i * q + j]).sum();
            let d_new = (1.0 - h).max(1e-10);
            if (d_new - d[i]).abs() > tol {
                changed = true;
            }
            d[i] = d_new;
        }
        if !changed {
            break;
        }
    }
    Some((lam, w_desc, iters))
}

/// SMC (Sata et al.) start for the uniqueness diagonal: 1 - 1/diag(inv R).
/// Mirrors psych::smc (cov2cor already applied, so R has unit diagonal).
fn smc(r: &[f64], p: usize) -> Vec<f64> {
    let mut inv = match invert(r, p) {
        Some(x) => x,
        None => return vec![1.0f64; p],
    };
    let mut out = vec![0.0f64; p];
    for j in 0..p {
        let d = inv[j * p + j];
        out[j] = if d.abs() < 1e-12 {
            1.0
        } else {
            (1.0 - 1.0 / d).clamp(0.0, 1.0)
        };
    }
    out
}

/// Given a uniqueness diagonal Psi, return the minres reduced-matrix loadings
/// (p x q) and the reproduced model L L' (p x p). Mirrors psych::fa:
/// reduced = R with diag <- 1 - Psi, eigen, top-q eigenvectors * sqrt(w).
fn minres_reduced(r: &[f64], psi: &[f64], p: usize, q: usize) -> (Vec<f64>, Vec<f64>) {
    let mut s = r.to_vec();
    for j in 0..p {
        s[j * p + j] = 1.0 - psi[j];
    }
    let (w, v) = jacobi_eig(&s, p);
    let mut order: Vec<usize> = (0..p).collect();
    order.sort_by(|a, b| w[*b].partial_cmp(&w[*a]).unwrap_or(std::cmp::Ordering::Equal));
    let mut lam = vec![0.0f64; p * q];
    for j in 0..q {
        let wj = w[order[j]].max(1e-300);
        for i in 0..p {
            lam[i * q + j] = v[i * p + order[j]] * wj.sqrt();
        }
    }
    let lt = trans(&lam, p, q);
    let model = matmul(&lam, p, q, &lt, p);
    (lam, model)
}

/// Minres objective: sum of the strictly-lower-triangular squares of
/// (R - (L L')), where L L' comes from the reduced matrix at Psi.
/// Mirrors psych::fit.residuals for fm=minres (lower.tri of (S - model)^2).
fn minres_obj(r: &[f64], psi: &[f64], p: usize, q: usize) -> f64 {
    let (_lam, model) = minres_reduced(r, psi, p, q);
    let mut acc = 0.0f64;
    for i in 1..p {
        for j in 0..i {
            let d = r[i * p + j] - model[i * p + j];
            acc += d * d;
        }
    }
    acc
}

/// Minres gradient w.r.t. Psi: diag(model + diag(Psi) - R). Mirrors
/// psych::FAgr.minres.
fn minres_grad(r: &[f64], psi: &[f64], p: usize, q: usize) -> Vec<f64> {
    let (_lam, model) = minres_reduced(r, psi, p, q);
    let mut g = vec![0.0f64; p];
    for j in 0..p {
        g[j] = model[j * p + j] + psi[j] - r[j * p + j];
    }
    g
}

/// Two-loop L-BFGS-B (Nocedal & Wright) with box bounds and backtracking
/// Armijo line search. Minimize f(x) over x in [lo, hi] elementwise.
/// Returns (x, f(x), iterations).
fn lbfgsb_b(
    f: impl Fn(&[f64]) -> f64,
    g: impl Fn(&[f64]) -> Vec<f64>,
    x0: &[f64],
    lo: &[f64],
    hi: &[f64],
    m: usize,
    max_iter: usize,
    tol: f64,
) -> (Vec<f64>, f64, usize) {
    let p = x0.len();
    let clip = |v: f64, j: usize| -> f64 { v.clamp(lo[j], hi[j]) };
    let mut x = x0.to_vec();
    let mut fx = f(&x);
    let mut gx = g(&x);
    let mut s_hist: Vec<Vec<f64>> = Vec::new(); // s = x_new - x
    let mut y_hist: Vec<Vec<f64>> = Vec::new(); // y = g_new - g
    let mut rho_hist: Vec<f64> = Vec::new(); // 1 / (y's)
    let mut alpha_hist: Vec<f64> = Vec::new();
    let mut iters = 0;
    for _ in 0..max_iter {
        iters += 1;
        // two-loop recursion to form search direction d (from current g)
        let k = s_hist.len();
        let mut alpha = vec![0.0f64; k];
        let mut r = gx.clone();
        // backward pass (i = k-1 .. 0)
        for i in (0..k).rev() {
            let mut yst = 0.0f64;
            let mut srt = 0.0f64;
            for t in 0..p {
                yst += y_hist[i][t] * s_hist[i][t];
                srt += s_hist[i][t] * r[t];
            }
            alpha[i] = rho_hist[i] * srt;
            for t in 0..p {
                r[t] -= (alpha[i] - rho_hist[i] * yst) * y_hist[i][t];
            }
        }
        // gamma from the most recent pair
        let gamma = if k == 0 {
            1.0
        } else {
            let j = k - 1;
            let mut yst = 0.0f64;
            let mut srt = 0.0f64;
            for t in 0..p {
                yst += y_hist[j][t] * s_hist[j][t];
                srt += s_hist[j][t] * r[t];
            }
            srt / yst
        };
        let mut d = vec![0.0f64; p];
        for t in 0..p {
            d[t] = gamma * r[t];
        }
        // forward pass (i = 0 .. k-1): d <- d + (alpha_i - beta_i) s_i,
        // beta_i = rho_i * y_i . d  (dot with the running direction)
        for i in 0..k {
            let mut beta = 0.0f64;
            for t in 0..p {
                beta += y_hist[i][t] * d[t];
            }
            beta *= rho_hist[i];
            for t in 0..p {
                d[t] += (alpha[i] - beta) * s_hist[i][t];
            }
        }
        // ensure descent: if d'g > 0, fall back to -g
        let mut dg = 0.0f64;
        for t in 0..p {
            dg += d[t] * gx[t];
        }
        if dg >= 0.0 {
            for t in 0..p {
                d[t] = -gx[t];
            }
        }
        // projected steepest-descent feasibility adjustment
        for t in 0..p {
            if (x[t] <= lo[t] && d[t] > 0.0) || (x[t] >= hi[t] && d[t] < 0.0) {
                d[t] = 0.0;
            }
        }
        // line search (backtracking Armijo on the projected step)
        let mut alpha0 = 1.0;
        let mut ok = false;
        for _ls in 0..30 {
            let mut xn = vec![0.0f64; p];
            for t in 0..p {
                xn[t] = clip(x[t] + alpha0 * d[t], t);
            }
            let fxn = f(&xn);
            let mut gd = 0.0f64;
            for t in 0..p {
                gd += gx[t] * (xn[t] - x[t]);
            }
            if fxn <= fx + 1e-4 * alpha0 * gd {
                ok = true;
                break;
            }
            alpha0 *= 0.5;
        }
        if !ok {
            break;
        }
        let mut xn = vec![0.0f64; p];
        for t in 0..p {
            xn[t] = clip(x[t] + alpha0 * d[t], t);
        }
        let fxn = f(&xn);
        let gxn = g(&xn);
        // convergence test
        let mut step = 0.0f64;
        for t in 0..p {
            step += (xn[t] - x[t]) * (xn[t] - x[t]);
        }
        let mut gmax = 0.0f64;
        for t in 0..p {
            gmax = gmax.max(gx[t].abs());
        }
        let conv = (step.sqrt() < tol * (1.0 + x.iter().map(|v| v.abs()).sum::<f64>()))
            && (gmax < tol * (1.0 + fx.abs()));
        if conv {
            x = xn;
            fx = fxn;
            break;
        }
        // update history
        let mut s = vec![0.0f64; p];
        let mut y = vec![0.0f64; p];
        let mut s_norm = 0.0f64;
        let mut y_norm = 0.0f64;
        for t in 0..p {
            s[t] = xn[t] - x[t];
            y[t] = gxn[t] - gx[t];
            s_norm += s[t] * s[t];
            y_norm += y[t] * y[t];
        }
        if s_norm.sqrt() > 1e-12 && y_norm.sqrt() > 1e-12 {
            let mut sy = 0.0f64;
            for t in 0..p {
                sy += s[t] * y[t];
            }
            s_hist.push(s);
            y_hist.push(y);
            rho_hist.push(1.0 / sy);
            alpha_hist.push(0.0);
            if s_hist.len() > m {
                s_hist.remove(0);
                y_hist.remove(0);
                rho_hist.remove(0);
                alpha_hist.remove(0);
            }
        }
        x = xn;
        fx = fxn;
        gx = gxn;
    }
    (x, fx, iters)
}

/// Minimum-residual (OLS) factoring, correlation basis. Minimize the
/// off-diagonal residual with L-BFGS-B over the uniqueness vector.
/// Mirrors psych::fa(fm="minres"). Returns (loadings, eigenvalues of R, iters).
fn minres_loadings(r: &[f64], p: usize, q: usize) -> Option<(Vec<f64>, Vec<f64>, usize)> {
    let start = smc(r, p);
    let lo: Vec<f64> = vec![0.005f64; p];
    let hi_v = r.iter().cloned().fold(1.0f64, |a, b| a.max(b));
    let hi: Vec<f64> = vec![hi_v.max(1.0); p];
    let mut s0 = vec![0.0f64; p];
    for j in 0..p {
        s0[j] = start[j].max(0.005);
    }
    // capture r, p, q in the closures
    let r_ = r.to_vec();
    let (psi, _fx, iters) =
        lbfgsb_b(
            |x| minres_obj(&r_, x, p, q),
            |x| minres_grad(&r_, x, p, q),
            &s0,
            &lo,
            &hi,
            20,
            10000,
            1e-10,
        );
    let (lam, _model) = minres_reduced(r, &psi, p, q);
    let (w, _v) = jacobi_eig(r, p);
    let mut order: Vec<usize> = (0..p).collect();
    order.sort_by(|a, b| w[*b].partial_cmp(&w[*a]).unwrap_or(std::cmp::Ordering::Equal));
    let w_desc: Vec<f64> = order.iter().map(|&i| w[i]).collect();
    Some((lam, w_desc, iters))
}

/// Kaiser-normalized promax (JASP's default oblique rotation). From psych
/// Promax.R + kaiser.R: W = L/sqrt(h2); varimax(W) -> (Wv, Th);
/// U = (Wv'Wv)^-1 Wv' (Wv|Wv|^(m-1)); scale U by diag(inv U'U) sqrt; U <- Th U;
/// Lr = Wv U * sqrt(h2); Phi = inv(U) inv(U)'. Returns (Lr, Phi).
pub fn kaiser_promax(lam: &[f64], p: usize, q: usize, m: f64) -> Option<(Vec<f64>, Vec<f64>)> {
    if q == 1 {
        return Some((lam.to_vec(), vec![1.0f64]));
    }
    // h2_i = sum_j L_ij^2
    let mut h2 = vec![0.0f64; p];
    for i in 0..p {
        for j in 0..q {
            h2[i] += lam[i * q + j] * lam[i * q + j];
        }
        h2[i] = h2[i].max(1e-12);
    }
    // W = L / sqrt(h2)
    let mut w = vec![0.0f64; p * q];
    for i in 0..p {
        let sc = h2[i].sqrt();
        for j in 0..q {
            w[i * q + j] = lam[i * q + j] / sc;
        }
    }
    // varimax(W) -> wv  (Guttman, sign-fixed).  Derive the pre-rotation
    // th_eff = lstsq(W, Wv) so W @ th_eff == Wv regardless of varimax's
    // column-sign convention (robust: absorbs any sign flips in wv).
    let (wv, _th, _it) = varimax(&w, p, q, 1e-8, 2000, 1e-12);
    // th_eff = (W' W)^-1 (W' Wv)   -- q x q
    let wt = trans(&w, p, q); // q x p
    let wtw_eff = matmul(&wt, q, p, &w, q); // W' W
    let wtq_eff = matmul(&wt, q, p, &wv, q); // W' Wv
    let th_eff = match invert(&wtw_eff, q) {
        Some(inv) => matmul(&inv, q, q, &wtq_eff, q),
        None => _th, // fallback: identity-ish if singular (shouldn't happen)
    };
    // Q = Wv * |Wv|^(m-1)
    let mut qm = vec![0.0f64; p * q];
    for i in 0..p {
        for j in 0..q {
            qm[i * q + j] = wv[i * q + j] * wv[i * q + j].abs().powf(m - 1.0);
        }
    }
    // U = (Wv' Wv)^-1 (Wv' Q)   -- solve q x q system (lstsq via normal eqs)
    let wt = trans(&wv, p, q); // q x p
    let wtw = matmul(&wt, q, p, &wv, q); // q x q = Wv' Wv
    let wtq = matmul(&wt, q, p, &qm, q); // q x q = Wv' Q
    let wtw_inv = invert(&wtw, q)?;
    let mut u = matmul(&wtw_inv, q, q, &wtq, q); // q x q
    // d = diag(inv(U' U));  U <- U * sqrt(d)  (row scale)
    let ut = trans(&u, q, q); // q x q
    let utu = matmul(&ut, q, q, &u, q); // U' U (q x q)
    let mut d = vec![1.0f64; q];
    if let Some(utu_inv) = invert(&utu, q) {
        for j in 0..q {
            d[j] = utu_inv[j * q + j].max(1e-300).sqrt();
        }
    }
    for j in 0..q {
        for i in 0..q {
            u[i * q + j] *= d[j];
        }
    }
    // Lr = Wv @ U_rescaled  (single-Th, since Wv = W @ Th).  Scaled by sqrt(h2).
    let u_rescaled = u;
    let mut lr = vec![0.0f64; p * q];
    for i in 0..p {
        let sc = h2[i].sqrt();
        for j in 0..q {
            let mut s = 0.0f64;
            for k in 0..q {
                s += wv[i * q + k] * u_rescaled[k * q + j];
            }
            lr[i * q + j] = s * sc;
        }
    }
    // sign-fix each column of Lr (largest |loading| positive)
    for j in 0..q {
        let mut bi = 0;
        let mut bv = f64::NEG_INFINITY;
        for i in 0..p {
            let a = lr[i * q + j].abs();
            if a > bv {
                bv = a;
                bi = i;
            }
        }
        if lr[bi * q + j] < 0.0 {
            for i in 0..p {
                lr[i * q + j] *= -1.0;
            }
        }
    }
    // U_final = th_eff @ U_rescaled  (the full promax rotation matrix)
    let u_final = matmul(&th_eff, q, q, &u_rescaled, q);
    // Phi = inv(U_final) @ inv(U_final)'
    let phi = {
        let ui = invert(&u_final, q)?; // q x q
        let uit = trans(&ui, q, q);
        matmul(&ui, q, q, &uit, q)
    };
    Some((lr, phi))
}

/// Guttman fixed-point varimax on loadings L (p x q). Returns rotated
/// loadings + iteration count. Verified against scipy (test/oracle_efa.py).
fn varimax(lam: &[f64], p: usize, q: usize, delta: f64, max_iter: usize, tol: f64) -> (Vec<f64>, Vec<f64>, usize) {
    if q < 2 {
        let mut ti = vec![0.0f64; q * q];
        for j in 0..q {
            ti[j * q + j] = 1.0;
        }
        return (lam.to_vec(), ti, 0);
    }
    let mut t = vec![0.0f64; q * q];
    for j in 0..q {
        t[j * q + j] = 1.0;
    }
    let mut iters = 0;
    for it in 0..max_iter {
        iters = it + 1;
        // U = L T  (p x q)
        let u = matmul(lam, p, q, &t, q);
        // gamma_j = sum_i U_ij^2
        let mut gamma = vec![0.0f64; q];
        for j in 0..q {
            for i in 0..p {
                let v = u[i * q + j];
                gamma[j] += v * v;
            }
        }
        // Us = U / (delta + gamma)  (column scale)
        let mut us = vec![0.0f64; p * q];
        for j in 0..q {
            let sc = 1.0 / (delta + gamma[j]);
            for i in 0..p {
                us[i * q + j] = u[i * q + j] * sc;
            }
        }
        // Tn = Us' U  (q x q, symmetric)
        let mut tn = vec![0.0f64; q * q];
        for i in 0..q {
            for j in 0..q {
                let mut s = 0.0f64;
                for k in 0..p {
                    s += us[k * q + i] * u[k * q + j];
                }
                tn[i * q + j] = s;
            }
        }
        // symmetrize
        for i in 0..q {
            for j in (i + 1)..q {
                let m = 0.5 * (tn[i * q + j] + tn[j * q + i]);
                tn[i * q + j] = m;
                tn[j * q + i] = m;
            }
        }
        // orthogonalize Tn: nearest orthogonal matrix.  Tn is symmetric,
        // so its polar (orthogonal) factor is V diag(sign(w)) V' (the
        // Jacobi eigenvectors V with eigenvalues w).
        let (w, v) = jacobi_eig(&tn, q);
        // sign_j = sign(w_j)
        let mut sign = vec![1.0f64; q];
        for j in 0..q {
            sign[j] = if w[j] < 0.0 { -1.0 } else { 1.0 };
        }
        // t_orth = (V diag(sign)) V'
        let mut t_orth = vec![0.0f64; q * q];
        for i in 0..q {
            for j in 0..q {
                let mij = v[i * q + j] * sign[j];
                for k in 0..q {
                    t_orth[i * q + k] += mij * v[k * q + j];
                }
            }
        }
        // sign-fix columns of T
        for j in 0..q {
            let mut bi = 0;
            let mut bv = f64::NEG_INFINITY;
            for i in 0..q {
                let a = t_orth[i * q + j].abs();
                if a > bv {
                    bv = a;
                    bi = i;
                }
            }
            if t_orth[bi * q + j] < 0.0 {
                for i in 0..q {
                    t_orth[i * q + j] *= -1.0;
                }
            }
        }
        let mut changed = false;
        for i in 0..q {
            for j in 0..q {
                if (t_orth[i * q + j] - t[i * q + j]).abs() > tol {
                    changed = true;
                }
            }
        }
        t = t_orth;
        if !changed {
            break;
        }
    }
    // Lrot = L T, then sign-fix each column (largest |loading| positive)
    let mut lrot = matmul(lam, p, q, &t, q);
    for j in 0..q {
        let mut bi = 0;
        let mut bv = f64::NEG_INFINITY;
        for i in 0..p {
            let a = lrot[i * q + j].abs();
            if a > bv {
                bv = a;
                bi = i;
            }
        }
        if lrot[bi * q + j] < 0.0 {
            for i in 0..p {
                lrot[i * q + j] *= -1.0;
            }
            for i in 0..q {
                t[i * q + j] *= -1.0;
            }
        }
    }
    (lrot, t, iters)
}

/// Identity factor-correlation matrix (orthogonal rotations).
fn identity_phi(q: usize) -> Vec<f64> {
    let mut phi = vec![0.0f64; q * q];
    for j in 0..q {
        phi[j * q + j] = 1.0;
    }
    phi
}

/// SRMR over the upper triangle (incl. diagonal) of (L L' + diag(uni) - R).
pub fn srml(r: &[f64], lam: &[f64], p: usize, q: usize, uni: &[f64]) -> f64 {
    // impl = L L' + diag(uni)
    let lt = trans(lam, p, q);
    let ll = matmul(lam, p, q, &lt, p);
    let mut cells = 0.0f64;
    let mut cnt = 0.0f64;
    for i in 0..p {
        for j in i..p {
            let impl_v = ll[i * p + j] + if i == j { uni[i] } else { 0.0 };
            let d = impl_v - r[i * p + j];
            cells += d * d;
            cnt += 1.0;
        }
    }
    (cells / cnt).sqrt()
}

pub fn efa_fit(
    data: &[f64],
    n: usize,
    p: usize,
    q: usize,
    method: &str,
    rotate: &str,
) -> Option<EfaResult> {
    if n < p + 1 || p < 2 || q < 1 || q > p {
        return None;
    }
    let cov = sample_cov_full(data, n, p);
    let r = cov2cor(&cov, p)?;
    // Eigenvalues of the correlation matrix (reported for all methods).
    let (w_r, _v_r) = jacobi_eig(&r, p);
    let mut order_r: Vec<usize> = (0..p).collect();
    order_r.sort_by(|a, b| w_r[*b].partial_cmp(&w_r[*a]).unwrap_or(std::cmp::Ordering::Equal));
    let w_desc: Vec<f64> = order_r.iter().map(|&i| w_r[i]).collect();
    let lam = match method {
        "pca" => pca_loadings(&r, p, q).0,
        "pa" => pa_loadings(&r, p, q, 10000, 1e-12)?.0,
        "minres" => minres_loadings(&r, p, q)?.0,
        _ => return None,
    };
    // communality / uniquenesses
    let mut h = vec![0.0f64; p];
    for i in 0..p {
        let mut s = 0.0f64;
        for j in 0..q {
            s += lam[i * q + j] * lam[i * q + j];
        }
        h[i] = s;
    }
    let uni: Vec<f64> = h.iter().map(|v| (1.0 - v).max(0.0)).collect();
    // rotate: none / varimax (orthogonal, Phi=I) or promax (oblique, Phi)
    let (lam_rot, phi, rot_iters) = match rotate {
        "none" => (lam.clone(), identity_phi(q), 0),
        "varimax" => {
            let (lrot, _th, it) = varimax(&lam, p, q, 1e-8, 2000, 1e-12);
            (lrot, identity_phi(q), it)
        }
        "promax" => {
            let (lrot, phi) = kaiser_promax(&lam, p, q, 4.0)?;
            (lrot, phi, 1)
        }
        _ => return None,
    };
    // proportion of variance
    let tot: f64 = w_desc.iter().sum();
    let prop: Vec<f64> = if tot > 0.0 {
        w_desc.iter().map(|v| v / tot).collect()
    } else {
        vec![0.0f64; p]
    };
    // SRMR must be computed before the fields are moved into the struct.
    let srm = srml(&r, &lam_rot, p, q, &uni);
    Some(EfaResult {
        p,
        q,
        n,
        method: method.to_string(),
        rotate: rotate.to_string(),
        r_: r,
        eigenvalues: w_desc,
        proportion_var: prop,
        lambda: lam,
        lambda_rot: lam_rot,
        uniquenesses: uni,
        communality: h,
        phi: phi,
        srml: srm,
        rotate_iters: rot_iters,
    })
}
