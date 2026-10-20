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
use crate::math::{matmul, trans};

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

/// Guttman fixed-point varimax on loadings L (p x q). Returns rotated
/// loadings + iteration count. Verified against scipy (test/oracle_efa.py).
fn varimax(lam: &[f64], p: usize, q: usize, delta: f64, max_iter: usize, tol: f64) -> (Vec<f64>, usize) {
    if q < 2 {
        return (lam.to_vec(), 0);
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
        // orthogonalize Tn: Tn (Tn' Tn)^{-1/2} via Jacobi
        let (w, v) = jacobi_eig(&tn, q);
        let mut t_new = vec![0.0f64; q * q];
        for j in 0..q {
            let wj = w[j].max(1e-300).sqrt();
            for i in 0..q {
                t_new[i * q + j] = v[i * q + j] / wj;
            }
        }
        // t_new = Tn @ diag(1/sqrt(w))
        let mut t_orth = vec![0.0f64; q * q];
        for i in 0..q {
            for j in 0..q {
                let mut s = 0.0f64;
                for k in 0..q {
                    s += tn[i * q + k] * t_new[k * q + j];
                }
                t_orth[i * q + j] = s;
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
    (lrot, iters)
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
    let (lam, w_desc) = match method {
        "pca" => pca_loadings(&r, p, q),
        "pa" => {
            let (l, w, _) = pa_loadings(&r, p, q, 10000, 1e-12)?;
            (l, w)
        }
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
    // rotate
    let (lam_rot, rot_iters) = match rotate {
        "none" => (lam.clone(), 0),
        "varimax" => varimax(&lam, p, q, 1e-8, 2000, 1e-12),
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
        srml: srm,
        rotate_iters: rot_iters,
    })
}
