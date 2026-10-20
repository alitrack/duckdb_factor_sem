#!/usr/bin/env python3
"""Generate test/test_factor.sql (full e2e): Part 1 cfa_cov, Part 2 cfa_fit,
Part 3 efa (v0.2.0). All reference values computed by the independent numpy
oracle (test/oracle_efa.py) so the built binary is checked end-to-end."""
import sys
sys.path.insert(0, "test")
import numpy as np
from oracle_efa import run_case

EXT = "/mnt/d/wsl2/duckdb_factor_sem/target/release/factor_sem_ext.duckdb_extension"

# ---- Part 1: cfa_cov (fixed, from factor_ref oracle — unchanged) ----
COV = ('400 4 1.059306696555 0.608905201368 0.483124436686 0.512072657011 '
       '0.608905201368 0.821927022365 0.364211667399 0.392906238896 '
       '0.483124436686 0.364211667399 0.732737594577 0.329043436641 '
       '0.512072657011 0.392906238896 0.329043436641 0.625895032989')
# ---- Part 2: cfa_fit raw (fixed raw40) ----
with open("test/raw40.txt") as fh:
    RAW = fh.read().strip()

# ---- Part 3: efa ----
A = run_case(40, 6, 2, seed=7, method="pca", rot="varimax")
B = run_case(30, 4, 1, seed=8, method="pa", rot="none")
C = run_case(60, 6, 2, seed=11, method="minres", rot="varimax")   # v0.2.1
D = run_case(60, 8, 2, seed=44, method="minres", rot="promax")    # v0.2.1 (oblique: phi != I)
data_str = lambda c: " | ".join(" ".join(f"{x:.6f}" for x in row) for row in c["data"])
lr = lambda c, i, k: c["loadings_rotated"][i][k]

sql = f"""-- duckdb_factor_sem e2e test (DuckDB-native). Each check emits (label, ok);
-- the final row reports n_checks|n_fail|failing; the runner (test/run_factor.sh)
-- asserts n_fail = 0.
--  Part 1: cfa_cov on a fixed covariance (S) — MLE independently verified by
--          the Rust unit test against the scipy L-BFGS-B oracle.
--  Part 2: cfa_fit (raw-data path) smoke.
--  Part 3: efa (v0.2.0) — pca/pa x varimax/none; reference values from the
--          independent numpy oracle (test/oracle_efa.py).
LOAD '{EXT}';

CREATE OR REPLACE MACRO num(j, key) AS json_extract(j, '$.' || key)::DOUBLE;
CREATE OR REPLACE MACRO ok(a, r, tol) AS
  CASE WHEN a IS NULL OR r IS NULL THEN false
       WHEN ABS(a - r) <= tol * GREATEST(1.0, ABS(r)) THEN true ELSE false END;

CREATE OR REPLACE TEMP TABLE fit_cov AS SELECT cfa_cov('{COV}', '1') AS j;
CREATE OR REPLACE TEMP TABLE fit_raw AS SELECT cfa_fit('{RAW}', '1') AS j;
CREATE OR REPLACE TEMP TABLE efa_a AS SELECT efa('{data_str(A)}', '2', 'pca', 'varimax') AS j;
CREATE OR REPLACE TEMP TABLE efa_b AS SELECT efa('{data_str(B)}', '1', 'pa', 'none') AS j;
CREATE OR REPLACE TEMP TABLE efa_c AS SELECT efa('{data_str(C)}', '2', 'minres', 'varimax') AS j;
CREATE OR REPLACE TEMP TABLE efa_d AS SELECT efa('{data_str(D)}', '2', 'minres', 'promax') AS j;

CREATE OR REPLACE TEMP TABLE checks AS
  SELECT 'cov.meta.p'  AS label, (num((SELECT j FROM fit_cov),'p') = 4) AS p
  UNION ALL SELECT 'cov.meta.q', (num((SELECT j FROM fit_cov),'q') = 1)
  UNION ALL SELECT 'cov.meta.n', (num((SELECT j FROM fit_cov),'n') = 400)
  UNION ALL SELECT 'cov.meta.df', (num((SELECT j FROM fit_cov),'df') = 2)
  UNION ALL SELECT 'cov.chi2',  ok(num((SELECT j FROM fit_cov),'chi2'),  0.7668204840169892,   1e-3)
  UNION ALL SELECT 'cov.cfi',   ok(num((SELECT j FROM fit_cov),'cfi'),   1.0,   1e-3)
  UNION ALL SELECT 'cov.tli',   ok(num((SELECT j FROM fit_cov),'tli'),   0.9987532065526415,   1e-3)
  UNION ALL SELECT 'cov.rmsea', ok(num((SELECT j FROM fit_cov),'rmsea'), 0.0, 1e-3)
  UNION ALL SELECT 'cov.ll',    ok(num((SELECT j FROM fit_cov),'ll_model'), -1779.562161172286, 1e-3)
  UNION ALL SELECT 'cov.lam[0]',  ok(num((SELECT j FROM fit_cov), 'lambda[0][0]')::DOUBLE, 0.8872922255534691, 2e-4)
  UNION ALL SELECT 'cov.lam[1]',  ok(num((SELECT j FROM fit_cov), 'lambda[1][0]')::DOUBLE, 0.6809603612022976, 2e-4)
  UNION ALL SELECT 'cov.lam[2]',  ok(num((SELECT j FROM fit_cov), 'lambda[2][0]')::DOUBLE, 0.5470201036164338, 2e-4)
  UNION ALL SELECT 'cov.lam[3]',  ok(num((SELECT j FROM fit_cov), 'lambda[3][0]')::DOUBLE, 0.5805361099537032, 2e-4)
  UNION ALL SELECT 'cov.psi[0]', ok(num((SELECT j FROM fit_cov), 'psi[0]')::DOUBLE, 0.27201916102214174, 2e-4)
  UNION ALL SELECT 'cov.psi[1]', ok(num((SELECT j FROM fit_cov), 'psi[1]')::DOUBLE, 0.35822002069132625, 2e-4)
  UNION ALL SELECT 'cov.psi[2]', ok(num((SELECT j FROM fit_cov), 'psi[2]')::DOUBLE, 0.433506611310045, 2e-4)
  UNION ALL SELECT 'cov.psi[3]', ok(num((SELECT j FROM fit_cov), 'psi[3]')::DOUBLE, 0.2888728618736866, 2e-4)
  UNION ALL SELECT 'raw.meta.p', (num((SELECT j FROM fit_raw),'p') = 4)
  UNION ALL SELECT 'raw.meta.q', (num((SELECT j FROM fit_raw),'q') = 1)
  UNION ALL SELECT 'raw.meta.n', (num((SELECT j FROM fit_raw),'n') = 40)
  UNION ALL SELECT 'raw.chi2_finite', (num((SELECT j FROM fit_raw),'chi2') IS NOT NULL)
  UNION ALL SELECT 'raw.cfi_range', (num((SELECT j FROM fit_raw),'cfi') IS NOT NULL)
  UNION ALL SELECT 'efaA.meta.p', (num((SELECT j FROM efa_a),'p') = 6)
  UNION ALL SELECT 'efaA.meta.q', (num((SELECT j FROM efa_a),'q') = 2)
  UNION ALL SELECT 'efaA.eig0',   ok(num((SELECT j FROM efa_a),'eigenvalues[0]'), {A['eigenvalues'][0]!r}, 1e-3)
  UNION ALL SELECT 'efaA.srml',   ok(num((SELECT j FROM efa_a),'srml'), {A['srml']!r}, 1e-3)
  UNION ALL SELECT 'efaA.lr00',   ok(num((SELECT j FROM efa_a),'lambda_rot[0][0]'), {lr(A,0,0)!r}, 2e-3)
  UNION ALL SELECT 'efaA.lr01',   ok(num((SELECT j FROM efa_a),'lambda_rot[0][1]'), {lr(A,0,1)!r}, 2e-3)
  UNION ALL SELECT 'efaB.meta.p', (num((SELECT j FROM efa_b),'p') = 4)
  UNION ALL SELECT 'efaB.meta.q', (num((SELECT j FROM efa_b),'q') = 1)
  UNION ALL SELECT 'efaB.eig0',   ok(num((SELECT j FROM efa_b),'eigenvalues[0]'), {B['eigenvalues'][0]!r}, 1e-3)
  UNION ALL SELECT 'efaB.srml',   ok(num((SELECT j FROM efa_b),'srml'), {B['srml']!r}, 1e-3)
  UNION ALL SELECT 'efaC.meta.p', (num((SELECT j FROM efa_c),'p') = 6)
  UNION ALL SELECT 'efaC.meta.q', (num((SELECT j FROM efa_c),'q') = 2)
  UNION ALL SELECT 'efaC.method', (json_extract_string((SELECT j FROM efa_c), '$.method') = 'minres')
  UNION ALL SELECT 'efaC.eig0',   ok(num((SELECT j FROM efa_c),'eigenvalues[0]'), {C['eigenvalues'][0]!r}, 1e-3)
  UNION ALL SELECT 'efaC.srml',   ok(num((SELECT j FROM efa_c),'srml'), {C['srml']!r}, 5e-3)
  UNION ALL SELECT 'efaC.phi00',  ok(num((SELECT j FROM efa_c),'phi[0][0]'), {C['phi'][0][0]!r}, 1e-3)
  UNION ALL SELECT 'efaD.meta.p', (num((SELECT j FROM efa_d),'p') = 8)
  UNION ALL SELECT 'efaD.rotate', (json_extract_string((SELECT j FROM efa_d), '$.rotate') = 'promax')
  UNION ALL SELECT 'efaD.eig0',   ok(num((SELECT j FROM efa_d),'eigenvalues[0]'), {D['eigenvalues'][0]!r}, 1e-3)
  UNION ALL SELECT 'efaD.srml',   ok(num((SELECT j FROM efa_d),'srml'), {D['srml']!r}, 5e-3)
  UNION ALL SELECT 'efaD.phi00',  ok(num((SELECT j FROM efa_d),'phi[0][0]'), {D['phi'][0][0]!r}, 1e-3)
  UNION ALL SELECT 'err.q0',      (cfa_fit('1 2 3 | 2 3 4 | 3 4 5', '0') IS NULL)
  UNION ALL SELECT 'err.qneg',    (cfa_fit('1 2 3 | 2 3 4 | 3 4 5', '-1') IS NULL)
  UNION ALL SELECT 'err.badd',    (cfa_fit('a b c | d e f | g h i', '1') IS NULL)
  UNION ALL SELECT 'err.q_gt_p',  (cfa_fit('1 2 | 3 4 | 5 6', '2') IS NULL)
  UNION ALL SELECT 'err.cov_n',   (cfa_cov('1 2 1.0 0.5 0.5 1.0', '1') IS NULL)
  UNION ALL SELECT 'err.efa_badmethod', (efa('1 2 | 3 4 | 5 6', '1', 'ml', 'none') IS NULL)
  UNION ALL SELECT 'err.efa_badrot',    (efa('1 2 | 3 4 | 5 6', '1', 'pca', 'quartmax') IS NULL);

SELECT
  COUNT(*) AS n_checks,
  COUNT(*) FILTER (WHERE NOT p) AS n_fail,
  string_agg(label, ', ' ORDER BY label) FILTER (WHERE NOT p) AS failing
FROM checks;
"""
with open("test/test_factor.sql", "w") as fh:
    fh.write(sql)
print(f"wrote test/test_factor.sql: efaA eig0={A['eigenvalues'][0]:.6f} srml={A['srml']:.6f} "
      f"lr00={lr(A,0,0):.4f} lr01={lr(A,0,1):.4f}; efaB eig0={B['eigenvalues'][0]:.6f} srml={B['srml']:.6f}; "
      f"efaC(minres/varimax) eig0={C['eigenvalues'][0]:.6f}; efaD(minres/promax) eig0={D['eigenvalues'][0]:.6f}")
print(f"total checks: {14+4+13+6} (cov 14 + raw 4 + efa 13 + err 6)")
