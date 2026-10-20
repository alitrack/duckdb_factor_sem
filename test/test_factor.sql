-- duckdb_factor_sem v0.1.0 end-to-end test (DuckDB-native).
-- Each check emits (label, ok); final row reports n_fail; the
-- runner (test/run_factor.sh) asserts n_fail = 0.
--
-- Part 1: cfa_cov on the oracle case-0 covariance (S). This S's
-- MLE is independently verified by the Rust unit test against the
-- scipy L-BFGS-B oracle, so this checks the built binary +
-- C-API glue + JSON + covariance parsing against a known value.
-- Loadings compared in |.| (eigenvector sign is arbitrary);
-- psi + fit indices are sign-invariant and compared directly.
-- Part 2: cfa_fit (raw-data path) smoke test.

LOAD '/mnt/d/wsl2/duckdb_factor_sem/target/release/factor_sem_ext.duckdb_extension';

CREATE OR REPLACE MACRO num(j, key) AS json_extract(j, '$.' || key)::DOUBLE;
CREATE OR REPLACE MACRO ok(a, r, tol) AS
  CASE WHEN a IS NULL OR r IS NULL THEN false
       WHEN ABS(a - r) <= tol * GREATEST(1.0, ABS(r)) THEN true ELSE false END;

CREATE OR REPLACE TEMP TABLE fit_cov AS SELECT cfa_cov('400 4 1.059306696555 0.608905201368 0.483124436686 0.512072657011 0.608905201368 0.821927022365 0.364211667399 0.392906238896 0.483124436686 0.364211667399 0.732737594577 0.329043436641 0.512072657011 0.392906238896 0.329043436641 0.625895032989', '1') AS j;

CREATE OR REPLACE TEMP TABLE fit_raw AS SELECT cfa_fit('0.6813 0.5260 -0.1817 0.2795 | -0.8721 -0.6602 -0.1467 -0.3975 | 1.0473 0.5999 0.6086 0.7210 | 0.0484 0.5303 0.3067 0.1204 | -1.9066 -0.6445 -1.6449 -0.4452 | -2.0937 -1.1601 -0.6922 -0.3300 | 0.5046 0.5304 -0.1143 -0.1893 | 0.1853 -0.3420 -0.8885 -0.7788 | -0.5187 0.2597 0.0679 0.3698 | -1.0018 -0.5529 -0.1692 -0.5960 | 1.0416 0.2970 0.3288 0.2306 | 0.0450 0.8501 0.2096 0.3957 | 0.3227 0.2941 0.4041 -0.0209 | 0.7827 0.8018 -0.2478 -0.2290 | -0.3037 -0.1956 0.4995 -0.2622 | -0.9805 0.0671 -0.7107 -0.0257 | -0.1795 0.1640 -0.2991 -0.0013 | -0.4027 -1.6653 -0.3374 -0.3492 | 0.4652 -0.1332 0.5666 0.1492 | 0.0825 -0.0255 0.8474 -0.1561 | -0.7270 -0.0405 0.0096 0.6520 | -0.1554 -0.3152 0.3929 -0.9916 | 0.7499 0.4094 0.5200 -0.1428 | 0.2088 -0.2376 -0.8983 -0.6335 | -0.2138 0.1378 0.8367 1.3818 | -0.0899 -0.8061 -1.3790 -0.0294 | 0.0338 0.1717 -0.0159 0.1890 | 0.9128 0.3601 0.1324 -0.3845 | -0.5458 0.0432 0.2182 1.1747 | 0.4591 0.8614 -0.0150 -0.4336 | 1.3989 1.2090 2.4508 0.6209 | 0.0935 -0.7994 0.2664 0.0076 | -0.5468 -0.4065 -0.6660 -0.0118 | -0.9816 -1.2816 -1.1882 -0.3124 | 1.4193 0.5496 0.3046 0.4645 | 1.7314 0.9669 0.4523 1.1704 | 0.1323 0.7557 0.0320 -0.7276 | -1.5055 0.2741 0.4400 -0.5184 | -0.9519 0.1821 -1.1010 -0.9023 | 0.9379 0.2718 0.3876 0.2358', '1') AS j;

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
  UNION ALL SELECT 'err.q0',     (cfa_fit('1 2 3 | 2 3 4 | 3 4 5', '0') IS NULL)
  UNION ALL SELECT 'err.qneg',   (cfa_fit('1 2 3 | 2 3 4 | 3 4 5', '-1') IS NULL)
  UNION ALL SELECT 'err.badd',   (cfa_fit('a b c | d e f | g h i', '1') IS NULL)
  UNION ALL SELECT 'err.q_gt_p', (cfa_fit('1 2 | 3 4 | 5 6', '2') IS NULL)
  UNION ALL SELECT 'err.cov_n',  (cfa_cov('1 2 1.0 0.5 0.5 1.0', '1') IS NULL);

SELECT
  COUNT(*) AS n_checks,
  COUNT(*) FILTER (WHERE NOT p) AS n_fail,
  string_agg(label, ', ' ORDER BY label) FILTER (WHERE NOT p) AS failing
FROM checks;
