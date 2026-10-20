#!/usr/bin/env bash
# duckdb_factor_sem e2e runner. Builds (if needed) + runs test/test_factor.sql
# and asserts the final row reports n_fail = 0.
set -euo pipefail
cd "$(dirname "$0")/.."
EXT=target/release/factor_sem_ext.duckdb_extension
if [ ! -f "$EXT" ]; then
  echo "missing $EXT — build first (see README 'Build')" >&2
  exit 2
fi
OUT=$(duckdb -unsigned -noheader -list < test/test_factor.sql)
echo "raw: $OUT"
# OUT is "<n_checks>|<n_fail>|<failing>" (DuckDB -list separator)
NFAIL=$(awk -F'|' '{print $2}' <<<"$OUT")
NCHK=$(awk -F'|' '{print $1}' <<<"$OUT")
FAILING=$(awk -F'|' '{print $3}' <<<"$OUT")
if [ "$NFAIL" = "0" ]; then
  echo "PASS: $NCHK checks, 0 failures"
  exit 0
else
  echo "FAIL: $NFAIL/$NCHK checks failed: $FAILING" >&2
  exit 1
fi
