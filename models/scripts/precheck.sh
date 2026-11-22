#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
P=$(uv python find 3.11 2>/dev/null || (uv python install 3.11 >/dev/null && uv python find 3.11))
fail=0
for f in $(find . -name "*.py" -not -path "./.venv/*"); do $P -m py_compile "$f" || { echo "FAIL $f"; fail=1; }; done
for f in setup.sh scripts/*.sh; do bash -n "$f" || { echo "FAIL $f"; fail=1; }; done
[ $fail = 0 ] && echo "compile ok on $($P --version)"
grep -q "prefix: DRY_" configs/config_dry.yaml && echo "dry names ok" || { echo "config_dry.yaml has no DRY_ prefix"; fail=1; }
PYBIN=${PYBIN:-.venv/bin/python}
[ -x "$PYBIN" ] || PYBIN=python3
$PYBIN run.py diffcheck
$PYBIN -m pytest -q -m "not slow"
exit $fail
