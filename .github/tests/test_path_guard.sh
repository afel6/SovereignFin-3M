#!/usr/bin/env bash
# Runs the exact `run:` block of .github/workflows/path-guard.yml against a stubbed `gh`.
# Usage: bash .github/tests/test_path_guard.sh
set -u
here=$(cd "$(dirname "$0")" && pwd)
workflow="$here/../workflows/path-guard.yml"
guard=$(sed -n '/# --- guard start/,/# --- guard end/p' "$workflow" | sed 's/^          //')
[ -n "$guard" ] || { echo "FAIL guard block not found in $workflow"; exit 1; }

# Login currently mapped to an allowlist (e.g. "frontend/"), so the cases keep working
# after the MN_LOGIN placeholders are replaced with real logins.
login_for() {
  printf '%s\n' "$guard" | sed -n "s|^ *\([^ )]*\)) *allow=\"$1[ \"].*|\1|p"
}
lead=$(printf '%s\n' "$guard" | sed -n 's/^LEAD="\(.*\)"$/\1/p')
m2=$(login_for data_pipeline/clean/)
m3=$(login_for rag_engine/contrib/)
m4=$(login_for database/charts/)
m5=$(login_for atlas_kv/contrib/)
m6=$(login_for frontend/)
m7=$(login_for evaluation/safety/)
m8=$(login_for evaluation/benchmark/)

fails=0
for who in lead m2 m3 m4 m5 m6 m7 m8; do
  [ -n "${!who}" ] || { echo "FAIL no login found for $who"; fails=1; }
done

run_case() { # name expected_exit author changed files...
  local name=$1 want=$2 author=$3 changed=$4; shift 4
  local list out got; list=$(printf '%s\n' "$@")
  out=$(FILES="$list" AUTHOR="$author" CHANGED="$changed" REPO=o/r PR=1 bash -c "
    gh() { printf '%s\n' \"\$FILES\"; }
    $guard" 2>&1); got=$?
  if [ "$got" = "$want" ]; then echo "PASS $name"; else echo "FAIL $name (want $want got $got)"; echo "$out"; fails=1; fi
}

run_case lead-anything          0 "$lead" 2 rdl_engine/gnn.py .github/workflows/ci.yml
run_case m2-clean               0 "$m2" 1 data_pipeline/clean/clean.py
run_case m2-tenk                0 "$m2" 2 database/tenk/tenk_snippets.sql database/tenk/tests/test_load.py
run_case m3-own                 0 "$m3" 1 rag_engine/contrib/rag_search.py
run_case m4-own                 0 "$m4" 1 database/charts/sql_guard.py
run_case m5-own                 0 "$m5" 1 atlas_kv/contrib/samples/triples_sample.csv
run_case m6-own                 0 "$m6" 1 frontend/app.py
run_case m7-own                 0 "$m7" 1 evaluation/safety/run_safety.py
run_case m8-benchmark           0 "$m8" 1 evaluation/benchmark/run_benchmark.py
run_case m8-slides              0 "$m8" 1 docs/presentation/slides.md
run_case space-in-name          0 "$m2" 1 "data_pipeline/clean/my file.py"
run_case outside-folder         1 "$m6" 1 rdl_engine/gnn.py
run_case core-in-same-package   1 "$m2" 1 data_pipeline/contract/validate.py
run_case other-member-folder    1 "$m3" 1 database/charts/sql_guard.py
run_case prefix-lookalike       1 "$m6" 1 frontend_evil/x.py
run_case benchmark-lookalike    1 "$m8" 1 evaluation/benchmark_evil/x.py
run_case docs-outside-slides    1 "$m8" 1 docs/contracts.md
run_case pyproject-edit         1 "$m4" 1 pyproject.toml
run_case engine-api-edit        1 "$m3" 1 rag_engine/api.py
run_case nested-agents-md       1 "$m6" 1 frontend/AGENTS.md
run_case codeowners-root        1 "$m6" 1 CODEOWNERS
run_case workflow-edit          1 "$m6" 1 .github/workflows/path-guard.yml
run_case rename-out-of-core     1 "$m6" 1 frontend/gnn.py rdl_engine/gnn.py
run_case unknown-author         1 stranger 1 frontend/app.py
run_case placeholder-literal    1 M6_LOGIN_X 1 frontend/app.py
run_case too-many-files         1 "$m6" 3000 frontend/app.py
run_case wildcard-filename      1 "$m6" 1 "*"
run_case cmd-injection-name     1 "$m6" 1 "::error::fake"
exit $fails
