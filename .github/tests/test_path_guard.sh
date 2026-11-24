#!/usr/bin/env bash
# Runs the exact `run:` block of .github/workflows/path-guard.yml against a stubbed `gh`.
# Usage: bash .github/tests/test_path_guard.sh
set -u
here=$(cd "$(dirname "$0")" && pwd)
workflow="$here/../workflows/path-guard.yml"
guard=$(sed -n '/# --- guard start/,/# --- guard end/p' "$workflow" | sed 's/^          //')
[ -n "$guard" ] || { echo "FAIL guard block not found in $workflow"; exit 1; }

# GitHub logins of the lead and of each contributor (M2..M8).
lead=afel6
m2=tsaeedsaifullah-blip
m3=moathkhaled2k5
m4=9yavi
m5=mohammeddf503-beep
m6=tahaelsanfaz
m7=ahmedkarrar742-afk
m8=mdxflame23

fails=0
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
run_case m2-next-to-tenk        1 "$m2" 1 database/queries/late_payments.sql
run_case other-member-folder    1 "$m3" 1 database/charts/sql_guard.py
run_case m4-next-folder         1 "$m4" 1 database/tenk/tenk_snippets.sql
run_case m5-next-folder         1 "$m5" 1 atlas_kv/schema.py
run_case m7-next-folder         1 "$m7" 1 evaluation/benchmark/run_benchmark.py
run_case m8-next-folder         1 "$m8" 1 evaluation/safety/run_safety.py
run_case lead-mixed-case        0 AFEL6 1 rdl_engine/gnn.py
run_case m8-upper-case          0 MDXFLAME23 1 evaluation/benchmark/run_benchmark.py
run_case m3-mixed-case          0 MoathKhaled2k5 1 rag_engine/contrib/rag_search.py
run_case mixed-case-outside     1 MDXFLAME23 1 rdl_engine/gnn.py
run_case login-lookalike        1 9yavi2 1 database/charts/sql_guard.py
run_case prefix-lookalike       1 "$m6" 1 frontend_evil/x.py
run_case benchmark-lookalike    1 "$m8" 1 evaluation/benchmark_evil/x.py
run_case docs-outside-slides    1 "$m8" 1 docs/contracts.md
run_case pyproject-edit         1 "$m4" 1 pyproject.toml
run_case engine-api-edit        1 "$m3" 1 rag_engine/api.py
run_case nested-agents-md       1 "$m6" 1 frontend/AGENTS.md
run_case nested-claude-local    1 "$m6" 1 frontend/CLAUDE.local.md
run_case codeowners-root        1 "$m6" 1 CODEOWNERS
run_case workflow-edit          1 "$m6" 1 .github/workflows/path-guard.yml
run_case rename-out-of-core     1 "$m6" 1 frontend/gnn.py rdl_engine/gnn.py
run_case unknown-author         1 stranger 1 frontend/app.py
run_case too-many-files         1 "$m6" 3000 frontend/app.py
run_case wildcard-filename      1 "$m6" 1 "*"
run_case cmd-injection-name     1 "$m6" 1 "::error::fake"
exit $fails
