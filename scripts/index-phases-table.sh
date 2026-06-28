#!/usr/bin/env bash
#
# Render the index phase-breakdown table from target/index_phases.csv,
# the raw numbers written by `cargo bench --bench phases`.
# Percentages are derived here from the raw ms.
#
# Usage: scripts/index-phases-table.sh [path-to-csv]
set -euo pipefail

csv="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/target/index_phases.csv}"

if [[ ! -f $csv ]]; then
  echo "no CSV at $csv; run: cargo bench --bench phases" >&2
  exit 1
fi

awk -F, '
BEGIN {
    print "biperm_index phase breakdown\n"
    printf "%-6s %3s   %13s   %13s   %11s\n", \
        "scheme", "mu", "aux_gen", "commit", "total"
}
NR == 1 { next }
{
    total = $5
    paux = total > 0 ? $3 / total * 100 : 0
    pcom = total > 0 ? $4 / total * 100 : 0
    printf "%-6s %3d   %8.3fms %3.0f%%   %8.3fms %3.0f%%   %9.3fms\n", \
        $1, $2, $3, paux, $4, pcom, total
}
' "$csv"
