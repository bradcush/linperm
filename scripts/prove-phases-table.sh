#!/usr/bin/env bash
#
# Render the prove phase-breakdown table from target/prove_phases.csv,
# the raw numbers written by `cargo bench --bench phases`.
# Percentages are derived here from the raw ms.
#
# Usage: scripts/prove-phases-table.sh [path-to-csv]
set -euo pipefail

csv="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/target/prove_phases.csv}"

if [[ ! -f $csv ]]; then
  echo "no CSV at $csv; run: cargo bench --bench phases" >&2
  exit 1
fi

awk -F, '
BEGIN {
    print "biperm_prove phase breakdown\n"
    printf "%-6s %3s   %13s   %13s   %13s   %13s   %11s\n", \
        "scheme", "mu", "commit", "aux", "sumcheck", "opens", "total"
}
NR == 1 { next }
{
    total = $7
    pc = total > 0 ? $3 / total * 100 : 0
    pa = total > 0 ? $4 / total * 100 : 0
    ps = total > 0 ? $5 / total * 100 : 0
    po = total > 0 ? $6 / total * 100 : 0
    printf "%-6s %3d   %8.3fms %3.0f%%   %8.3fms %3.0f%%   %8.3fms %3.0f%%   %8.3fms %3.0f%%   %9.3fms\n", \
        $1, $2, $3, pc, $4, pa, $5, ps, $6, po, total
}
' "$csv"
