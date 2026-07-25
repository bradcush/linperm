#!/usr/bin/env bash
#
# Render one protocol's prove phase-breakdown table, the raw
# numbers written by `cargo bench -p <protocol> --bench phases`.
# Percentages are derived here from the raw ms.
#
# Usage: scripts/prove-phases-table.sh [biperm|prodperm]
set -euo pipefail

proto="${1:-biperm}"
case "$proto" in
biperm | prodperm) ;;
*)
  echo "usage: $(basename "$0") [biperm|prodperm]" >&2
  exit 2
  ;;
esac

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
csv="$root/target/${proto}_prove_phases.csv"

if [[ ! -f $csv ]]; then
  echo "no CSV at $csv; run: cargo bench -p $proto --bench phases" >&2
  exit 1
fi

awk -F, -v proto="$proto" '
BEGIN {
    print proto "_prove phase breakdown\n"
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
