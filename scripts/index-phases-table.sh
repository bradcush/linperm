#!/usr/bin/env bash
#
# Render one protocol's index phase-breakdown table, the raw
# numbers written by `cargo bench -p <protocol> --bench phases`.
# Percentages are derived here from the raw ms.
#
# Usage: scripts/index-phases-table.sh [biperm|prodperm]
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
csv="$root/target/${proto}_index_phases.csv"

if [[ ! -f $csv ]]; then
  echo "no CSV at $csv; run: cargo bench -p $proto --bench phases" >&2
  exit 1
fi

awk -F, -v proto="$proto" '
BEGIN {
    print proto "_index phase breakdown\n"
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
