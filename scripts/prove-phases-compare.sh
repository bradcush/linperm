#!/usr/bin/env bash
#
# Side-by-side BiPerm vs ProdPerm `prove` phase breakdown, from the CSVs.
# Written by `cargo bench -p {biperm,prodperm} --bench phases`. Percentages
# live in the per-protocol tables; this one shows ratios. Reads what's
# on disk; `--run` regenerates both back-to-back first
#
# Usage: scripts/prove-phases-compare.sh [real|mock] [--run]
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode=real
run=0

for arg in "$@"; do
  case "$arg" in
  real | mock) mode=$arg ;;
  --run) run=1 ;;
  *)
    echo "usage: $(basename "$0") [real|mock] [--run]" >&2
    exit 2
    ;;
  esac
done

if [[ $mode == real ]]; then
  bi_scheme=shyrax
  # Dense because there's no
  # benefit to using a sparse one
  pr_scheme=hyrax
else
  bi_scheme=mock
  pr_scheme=mock
fi

bi_csv="$root/target/biperm_prove_phases.csv"
pr_csv="$root/target/prodperm_prove_phases.csv"

if ((run)); then
  # Back-to-back so both protocols see the same machine state.
  # Each bench writes its index and prove CSVs together.
  echo "regenerating both phase breakdowns..." >&2
  cargo bench -p biperm --bench phases >/dev/null
  cargo bench -p prodperm --bench phases >/dev/null
fi

for csv in "$bi_csv" "$pr_csv"; do
  if [[ ! -f $csv ]]; then
    echo "no CSV at $csv; re-run with --run" >&2
    exit 1
  fi
done

# Numbers taken far apart aren't same-session; say so
# rather than silently rendering them as if they were.
# Arbitrarily chosen 60 minutes for the threshold.
skew=$(($(stat -c %Y "$bi_csv") - $(stat -c %Y "$pr_csv")))
skew=${skew#-}
if ((skew > 3600)); then
  printf 'warning: CSVs written %dh%02dm apart; re-run with --run\n\n' \
    $((skew / 3600)) $((skew % 3600 / 60)) >&2
fi

# Both files are keyed by (scheme, mu). Rows for the chosen scheme are
# pulled from each and joined on mu, in file order (already ascending).
awk -F, -v s1="$bi_scheme" -v s2="$pr_scheme" '
BEGIN {
    printf "prove: biperm/%s vs prodperm/%s\n\n", s1, s2
    printf "%4s  %-9s%12s%12s%12s%12s%12s\n", \
        "mu", "protocol", "commit", "aux", "sumcheck", "opens", "total"
}
FNR == 1 { next }
FILENAME == ARGV[1] && $1 == s1 {
    if (!($2 in seen)) { seen[$2] = 1; order[++n] = $2 }
    for (i = 3; i <= 7; i++) a[$2, i] = $i
}
FILENAME == ARGV[2] && $1 == s2 { for (i = 3; i <= 7; i++) b[$2, i] = $i }
END {
    for (k = 1; k <= n; k++) {
        mu = order[k]
        if (!((mu, 3) in b)) continue
        printf "%4d  %-9s", mu, "biperm"
        for (i = 3; i <= 7; i++) printf "%10.3fms", a[mu, i]
        printf "\n%4d  %-9s", mu, "prodperm"
        for (i = 3; i <= 7; i++) printf "%10.3fms", b[mu, i]
        printf "\n%4d  %-9s", mu, "ratio"
        for (i = 3; i <= 7; i++) {
            # A zero baseline (mock commit at small mu) has no ratio.
            if (a[mu, i] + 0 == 0) printf "%12s", "n/a"
            else printf "%11.2fx", b[mu, i] / a[mu, i]
        }
        printf "\n\n"
    }
}
' "$bi_csv" "$pr_csv"
