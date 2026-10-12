#!/usr/bin/env bash
# Fetch the reference PDFs kept in the private papers repository into reference/ (gitignored).
#
# Usage: scripts/fetch_private_papers.sh [repo-url]
#
# The PDFs are third-party papers, so they live in a private repository rather than this
# public one: https://github.com/suv11235/autokernel-PBT-papers (ask the owner for access).
# It mirrors reference/'s layout, so every PDF lands where reference/manifest.csv and the
# READMEs' links expect it. scripts/download_reference_pdfs.sh fetches the arXiv papers on its
# own; this is the only source for the papers that have no arXiv id.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="${1:-${PAPERS_REPO:-https://github.com/suv11235/autokernel-PBT-papers.git}}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

git clone --quiet --depth 1 "$REPO" "$TMP/papers"

n=0
while IFS= read -r -d '' pdf; do
  rel="${pdf#"$TMP/papers/"}"
  dest="$ROOT/reference/$rel"
  mkdir -p "$(dirname "$dest")"
  cp -p "$pdf" "$dest"
  n=$((n + 1))
done < <(find "$TMP/papers" -name '*.pdf' -not -path '*/.git/*' -print0)

echo "copied $n PDFs into reference/ (gitignored)"
