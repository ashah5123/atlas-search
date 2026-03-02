#!/usr/bin/env bash
#
# Download MS MARCO Passage Ranking dataset into data/raw/msmarco_passage/.
# Fetches: collection.tsv, queries.train.tsv, queries.dev.small.tsv,
#          qrels.train.tsv, qrels.dev.small.tsv from official Microsoft URLs.
# Idempotent: skips existing files. Verifies all files exist and are non-empty.
#
set -euo pipefail

BASE_URL="https://msmarco.z22.web.core.windows.net/msmarcoranking"
DIR="data/raw/msmarco_passage"
TMP_DIR="${TMPDIR:-/tmp}/msmarco_download_$$"

mkdir -p "$DIR"
trap 'rm -rf "$TMP_DIR"' EXIT

download_if_missing() {
  local dest="$1"
  local url="$2"
  if [[ -s "$dest" ]]; then
    echo "Skipping (exists): $dest"
    return 0
  fi
  echo "Downloading: $url -> $dest"
  curl -L -o "$dest" -- "$url"
}

# collection.tsv, queries.dev.small.tsv, qrels.dev.small.tsv from collectionandqueries.tar.gz
need_caq=false
for f in collection.tsv queries.dev.small.tsv qrels.dev.small.tsv; do
  [[ -s "$DIR/$f" ]] || need_caq=true
done
if [[ "$need_caq" == true ]]; then
  mkdir -p "$TMP_DIR"
  archive="$TMP_DIR/collectionandqueries.tar.gz"
  if [[ ! -s "$archive" ]]; then
    echo "Downloading collectionandqueries.tar.gz (large, ~2.9GB)..."
    curl -L -o "$archive" -- "$BASE_URL/collectionandqueries.tar.gz"
  fi
  for member in collection.tsv queries.dev.small.tsv qrels.dev.small.tsv; do
    if [[ -s "$DIR/$member" ]]; then
      echo "Skipping (exists): $DIR/$member"
    else
      echo "Extracting $member..."
      tar -xzf "$archive" -C "$TMP_DIR" "$member" 2>/dev/null || true
      if [[ -f "$TMP_DIR/$member" ]]; then
        mv "$TMP_DIR/$member" "$DIR/$member"
      else
        # list archive to find exact path
        exact=$(tar -tzf "$archive" | grep -E "(^|/)$(basename "$member")$" | head -1 | tr -d '\r')
        if [[ -n "$exact" ]]; then
          tar -xzf "$archive" -C "$TMP_DIR" "$exact"
          if [[ -f "$TMP_DIR/$exact" ]]; then
            mv "$TMP_DIR/$exact" "$DIR/$member"
          else
            mv "$TMP_DIR/$(basename "$exact")" "$DIR/$member"
          fi
        else
          echo "ERROR: $member not found in collectionandqueries.tar.gz" >&2
          exit 1
        fi
      fi
    fi
  done
fi

# queries.train.tsv from queries.tar.gz
if [[ ! -s "$DIR/queries.train.tsv" ]]; then
  mkdir -p "$TMP_DIR"
  archive="$TMP_DIR/queries.tar.gz"
  if [[ ! -s "$archive" ]]; then
    echo "Downloading queries.tar.gz..."
    curl -L -o "$archive" -- "$BASE_URL/queries.tar.gz"
  fi
  echo "Extracting queries.train.tsv..."
  tar -xzf "$archive" -C "$TMP_DIR" "queries.train.tsv" 2>/dev/null || true
  if [[ -f "$TMP_DIR/queries.train.tsv" ]]; then
    mv "$TMP_DIR/queries.train.tsv" "$DIR/queries.train.tsv"
  else
    exact=$(tar -tzf "$archive" | grep "queries.train.tsv" | head -1)
    if [[ -n "$exact" ]]; then
      tar -xzf "$archive" -C "$TMP_DIR" "$exact"
      mv "$TMP_DIR/$exact" "$DIR/queries.train.tsv" 2>/dev/null || mv "$TMP_DIR/queries.train.tsv" "$DIR/queries.train.tsv"
    else
      echo "ERROR: queries.train.tsv not found in queries.tar.gz" >&2
      exit 1
    fi
  fi
else
  echo "Skipping (exists): $DIR/queries.train.tsv"
fi

# qrels.train.tsv direct
download_if_missing "$DIR/qrels.train.tsv" "$BASE_URL/qrels.train.tsv"

# qrels.dev.small.tsv: use official qrels.dev.tsv if not from archive
if [[ ! -s "$DIR/qrels.dev.small.tsv" ]]; then
  download_if_missing "$DIR/qrels.dev.small.tsv" "$BASE_URL/qrels.dev.tsv"
fi

# Verify all required files exist and are non-empty
required=(collection.tsv queries.train.tsv queries.dev.small.tsv qrels.train.tsv qrels.dev.small.tsv)
failed=0
for f in "${required[@]}"; do
  path="$DIR/$f"
  if [[ ! -f "$path" ]] || [[ ! -s "$path" ]]; then
    echo "ERROR: Required file missing or empty: $path" >&2
    failed=1
  fi
done
if [[ $failed -eq 1 ]]; then
  exit 1
fi

echo ""
echo "All files present and non-empty. Summary:"
ls -lh "$DIR"
