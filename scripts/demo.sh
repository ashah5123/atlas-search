#!/usr/bin/env bash
#
# Demo script for the FastAPI search service running locally.
# Requires the API to already be running at http://127.0.0.1:8000
#
set -euo pipefail

BASE_URL="http://127.0.0.1:8000"

fail_curl() {
  echo "ERROR: Request failed. Is the API running at ${BASE_URL}?" >&2
  echo "Start it with: uvicorn src.serving.app:app --reload" >&2
  exit 1
}

# Quick sanity check
curl -fsS "${BASE_URL}/health" >/dev/null || fail_curl

QUERIES=(
  "how many calories are in a slice of ham"
  "who sang i feel wonderful tonight"
  "what causes rainbows"
)

for q in "${QUERIES[@]}"; do
  echo "================================================================"
  echo "QUERY: ${q}"
  echo "----------------------------------------------------------------"

  body="$(python - "$q" <<'PY'
import json
import sys
query = sys.argv[1]
print(json.dumps({"query": query, "topk": 3, "candidates_k": 100}))
PY
)"

  curl -fsS "${BASE_URL}/search" \
    -H "Content-Type: application/json" \
    -d "$body" \
    | python -m json.tool || fail_curl
done

echo "================================================================"
echo "Done."

