#!/usr/bin/env bash
set -Eeuo pipefail
REPO="${REPO:-/data/bohan_zhang/nsco_workspace/nsco-lab}"
RUN="${RUN:-$(cat "$REPO/local/causal_v1_last_run.txt")}"
[ -f "$RUN/RUN_INDEX.json" ] || { echo 'ERROR: no RUN_INDEX.json'; exit 2; }
python - "$RUN/RUN_INDEX.json" <<'PY'
import json,sys
j=json.load(open(sys.argv[1]));assert j['status']=='completed',j
PY
DEST="${DEST:-$REPO/outputs/$(basename "$RUN")_transfer.tar.gz}"
[ ! -e "$DEST" ] || { echo 'ERROR: destination already exists'; exit 2; }
# Include only curated JSON/CSV and logs, never private image data or model weights.
tar -czf "$DEST" --exclude='*.pt' --exclude='*.pth' -C "$(dirname "$RUN")" "$(basename "$RUN")"
sha256sum "$DEST" > "$DEST.sha256"
echo "PACKAGE=$DEST"