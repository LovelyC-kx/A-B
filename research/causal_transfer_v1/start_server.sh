#!/usr/bin/env bash
set -Eeuo pipefail
# Run from any directory; no destructive operations, no automatic git pull or pip install.
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-/data/bohan_zhang/nsco_workspace/nsco-lab}"
EXPERIMENT="${EXPERIMENT:-}"
GPU_A="${GPU_A:-}"
GPU_B="${GPU_B:-}"
VAR_A="${VAR_A:-N0_nafnet}"
VAR_B="${VAR_B:-N4_cross_extra}"
CHECKPOINT="${CHECKPOINT:-best}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
[ -n "$EXPERIMENT" ] && [ -d "$EXPERIMENT" ] || { echo 'ERROR: set EXPERIMENT to existing frozen formal run'; exit 2; }
[ "$VAR_A" != "$VAR_B" ] || { echo "ERROR: VAR_A and VAR_B must differ"; exit 2; }
[ -n "$GPU_A" ] && [ -n "$GPU_B" ] && [ "$GPU_A" != "$GPU_B" ] || { echo 'ERROR: set distinct GPU_A and GPU_B (GPU index or UUID)'; exit 2; }
[ -f "$REPO/formal/p0_gain_sweep.py" ] || { echo 'ERROR: p0_gain_sweep.py missing in repo formal/'; exit 2; }
[ -f "$EXPERIMENT/frozen_config.json" ] && [ -f "$EXPERIMENT/snapshot_hash.json" ] || { echo 'ERROR: immutable run metadata missing'; exit 2; }
PYTHON="${PYTHON:-python}"
"$PYTHON" "$PKG/test_cpu.py" -v
"$PYTHON" - <<'PY'
import torch
assert torch.cuda.is_available(),'CUDA unavailable'
print('PyTorch/CUDA:',torch.__version__,torch.version.cuda,'GPU count',torch.cuda.device_count())
PY
nvidia-smi -i "$GPU_A" --query-gpu=index,uuid,name,memory.free --format=csv,noheader
nvidia-smi -i "$GPU_B" --query-gpu=index,uuid,name,memory.free --format=csv,noheader
ROOT="${ROOT:-$REPO/outputs/causal_v1_$RUN_ID}"
[ ! -e "$ROOT" ] || { echo "ERROR: output already exists: $ROOT"; exit 2; }
mkdir -p "$ROOT"
printf '%s\n' "$ROOT" > "$REPO/local/causal_v1_last_run.txt"
printf '%s\n' "REPO=$REPO" "EXPERIMENT=$EXPERIMENT" "VAR_A=$VAR_A" "VAR_B=$VAR_B" "GPU_A=$GPU_A" "GPU_B=$GPU_B" > "$ROOT/run_env.txt"
EXTRA=(--crop "${CROP:-128}" --train-images "${TRAIN_IMAGES:-6}" --pilot-images "${PILOT_IMAGES:-6}" --audit-images "${AUDIT_IMAGES:-6}" --eta "${ETA:-0.003}")
set +e
CUDA_VISIBLE_DEVICES="$GPU_A" "$PYTHON" -u "$PKG/causal_probe.py" --repo "$REPO" --experiment "$EXPERIMENT" --variant "$VAR_A" --checkpoint "$CHECKPOINT" --out "$ROOT/$VAR_A" --device cuda:0 --trust-own-checkpoints "${EXTRA[@]}" > "$ROOT/$VAR_A.log" 2>&1 & PID_A=$!
CUDA_VISIBLE_DEVICES="$GPU_B" "$PYTHON" -u "$PKG/causal_probe.py" --repo "$REPO" --experiment "$EXPERIMENT" --variant "$VAR_B" --checkpoint "$CHECKPOINT" --out "$ROOT/$VAR_B" --device cuda:0 --trust-own-checkpoints "${EXTRA[@]}" > "$ROOT/$VAR_B.log" 2>&1 & PID_B=$!
printf '%s\n' "PID_A=$PID_A" "PID_B=$PID_B" > "$ROOT/pids.txt"
wait "$PID_A"; STATUS_A=$?
wait "$PID_B"; STATUS_B=$?
set -e
"$PYTHON" - "$ROOT" "$VAR_A" "$VAR_B" "$STATUS_A" "$STATUS_B" <<'PY'
import json,sys,pathlib
root=pathlib.Path(sys.argv[1]); vs=sys.argv[2:4]; codes=list(map(int,sys.argv[4:]))
status={'status':'completed' if all(c==0 and (root/v/'result.json').is_file() for v,c in zip(vs,codes)) else 'failed','variants':dict(zip(vs,codes)),'root':str(root)}
(root/'RUN_INDEX.json').write_text(json.dumps(status,indent=2)+'\n')
print(status)
PY
[ "$STATUS_A" -eq 0 ] && [ "$STATUS_B" -eq 0 ] || { echo "ERROR: inspect $ROOT/*.log"; exit 1; }
echo "SUCCESS: $ROOT"