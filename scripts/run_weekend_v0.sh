#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/outputs/bnci2014_001_baseline_trajectory_identifiability_v0"
LOG_DIR="$OUT/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/weekend_v0_${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1
cd "$ROOT"

failure() {
  code=$?
  python - "$OUT" "$code" "$LOG" <<'PY'
import json, os, sys
from pathlib import Path
out, code, log = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
p = out / "failure.json.partial"
p.write_text(json.dumps({"exit_code": code, "log": log}, indent=2) + "\n")
os.replace(p, out / "failure.json")
PY
  exit "$code"
}
trap failure ERR

echo "PID=$$ LOG=$LOG START=$(date -Iseconds)"
python scripts/60_freeze_protocol_v0.py
python -m pytest -q tests/test_baseline_trajectory_geometry_v0.py tests/test_baseline_trajectory_action_v0.py
test -f "$OUT/data_timing_audit.json"
python scripts/60_phase0_task_v0.py --resume
python scripts/60_phase0_identifiability_v0.py --resume
python scripts/60_phase0_action_v0.py --resume
python scripts/60_decide_v0.py
python - "$OUT" <<'PY'
import json, subprocess, sys
from pathlib import Path
out = Path(sys.argv[1])
gates = json.loads((out / "decisions/gates.json").read_text())
if gates["conditional_gru"]["run"]:
    subprocess.run([sys.executable, "scripts/60_phase1_model_v0.py", "--resume"], check=True)
PY
python -m pytest -q tests/test_baseline_trajectory_geometry_v0.py tests/test_baseline_trajectory_action_v0.py
git diff --check
if git status --short | awk '{print $2}' | grep -E '^(cache/|.*\.mat$|.*\.npz$)' >/dev/null; then
  echo "forbidden raw/cache staging candidate detected" >&2
  exit 90
fi
echo "COMPLETE $(date -Iseconds)"
