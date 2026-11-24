#!/usr/bin/env bash
# Phase 6 runs, one at a time on the GPU (WSL, ~/ic2env). A run whose summary.json exists is skipped,
# so the queue can be started again after an interruption (a half-done run starts over).
set -u
cd "$(dirname "$0")/.."
source ~/ic2env/bin/activate
LOG=runs/phase6_queue.log
run() {  # name config val seed
  local name=$1 cfg=$2 val=$3 seed=$4
  if [ -f "runs/$name/summary.json" ]; then echo "$(date +%T) skip $name" >>"$LOG"; return; fi
  rm -rf "runs/$name"
  echo "$(date +%T) start $name" >>"$LOG"
  python scripts/train_phase2.py --config "$cfg" --val "$val" --run-name "$name" --set runtime.seed="$seed" \
    >"runs/$name.out" 2>&1
  echo "$(date +%T) end $name (exit $?)" >>"$LOG"
}
VAL=data/provisorio_MOL_n2_stage_c_val20.h5
# item 2 first (the decision on item 3 depends on it), then item 1; item 3 only with "hot400" as argument
for seed in 0 1; do run phase6c_hot_200k_seed$seed configs/phase6_hot_200k.yaml $VAL $seed; done
for seed in 0 1; do run phase6c_400k_seed$seed configs/phase6_stage_c_400k.yaml $VAL $seed; done
if [ "${1:-}" = "hot400" ]; then
  for seed in 0 1; do run phase6c_hot_400k_seed$seed configs/phase6_hot_400k.yaml $VAL $seed; done
fi
echo "$(date +%T) queue done ${1:-}" >>"$LOG"
