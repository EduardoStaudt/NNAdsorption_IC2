#!/usr/bin/env bash
# Phase 5 runs, one at a time on the GPU (WSL, ~/ic2env). A run whose summary.json exists is skipped,
# so the queue can be started again after an interruption (a half-done run starts over).
set -u
cd "$(dirname "$0")/.."
source ~/ic2env/bin/activate
LOG=runs/phase5_queue.log
run() {  # name config val seed
  local name=$1 cfg=$2 val=$3 seed=$4
  if [ -f "runs/$name/summary.json" ]; then echo "$(date +%T) skip $name" >>"$LOG"; return; fi
  rm -rf "runs/$name"
  echo "$(date +%T) start $name" >>"$LOG"
  python scripts/train_phase2.py --config "$cfg" --val "$val" --run-name "$name" --set runtime.seed="$seed" \
    >"runs/$name.out" 2>&1
  echo "$(date +%T) end $name (exit $?)" >>"$LOG"
}
for seed in 0 1; do run phase5a_seed$seed configs/phase5_stage_a.yaml data/provisorio_MOL_n2_stage_a_val20.h5 $seed; done
for seed in 0 1; do run phase5b_cheap_seed$seed configs/phase5_cheap_b.yaml data/provisorio_MOL_n2_stage_b_val20.h5 $seed; done
for seed in 0 1; do run phase5c_cheap_seed$seed configs/phase5_cheap_c.yaml data/provisorio_MOL_n2_stage_c_val20.h5 $seed; done
for seed in 0 1; do run phase5c_long_seed$seed configs/phase5_stage_c_long.yaml data/provisorio_MOL_n2_stage_c_val20.h5 $seed; done
echo "$(date +%T) queue done" >>"$LOG"
