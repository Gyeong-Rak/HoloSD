#!/usr/bin/env bash
# 10-seed sweep of the adopted HoloSD (holosd_final) on the i4_sepchan goal-nav task.
# Same task/metric as URL/baselines (random/ring/ring_hard suites + R_min sweep, deploy==train).
# Dynamic flock queue across NGPU GPUs; each job trains (Phase A geom -> Phase B SAC) then evals.
# Outputs -> exp/bicycle/HoloSD/seeds/s<seed>/{ckpt_final.pkl, train_log.jsonl, eval/metrics.json}
# Logs    -> _trash/holosd_seedrun/logs/ ; resume-safe (skips seeds whose eval/metrics.json exists).
set -u
cd "$(dirname "$0")/.."                       # -> URL/HoloSD
VENVPY=.venv/bin/python

NGPU="${NGPU:-8}"
NSEED="${NSEED:-10}"
RUNDIR=/media/satassd/grchoe/_trash/holosd_seedrun
LOGDIR="$RUNDIR/logs"
QUEUE="$RUNDIR/queue.txt"
LOCK="$RUNDIR/queue.lock"
PROG="$RUNDIR/progress.log"
CACHE=/media/satassd/grchoe/.cache/jax
mkdir -p "$LOGDIR" "$CACHE"

# build queue (one seed per line); RESUME: skip seeds already evaluated
: > "$QUEUE"
for ((s=0; s<NSEED; s++)); do
  [ -f "exp/bicycle/HoloSD/seeds/s$s/eval/metrics.json" ] && continue
  echo "$s" >> "$QUEUE"
done
TOTAL=$(wc -l < "$QUEUE")
echo "[holosd_sweep] $TOTAL jobs over $NGPU GPUs | start $(date +%F_%H:%M:%S)" | tee "$PROG"

pop_job() {            # atomic pop of the first queue line -> stdout
  ( flock -x 200
    job=$(head -n1 "$QUEUE" 2>/dev/null)
    [ -n "$job" ] && sed -i '1d' "$QUEUE"
    printf '%s' "$job"
  ) 200>"$LOCK"
}

worker() {
  local gpu=$1
  export HOLOSD_GPU=$gpu                      # holosd_final._gpu() sets CUDA_VISIBLE_DEVICES=$HOLOSD_GPU
  export JAX_PLATFORMS=cuda
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
  export JAX_COMPILATION_CACHE_DIR="$CACHE/gpu$gpu"
  export MPLBACKEND=Agg
  mkdir -p "$JAX_COMPILATION_CACHE_DIR"
  while true; do
    local s; s=$(pop_job)
    [ -z "$s" ] && break
    local out="exp/bicycle/HoloSD/seeds/s$s"
    mkdir -p "$out"
    echo "[gpu$gpu] START s$s $(date +%H:%M:%S)" >> "$PROG"
    $VENVPY -m holosd.bicycle.holosd_final train --out_dir "$out" --seed "$s" \
        --exp_name "HoloSD_s$s" > "$LOGDIR/s${s}.log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
      $VENVPY -m holosd.bicycle.holosd_final eval "$out/ckpt_final.pkl" --T 300 \
          >> "$LOGDIR/s${s}.log" 2>&1
      rc=$?
    fi
    local r=$($VENVPY -c "import json;print('%.0f%%'%(json.load(open('$out/eval/metrics.json'))['avg']['reach']*100))" 2>/dev/null || echo ERR)
    echo "[gpu$gpu] DONE  s$s rc=$rc avg_reach=$r $(date +%H:%M:%S)" >> "$PROG"
  done
}

for ((g=0; g<NGPU; g++)); do worker "$g" & done
wait
echo "[holosd_sweep] all jobs done $(date +%F_%H:%M:%S)" | tee -a "$PROG"
$VENVPY scripts/agg_holosd.py 2>&1 | tee -a "$PROG"
echo "[holosd_sweep] FINISHED $(date +%F_%H:%M:%S)" | tee -a "$PROG"
