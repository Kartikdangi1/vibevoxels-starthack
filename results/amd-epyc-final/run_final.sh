#!/bin/bash
# One benchmark session for every variant: same machine, same core, interleaved rounds.
OUT=/root/final_bench; mkdir -p $OUT; rm -f /root/run_final.done
run() { # name dir venv [env...]
  local name=$1 dir=$2 venv=$3; shift 3
  cd /root/$dir && env "$@" taskset -c 72 /root/$venv/bin/python benchmark.py --cpus 1 --repeats 10 \
    --output $OUT/${name}_$i.json --save-arrays $OUT/${name}_$i.npz > $OUT/${name}_$i.log 2>&1
  echo "EXIT $?" >> $OUT/${name}_$i.log
}
for i in 1 2 3; do
  run baseline    hk_baseline    hk_baseline/venv
  run cpp         hk_cpp         hk_cpp/venv
  run stage3      hk_stage3      hk_stage3/venv
  run fastreadout hk_fastreadout hk_fastreadout/venv
  run v2_head     hk_head        hk_v2/venv
  run final       hk_final       hk_v2/venv
  run final_fast  hk_final       hk_v2/venv DIFFEROMETOR_XLA_FLAGS=fast
done
echo ALLDONE > /root/run_final.done
