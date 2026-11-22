#!/bin/bash
# 20k-frequency sweeps (10 Hz - 20 kHz, log-spaced) on the shipped code (c8066b8, prebuilt solver,
# official scripts); the official code on a 1-in-200 sample of the same grid. One pinned core.
W=/root/validation; O=$W/scale; T=$W/tools; PO=/root/hk_v2/venv/bin/python; PR=/root/hk_baseline/venv/bin/python
mkdir -p $O
run() { local name=$1 dir=$2 py=$3; shift 3
  (cd $dir && taskset -c 20 timeout 5400 $py $T/scale.py "$@") > $O/$name.log 2>&1; echo "EXIT $?" >> $O/$name.log; }
run dev_1k        $W/opt_prebuilt $PO dev   1000  10 20000 $O/dev_1k
run dev_5k        $W/opt_prebuilt $PO dev   5000  10 20000 $O/dev_5k
run dev_20k       $W/opt_prebuilt $PO dev   20000 10 20000 $O/dev_20k
run aligo_20k     $W/opt_prebuilt $PO aligo 20000 10 20000 $O/aligo_20k
run dev_sub_opt   $W/opt_prebuilt $PO dev   20000 10 20000 $O/dev_sub_opt   --subset 200
run aligo_sub_opt $W/opt_prebuilt $PO aligo 20000 10 20000 $O/aligo_sub_opt --subset 200
run dev_sub_ref   $W/ref          $PR dev   20000 10 20000 $O/dev_sub_ref   --subset 200
run aligo_sub_ref $W/ref          $PR aligo 20000 10 20000 $O/aligo_sub_ref --subset 200
$PO $T/scale_compare.py > $O/comparison.txt
