#!/bin/bash
# Rerun the steps that used the (since fixed) probe/vcase tools. Official benchmark runs are unaffected.
set -u
W=/root/validation; A=$W/artifacts; T=$W/tools; CORE=20
PO=/root/hk_v2/venv/bin/python; PR=/root/hk_baseline/venv/bin/python
pin() { taskset -c $CORE timeout 3600 "$@"; }
step() { echo "=== $(date -Is) $*" | tee -a $A/steps.log; }
while pgrep -f run_validation.sh >/dev/null; do sleep 10; done
rm -f $W/DONE2
step "RERUN backend probes"
(cd $W/opt && pin $PO $T/run_official.py $A/probe/dev_opt.json benchmark.py --cpus 1 --repeats 3 --output /tmp/v_dev_probe.json) > $A/probe/dev_opt.log 2>&1; echo "EXIT $?" >> $A/probe/dev_opt.log
(cd $W/opt && pin $PO $T/run_official.py $A/probe/aligo_opt.json benchmark_aligo.py --cpus 1 --repeats 3 --output /tmp/v_aligo_probe.json) > $A/probe/aligo_opt.log 2>&1; echo "EXIT $?" >> $A/probe/aligo_opt.log
(cd $W/opt_prebuilt && pin $PO $T/run_official.py $A/prebuilt/aligo_probe.json benchmark_aligo.py --cpus 1 --repeats 3 --output /tmp/v_pb_probe.json) > $A/prebuilt/aligo_probe.log 2>&1; echo "EXIT $?" >> $A/prebuilt/aligo_probe.log
(cd $W/opt_prebuilt && pin $PO $T/run_official.py $A/prebuilt/dev_probe.json benchmark.py --cpus 1 --repeats 3 --output /tmp/v_pbd_probe.json) > $A/prebuilt/dev_probe.log 2>&1; echo "EXIT $?" >> $A/prebuilt/dev_probe.log
step "RERUN vcase"
for c in aligo dev uifo20 voyager_fm voyager_am voyager_space; do
  (cd $W/opt && pin $PO $T/vcase.py run $c $A/vcase/${c}_opt.npz --jitter --fd $A/fd/${c}_opt_fd.json) > $A/vcase/${c}_opt.log 2>&1; echo "EXIT $?" >> $A/vcase/${c}_opt.log
  (cd $W/ref && pin $PR $T/vcase.py run $c $A/vcase/${c}_ref.npz --jitter) > $A/vcase/${c}_ref.log 2>&1; echo "EXIT $?" >> $A/vcase/${c}_ref.log
done
step "RERUN comparisons"
$PO $T/vcompare.py $A/official/dev_opt.npz $A/official/dev_ref.npz "official dev benchmark arrays" > $A/compare/dev_official.json 2>&1
$PO $T/vcompare.py $A/official/aligo_opt.npz $A/official/aligo_ref.npz "official aLIGO benchmark arrays" > $A/compare/aligo_official.json 2>&1
for c in aligo dev uifo20 voyager_fm voyager_am voyager_space; do $PO $T/vcompare.py $A/vcase/${c}_opt.npz $A/vcase/${c}_ref.npz "vcase $c" > $A/compare/${c}_vcase.json 2>&1; done
step "rerun done"; echo DONE > $W/DONE2
