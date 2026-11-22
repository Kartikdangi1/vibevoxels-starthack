#!/bin/bash
# Serial validation of final/combined-v2 (c8066b8) vs the official Feyer implementation (83f8d2f).
set -u
W=/root/validation; A=$W/artifacts; T=$W/tools; CORE=20
PO=/root/hk_v2/venv/bin/python          # env for the optimised branch
PR=/root/hk_baseline/venv/bin/python    # env for the official reference (same pins)
mkdir -p $A/{env,build,official,probe,vcase,compare,fd,prebuilt}
pin() { taskset -c $CORE timeout 1800 "$@"; }
step() { echo "=== $(date -Is) $*" | tee -a $A/steps.log; }

step environment
{ date -Is; uname -a; grep PRETTY /etc/os-release; lscpu | grep -E "Model name|Architecture|^CPU\(s\)|Flags" | cut -c1-400
  free -g | head -2; ldd --version | head -1; c++ --version | head -1
  for p in $PO $PR; do echo "--- $p"; $p -c "import sys,jax,jaxlib,numpy,scipy;print(sys.version.split()[0],'jax',jax.__version__,'jaxlib',jaxlib.__version__,'numpy',numpy.__version__,'scipy',scipy.__version__)"; done
  echo "core pinned: $CORE"; env | grep -E "XLA|JAX_|DIFFEROMETOR|_NUM_THREADS" || echo "no XLA/JAX/DIFFEROMETOR/thread env vars"
} > $A/env/environment.txt 2>&1

step "build native library (differometor/native/build.py)"
(cd $W/opt/differometor/native && grep -n "FLAGS\|ARCH_FLAGS" build.py | head -4 && pin $PO build.py) > $A/build/build.log 2>&1; echo "EXIT $?" >> $A/build/build.log
ls -la $W/opt/differometor/native >> $A/build/build.log
(cd $W/opt && $PO -c "from differometor.native import library_path, available; print('available', available(), 'loaded', library_path())") >> $A/build/build.log 2>&1
objdump -p $W/opt/differometor/native/libdfsparse.so | grep NEEDED >> $A/build/build.log

step "official development benchmark, optimised (timing run, unchanged script)"
(cd $W/opt && pin $PO benchmark.py --cpus 1 --repeats 3 --output $A/official/dev_opt.json --save-arrays $A/official/dev_opt.npz) > $A/official/dev_opt.log 2>&1; echo "EXIT $?" >> $A/official/dev_opt.log
step "official aLIGO benchmark, optimised (timing run, unchanged script)"
(cd $W/opt && pin $PO benchmark_aligo.py --cpus 1 --output $A/official/aligo_opt.json --save-arrays $A/official/aligo_opt.npz) > $A/official/aligo_opt.log 2>&1; echo "EXIT $?" >> $A/official/aligo_opt.log

step "backend probe runs (same unchanged scripts, solver entry points counted)"
(cd $W/opt && pin $PO $T/run_official.py $A/probe/dev_opt.json benchmark.py --cpus 1 --repeats 3 --output /tmp/v_dev_probe.json) > $A/probe/dev_opt.log 2>&1; echo "EXIT $?" >> $A/probe/dev_opt.log
(cd $W/opt && pin $PO $T/run_official.py $A/probe/aligo_opt.json benchmark_aligo.py --cpus 1 --repeats 3 --output /tmp/v_aligo_probe.json) > $A/probe/aligo_opt.log 2>&1; echo "EXIT $?" >> $A/probe/aligo_opt.log

step "official reference implementation: development + aLIGO (correctness reference)"
(cd $W/ref && pin $PR benchmark.py --cpus 1 --repeats 3 --output $A/official/dev_ref.json --save-arrays $A/official/dev_ref.npz) > $A/official/dev_ref.log 2>&1; echo "EXIT $?" >> $A/official/dev_ref.log
(cd $W/ref && pin $PR benchmark_aligo.py --cpus 1 --repeats 3 --output $A/official/aligo_ref.json --save-arrays $A/official/aligo_ref.npz) > $A/official/aligo_ref.log 2>&1; echo "EXIT $?" >> $A/official/aligo_ref.log

step "prebuilt-only install (no compile step): library moved out, same scripts"
rm -rf $W/opt_prebuilt && cp -r $W/opt $W/opt_prebuilt && rm -f $W/opt_prebuilt/differometor/native/libdfsparse.so && (cd $W/opt_prebuilt && git add -A >/dev/null 2>&1; git -c user.name=v -c user.email=v@l commit -qm prebuilt >/dev/null 2>&1)
(cd $W/opt_prebuilt && $PO -c "from differometor.native import library_path; print('loaded', library_path())") > $A/prebuilt/loaded.txt 2>&1
(cd $W/opt_prebuilt && pin $PO benchmark.py --cpus 1 --repeats 3 --output $A/prebuilt/dev.json) > $A/prebuilt/dev.log 2>&1; echo "EXIT $?" >> $A/prebuilt/dev.log
(cd $W/opt_prebuilt && pin $PO benchmark_aligo.py --cpus 1 --output $A/prebuilt/aligo.json) > $A/prebuilt/aligo.log 2>&1; echo "EXIT $?" >> $A/prebuilt/aligo.log
(cd $W/opt_prebuilt && pin $PO $T/run_official.py $A/prebuilt/aligo_probe.json benchmark_aligo.py --cpus 1 --repeats 3 --output /tmp/v_pb_probe.json) > $A/prebuilt/aligo_probe.log 2>&1

step "independent full outputs + gradients + finite differences (vcase)"
for c in aligo dev uifo20 voyager_fm voyager_am voyager_space; do
  (cd $W/opt && pin $PO $T/vcase.py run $c $A/vcase/${c}_opt.npz --jitter --fd $A/fd/${c}_opt_fd.json) > $A/vcase/${c}_opt.log 2>&1; echo "EXIT $?" >> $A/vcase/${c}_opt.log
  (cd $W/ref && pin $PR $T/vcase.py run $c $A/vcase/${c}_ref.npz --jitter) > $A/vcase/${c}_ref.log 2>&1; echo "EXIT $?" >> $A/vcase/${c}_ref.log
done

step "array comparisons"
$PO $T/vcompare.py $A/official/dev_opt.npz $A/official/dev_ref.npz "official dev benchmark arrays" > $A/compare/dev_official.json 2>&1
$PO $T/vcompare.py $A/official/aligo_opt.npz $A/official/aligo_ref.npz "official aLIGO benchmark arrays" > $A/compare/aligo_official.json 2>&1
for c in aligo dev uifo20 voyager_fm voyager_am voyager_space; do $PO $T/vcompare.py $A/vcase/${c}_opt.npz $A/vcase/${c}_ref.npz "vcase $c" > $A/compare/${c}_vcase.json 2>&1; done

step "native solver vs SuperLU vs JAX on 12 topologies (tests/check_native_solver.py)"
(cd $W/opt && pin $PO tests/check_native_solver.py) > $A/check_native_solver.log 2>&1; echo "EXIT $?" >> $A/check_native_solver.log

step done
echo DONE > $W/DONE
