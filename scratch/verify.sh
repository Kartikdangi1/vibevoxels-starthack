cd /work
echo "== pytest (default)"; python -m pytest tests -q 2>&1 | tail -3
for m in sites full; do echo "== pytest example setups, DIFFEROMETOR_DISPATCH=$m"; DIFFEROMETOR_DISPATCH=$m python -m pytest tests/test_example_setups.py tests/test_sparse_pipeline.py -q -k "not self_check" 2>&1 | tail -3; done
echo "== start_here"; python start_here.py 2>&1 | tail -6
for f in cavity aligo voyager uifo; do echo "== examples/$f.py"; MPLBACKEND=Agg python examples/$f.py > /tmp/$f.log 2>&1; echo "exit $?"; tail -2 /tmp/$f.log; done
echo DONE
