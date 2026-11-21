for i in 1 2 3; do
 for mode in unified sites; do
  r=$(DIFFEROMETOR_DISPATCH=$mode python benchmark.py --cpus 1 --repeats 3 --output scratch/ab.json 2>&1 | grep -E "Forward: compile|Gradient: compile|Complete" | sed 's/.*: //' | tr '\n' ' ')
  echo "$mode $r"
 done
done
