import json, os
os.environ["JAX_PLATFORMS"]="cpu"
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import differometor as df
from differometor.components import FUNCTIONS, evaluate_components
from differometor.setups import constrain_inter_grid_cell_spaces
fx=json.load(open('examples/data/uifo_800_3000.json')); rf=json.load(open('examples/data/uifo_800_3000_test_data.json'))
setup=df.Setup.from_data(fx); pairs=constrain_inter_grid_cell_spaces(setup.parameters,["reflectivity","tuning","db","angle","power","mass","length"])
arrays,*_=df.run_build_step(setup,[("f","frequency")],jnp.asarray(rf['frequencies'])[::25],pairs)
P=jnp.real(arrays["parameters"])[0]
ids=arrays["carrier_function_indices"]; idx=arrays["carrier_function_input_indices"]
rows=P[idx]
J=jax.jacrev(lambda r: jnp.real(evaluate_components(ids,r)))(rows)  # (K,4,K,7)
bad=np.argwhere(~np.isfinite(np.asarray(J)))
print(len(bad), "nan entries"); 
for b in bad[:8]: print("row",b[0],"fid",int(ids[b[0]]),"slot",b[1],"input row",b[2],"col",b[3], "inrow",np.asarray(rows[b[0]]))
print("P[0]",P[0])
