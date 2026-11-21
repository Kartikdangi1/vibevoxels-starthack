import json, os, cProfile, pstats, time
os.environ["JAX_PLATFORMS"]="cpu"
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import differometor as df
from differometor.setups import constrain_inter_grid_cell_spaces
fx=json.load(open('examples/data/uifo_800_3000.json')); rf=json.load(open('examples/data/uifo_800_3000_test_data.json'))
def run():
    setup=df.Setup.from_data(fx); pairs=constrain_inter_grid_cell_spaces(setup.parameters,["reflectivity","tuning","db","angle","power","mass","length"])
    arrays,ports,*_=df.run_build_step(setup,[("f","frequency")],jnp.asarray(rf['frequencies']),pairs)
    jax.block_until_ready(arrays)
pr0=cProfile.Profile(); pr0.enable(); t=time.time(); run(); pr0.disable(); print("first",time.time()-t); pstats.Stats(pr0).sort_stats("cumulative").print_stats(18)
pr=cProfile.Profile(); pr.enable(); t=time.time(); run(); pr.disable(); print("second",time.time()-t)
pstats.Stats(pr).sort_stats("cumulative").print_stats(14)
