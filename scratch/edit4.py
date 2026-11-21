import re
p = 'differometor/build.py'
s = open(p, encoding='utf-8').read()
s = s.replace("jnp.array(", "_arr(")
helper = '''

def _arr(values, dtype=None):
    """`jnp.array` without the XLA compile: `jnp.array(host_array, dtype=...)` compiles a separate
    `convert_element_type` executable for every distinct shape/dtype (~1 s of first-run time in
    total), so convert on the host with NumPy and only transfer."""
    return jnp.asarray(np.asarray(values, dtype=dtype))

'''
s = s.replace("# ---------------------------------------------------------------------\n# Global state for parameter linking", helper.lstrip("\n") + "\n# ---------------------------------------------------------------------\n# Global state for parameter linking", 1)
# host-side indexing of the parameter row in prepare_arrays
s = s.replace("_arr(parameters[0][optimized_parameter_indices])", "_arr(np.asarray(parameters)[0][np.asarray(optimized_parameter_indices)])")
s = s.replace("_arr([parameters[0][carrier_changing_parameter_indices]])", "_arr([np.asarray(parameters)[0][np.asarray(carrier_changing_parameter_indices)]])")
s = s.replace("_arr([parameters[0][signal_changing_parameter_indices]])", "_arr([np.asarray(parameters)[0][np.asarray(signal_changing_parameter_indices)]])")
open(p, 'w', encoding='utf-8').write(s)
print(s.count("_arr("))
