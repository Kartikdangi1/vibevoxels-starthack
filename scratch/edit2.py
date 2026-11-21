p = 'differometor/simulate.py'
s = open(p, encoding='utf-8').read()


def rep(old, new):
    global s
    assert s.count(old) >= 1, old[:70]
    s = s.replace(old, new, 1)


rep("def _last_wins(", '''def scatter_outputs(
    output_matrix: jnp.ndarray,
    matrix: jnp.ndarray,
    output_indices: tuple[jnp.ndarray, jnp.ndarray],
    matrix_indices: tuple[jnp.ndarray, jnp.ndarray],
) -> jnp.ndarray:
    """Write already evaluated component outputs `output_matrix` (K, 4) into `matrix` (the scatter
    half of `update`)."""
    outputs = output_matrix[output_indices[0], output_indices[1]]
    return matrix.at[matrix_indices[0], matrix_indices[1]].set(outputs)


def _component_outputs(
    carrier_parameters, signal_parameters,
    carrier_function_input_indices, carrier_function_indices,
    signal_function_input_indices, signal_function_indices,
    noise_function_input_indices, noise_function_indices,
):
    """
    Evaluate every component function of the three systems: carrier (CV, Kc, 4), signal (SV, Ks, 4)
    and noise (Kn, 4; frequency independent, so evaluated on the first parameter row only).

    In the default unified mode this is ONE flat call of the branch-free kernel over all rows of all
    three systems: XLA then compiles (and differentiates) the kernel once instead of once per
    system, which is most of the compile time. Other dispatch modes evaluate per system.
    """
    carrier_rows = carrier_parameters[:, carrier_function_input_indices]  # (CV, Kc, 7)
    signal_rows = signal_parameters[:, signal_function_input_indices]  # (SV, Ks, 7)
    noise_rows = signal_parameters[0][noise_function_input_indices]  # (Kn, 7)
    if _DISPATCH_MODE != "unified":
        per_system = lambda dispatch, ids, rows: vmap(lambda r: _evaluate(dispatch, ids, r))(rows)
        return (
            per_system(_CARRIER_DISPATCH, carrier_function_indices, carrier_rows),
            per_system(_SIGNAL_DISPATCH, signal_function_indices, signal_rows),
            _evaluate(_NOISE_DISPATCH, noise_function_indices, noise_rows),
        )
    n_carrier = carrier_rows.shape[0] * carrier_rows.shape[1]
    n_signal = signal_rows.shape[0] * signal_rows.shape[1]
    outputs = evaluate_components(
        jnp.concatenate(
            [
                jnp.tile(carrier_function_indices, carrier_rows.shape[0]),
                jnp.tile(signal_function_indices, signal_rows.shape[0]),
                noise_function_indices,
            ]
        ),
        jnp.concatenate([carrier_rows.reshape(-1, 7), signal_rows.reshape(-1, 7), noise_rows]),
    )
    return (
        outputs[:n_carrier].reshape(carrier_rows.shape[:2] + (4,)),
        outputs[n_carrier : n_carrier + n_signal].reshape(signal_rows.shape[:2] + (4,)),
        outputs[n_carrier + n_signal :],
    )


def _last_wins(''')

rep('''def _signal_and_noise_sparse(
    parameters, carrier, signal_matrix, noise_selection_vectors, qhd_placing_indices, qhd_phase_values,
    signal_function_input_indices, signal_function_indices, signal_output_indices, signal_matrix_indices,
    signal_carrier_indices, signal_carrier_placing_indices,
    noise_function_input_indices, noise_function_indices, noise_output_indices, noise_system_matrix_indices,
):''', '''def _signal_and_noise_sparse(
    signal_outputs, noise_outputs, carrier, signal_matrix, noise_selection_vectors, qhd_placing_indices,
    qhd_phase_values, signal_output_indices, signal_matrix_indices, signal_carrier_indices,
    signal_carrier_placing_indices, noise_output_indices, noise_system_matrix_indices,
):''')
rep('''    vals = vmap(update_values, in_axes=(0, None, None, None, None))(
        parameters, signal_function_input_indices, signal_output_indices, signal_function_indices,
        _SIGNAL_DISPATCH,
    )''', '''    vals = signal_outputs[:, signal_output_indices[0], signal_output_indices[1]]''')
rep('''    noise_values = update_values(
        parameters[0], noise_function_input_indices, noise_output_indices, noise_function_indices, _NOISE_DISPATCH
    )''', '''    noise_values = noise_outputs[noise_output_indices[0], noise_output_indices[1]]''')

rep('''def _signal_and_noise_dense(
    parameters, carrier, signal_matrix, noise_matrix, noise_selection_vectors, qhd_placing_indices,
    qhd_phase_values, signal_function_input_indices, signal_function_indices, signal_output_indices,
    signal_matrix_indices, signal_carrier_indices, signal_carrier_placing_indices,
    noise_function_input_indices, noise_function_indices, noise_output_indices, noise_system_matrix_indices,
):''', '''def _signal_and_noise_dense(
    signal_outputs, noise_outputs, carrier, signal_matrix, noise_matrix, noise_selection_vectors,
    qhd_placing_indices, qhd_phase_values, signal_output_indices, signal_matrix_indices,
    signal_carrier_indices, signal_carrier_placing_indices, noise_output_indices, noise_system_matrix_indices,
):''')
rep('''    signal_matrix = vmap(update, in_axes=(0, None, None, None, None, None, None))(
        parameters,
        signal_matrix,
        signal_function_input_indices,
        signal_output_indices,
        signal_matrix_indices,
        signal_function_indices,
        _SIGNAL_DISPATCH,
    )''', '''    signal_matrix = vmap(scatter_outputs, in_axes=(0, None, None, None))(
        signal_outputs, signal_matrix, signal_output_indices, signal_matrix_indices
    )''')
rep('''    noise_matrix = vmap(update, in_axes=(0, None, None, None, None, None, None))(
        parameters,
        noise_matrix,
        noise_function_input_indices,
        noise_output_indices,
        noise_system_matrix_indices,
        noise_function_indices,
        _NOISE_DISPATCH,
    )''', '''    noise_matrix = scatter_outputs(noise_outputs, noise_matrix, noise_output_indices, noise_system_matrix_indices)[None]''')

a = s.index("    # --------------------\n    # Carrier solve")
b = s.index("    return carrier.T, signal.T, noise.squeeze()")
s = s[:a] + '''    # --------------------
    # Component functions of all three systems, evaluated together
    # --------------------
    carrier_parameters = expand_parameters(parameters, carrier_changing_parameter_indices, carrier_changing_values)
    signal_parameters = expand_parameters(carrier_parameters, signal_changing_parameter_indices, signal_changing_values)
    carrier_outputs, signal_outputs, noise_outputs = _component_outputs(
        carrier_parameters, signal_parameters,
        carrier_function_input_indices, carrier_function_indices,
        signal_function_input_indices, signal_function_indices,
        noise_function_input_indices, noise_function_indices,
    )

    # --------------------
    # Carrier solve
    # --------------------
    carrier_matrix = vmap(scatter_outputs, in_axes=(0, None, None, None))(
        carrier_outputs, carrier_matrix, carrier_output_indices, carrier_matrix_indices
    )
    carrier = vmap(solve)(carrier_matrix[:, :, :-1], carrier_matrix[:, :, -1])

    # --------------------
    # Signal solve + quantum noise
    # --------------------
    if _DENSE:
        signal, noise = _signal_and_noise_dense(
            signal_outputs, noise_outputs, carrier, signal_matrix, noise_matrix, noise_selection_vectors,
            qhd_placing_indices, qhd_phase_values, signal_output_indices, signal_matrix_indices,
            signal_carrier_indices, signal_carrier_placing_indices, noise_output_indices,
            noise_system_matrix_indices,
        )
    else:
        # The noise source matrix is assumed to start out empty (as `build` always creates it): all
        # its nonzeros are written by the noise component functions.
        signal, noise = _signal_and_noise_sparse(
            signal_outputs, noise_outputs, carrier, signal_matrix, noise_selection_vectors,
            qhd_placing_indices, qhd_phase_values, signal_output_indices, signal_matrix_indices,
            signal_carrier_indices, signal_carrier_placing_indices, noise_output_indices,
            noise_system_matrix_indices,
        )

''' + s[b:]
open(p, 'w', encoding='utf-8').write(s)
print("ok")
