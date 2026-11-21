"""
JAX-based optical simulation runner.

This module builds linear systems from a `differometor` `Setup` and solves:
- a carrier (steady-state) system,
- a signal (sideband) system,
- and a quantum noise propagation system.

The main public entry points are:
- `run`: build once and simulate once
- `run_setups`: run multiple setups over a shared frequency grid
- `run_with_parameter_sets`: run the same setup over many optimized-parameter sets

Conventions used throughout:
- Arrays are JAX arrays (`jax.numpy`).
- Shapes are documented explicitly for every function.
- All indices are integer arrays that index into parameter vectors and matrix entries.
"""

import jax
from time import time
import jax.numpy as jnp
from jax import lax, vmap, jit
from jax.scipy.linalg import lu_factor, lu_solve

from differometor.setups import Setup
from differometor.build import build, pairs_to_arrays, prepare_arrays, LINKING_FUNCTIONS
from differometor.components import FUNCTIONS, UNIT_VACUUM, H_PLANCK, F0
from differometor.plans import MatrixUpdatePlan, SimulationPlan, make_simulation_plan


jax.config.update("jax_enable_x64", True)


def solve(matrix: jnp.ndarray, right_hand_side: jnp.ndarray) -> jnp.ndarray:
    """
    Solve a dense linear system for each batch item.

    Parameters
    ----------
    matrix : jnp.ndarray
        System matrix with shape (..., system_size, system_size).
    right_hand_side : jnp.ndarray
        Right-hand side vector with shape (..., system_size).

    Returns
    -------
    solution : jnp.ndarray
        Solution vector with shape (..., system_size).

    Notes
    -----
    Matrix assembly guarantees complex128 inputs. The function intentionally
    returns the raw complex solution rather than derived
    quantities (such as phase or amplitude), because some downstream transforms
    (notably `jnp.angle`) can produce invalid gradients in some configurations.
    """
    return jnp.linalg.solve(matrix, right_hand_side)


def solve_factored(
    matrix: jnp.ndarray,
    factors: tuple[jnp.ndarray, jnp.ndarray],
    right_hand_side: jnp.ndarray,
    trans: int = 0,
) -> jnp.ndarray:
    """Reuse LU factors with the implicit derivative of a linear solve."""
    if trans == 0:
        matvec = lambda value: matrix @ value
        solve = lambda _matvec, value: lu_solve(factors, value, trans=0)
        transpose_solve = lambda _matvec, value: lu_solve(factors, value, trans=1)
    elif trans == 2:
        matvec = lambda value: jnp.conjugate(matrix.T) @ value
        solve = lambda _matvec, value: lu_solve(factors, value, trans=2)
        transpose_solve = lambda _matvec, value: jnp.conjugate(
            lu_solve(factors, jnp.conjugate(value), trans=0)
        )
    else:
        raise ValueError("Factored solve supports trans=0 or trans=2.")
    return lax.custom_linear_solve(
        matvec,
        right_hand_side,
        solve=solve,
        transpose_solve=transpose_solve,
    )


def update(
    parameters: jnp.ndarray,
    matrix: jnp.ndarray,
    function_input_indices: jnp.ndarray,
    output_indices: tuple[jnp.ndarray, jnp.ndarray],
    matrix_indices: tuple[jnp.ndarray, jnp.ndarray],
    function_indices: jnp.ndarray,
) -> jnp.ndarray:
    """
    Update selected entries of a system matrix by evaluating component functions.

    This function takes a parameter vector, selects the subset that serves as
    inputs to component transfer functions, evaluates those functions, selects
    certain outputs, and writes them into `matrix` at specified indices.

    Parameters
    ----------
    parameters : jnp.ndarray
        Parameter values for one simulation instance, shape (P,).
    matrix : jnp.ndarray
        Base matrix to update, shape (N, N + 1) for augmented systems, or (N, N).
        The function writes entries in-place (functionally) using `.at[...].set(...)`.
    function_input_indices : jnp.ndarray
        Indices into `parameters` selecting inputs to the component functions,
        shape (K,).
    output_indices : tuple[jnp.ndarray, jnp.ndarray]
        Pair of index arrays selecting entries from the evaluated output matrix.
        Each element has shape (M,). Together they define M output positions.
    matrix_indices : tuple[jnp.ndarray, jnp.ndarray]
        Pair of index arrays selecting where to write into `matrix`.
        Each element has shape (M,).
    function_indices : jnp.ndarray
        Indices selecting which function to use for each input, shape (K,).
        Values index into the global `FUNCTIONS` list.

    Returns
    -------
    updated_matrix : jnp.ndarray
        Matrix with updated entries, same shape as `matrix`.

    Notes
    -----
    `lax.switch` selects functions from the global `FUNCTIONS` list. In JAX, a
    Python list of callables referenced as a global is treated as static during
    tracing, which keeps the function jittable as long as the list is stable.
    """
    function_inputs = parameters[function_input_indices]
    output_matrix = vmap(lambda i, x: lax.switch(i, FUNCTIONS, x))(function_indices, function_inputs)
    outputs = output_matrix[output_indices[0], output_indices[1]]
    return matrix.at[matrix_indices[0], matrix_indices[1]].set(outputs)


def grouped_update(
    parameters: jnp.ndarray,
    matrix: jnp.ndarray,
    plan: MatrixUpdatePlan | tuple,
) -> jnp.ndarray:
    """Evaluate only the component functions present in this matrix update."""
    groups = plan.groups if isinstance(plan, MatrixUpdatePlan) else plan
    values = []
    rows = []
    columns = []
    for group in groups:
        input_indices = jnp.asarray(group.input_indices, dtype=jnp.int32)
        outputs = vmap(FUNCTIONS[group.function_id])(parameters[input_indices])
        output_rows = jnp.asarray(group.output_rows, dtype=jnp.int32)
        output_columns = jnp.asarray(group.output_columns, dtype=jnp.int32)
        values.append(outputs[output_rows, output_columns])
        rows.extend(group.matrix_rows)
        columns.extend(group.matrix_columns)

    if not values:
        return matrix
    updates = jnp.concatenate(values)
    return matrix.at[
        jnp.asarray(rows, dtype=jnp.int32),
        jnp.asarray(columns, dtype=jnp.int32),
    ].set(updates)


def expand_parameters(array: jnp.ndarray, indices: jnp.ndarray, values: jnp.ndarray) -> jnp.ndarray:
    """
    Expand a single parameter row into multiple rows by replacing selected entries.

    Parameters
    ----------
    array : jnp.ndarray
        Base parameter array with shape (1, P).
    indices : jnp.ndarray
        Column indices to replace, shape (V,).
    values : jnp.ndarray
        Replacement values stacked row-wise for each index, shape (V, R),
        where R is the number of expanded rows.

        Example:
        - array shape (1, 5)
        - indices shape (2,)
        - values shape (2, 3)
        Output has shape (3, 5), where the two indexed columns are filled from
        `values[:, row]`.

    Returns
    -------
    expanded : jnp.ndarray
        Expanded parameter array with shape (R, P).

    Notes
    -----
    This implementation uses indexed updates that remain compatible with JAX
    transformations and compilation.
    """
    if values.shape[1] == 1:
        return array.at[0, indices].set(values[:, 0])

    tiled_array = jnp.tile(array, (values.shape[1], 1))
    row_indices = jnp.repeat(jnp.arange(values.shape[1]), len(indices))
    column_indices = jnp.tile(indices, values.shape[1])
    flat_values = values.T.flatten()
    return tiled_array.at[row_indices, column_indices].set(flat_values)


def simulate(
    # prepared arrays
    optimized_parameters: jnp.ndarray,
    optimized_parameter_indices: jnp.ndarray,
    optimized_value_indices: jnp.ndarray,
    carrier_changing_parameter_indices: jnp.ndarray,
    carrier_changing_values: jnp.ndarray,
    signal_changing_parameter_indices: jnp.ndarray,
    signal_changing_values: jnp.ndarray,
    # matrices
    parameters: jnp.ndarray,
    carrier_matrix: jnp.ndarray,
    signal_matrix: jnp.ndarray,
    noise_matrix: jnp.ndarray,
    noise_selection_vectors: jnp.ndarray,
    qhd_parameter_indices: jnp.ndarray,
    qhd_placing_indices: jnp.ndarray,
    linked_indices: jnp.ndarray,
    linking_function_indices: jnp.ndarray,
    indices_to_link: jnp.ndarray,
    # carrier arrays
    carrier_function_input_indices: jnp.ndarray,
    carrier_function_indices: jnp.ndarray,
    carrier_output_indices: tuple[jnp.ndarray, jnp.ndarray],
    carrier_matrix_indices: tuple[jnp.ndarray, jnp.ndarray],
    # signal arrays
    signal_function_input_indices: jnp.ndarray,
    signal_function_indices: jnp.ndarray,
    signal_output_indices: tuple[jnp.ndarray, jnp.ndarray],
    signal_matrix_indices: tuple[jnp.ndarray, jnp.ndarray],
    signal_carrier_indices: jnp.ndarray,
    signal_carrier_placing_indices: tuple[jnp.ndarray, jnp.ndarray],
    # noise arrays
    noise_function_input_indices: jnp.ndarray,
    noise_function_indices: jnp.ndarray,
    noise_output_indices: tuple[jnp.ndarray, jnp.ndarray],
    noise_system_matrix_indices: tuple[jnp.ndarray, jnp.ndarray],
    update_plan: SimulationPlan,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Run one full simulation pass: carrier, signal, and quantum noise.

    Parameters
    ----------
    optimized_parameters : jnp.ndarray
        Optimized parameter values provided by an optimizer or sampling routine,
        shape (OPV,) or (num_ranges, OPV) in outer vectorization contexts.
    optimized_parameter_indices : jnp.ndarray
        Indices into `parameters` for parameters that are driven by optimized values,
        shape (OP,).
    optimized_value_indices : jnp.ndarray
        Indices into `optimized_parameters` mapping each optimized parameter to
        its source value, shape (OP,). This supports shared optimized values.
    carrier_changing_parameter_indices : jnp.ndarray
        Indices into `parameters` that change for the carrier sweep, shape (CCP,).
    carrier_changing_values : jnp.ndarray
        Values for those carrier-changing parameters, shape (CCP, CV).
    signal_changing_parameter_indices : jnp.ndarray
        Indices into `parameters` that change for the signal sweep, shape (SCP,).
    signal_changing_values : jnp.ndarray
        Values for those signal-changing parameters, shape (SCP, SV).

    parameters : jnp.ndarray
        Base parameter array, shape (1, P). It is expanded internally.
    carrier_matrix : jnp.ndarray
        Base carrier augmented matrix, shape (CN, CN + 1).
    signal_matrix : jnp.ndarray
        Base signal augmented matrix, shape (SN, SN + 1).
    noise_matrix : jnp.ndarray
        Noise source covariance matrix, shape (NN, NN) or broadcastable for vmap.
    noise_selection_vectors : jnp.ndarray
        Detector selection vectors, shape (D, 1, setup_size).
    qhd_parameter_indices : jnp.ndarray
        Indices of quadrature-homodyne-detector phase parameters in `parameters`,
        shape (Q,).
    qhd_placing_indices : jnp.ndarray
        Indices describing where detector phases are applied into
        `noise_selection_vectors`, shape (Q, 3). Each row contains:
        (detector_index, row_index, column_index).
    linked_indices : jnp.ndarray
        Parameter indices used as inputs to linking functions, shape (L,).
    linking_function_indices : jnp.ndarray
        Function selectors for linking functions, shape (L,).
    indices_to_link : jnp.ndarray
        Parameter indices to receive outputs of linking functions, shape (L,).

    carrier_function_input_indices, carrier_function_indices, carrier_output_indices,
    carrier_matrix_indices :
        Indexing and function-selection arrays used by `update` to fill the carrier matrix.

    signal_function_input_indices, signal_function_indices, signal_output_indices,
    signal_matrix_indices :
        Indexing and function-selection arrays used by `update` to fill the signal matrix.

    signal_carrier_indices : jnp.ndarray
        Indices selecting the carrier solution entries used to scale signal connectors.
    signal_carrier_placing_indices : tuple[jnp.ndarray, jnp.ndarray]
        Matrix indices locating the signal connector entries that must be scaled by carrier.

    noise_function_input_indices, noise_function_indices, noise_output_indices,
    noise_system_matrix_indices :
        Indexing and function-selection arrays used by `update` to fill the noise matrix.

    Returns
    -------
    carrier : jnp.ndarray
        Carrier solution, transposed to shape (CN, CV).
    signal : jnp.ndarray
        Signal solution, transposed to shape (SN, SV).
    noise : jnp.ndarray
        Quantum noise amplitude spectral density, shape (D, SV) after squeeze.

    Notes
    -----
    - Linking functions are applied before parameter expansion so they operate on a single row.
    - The signal path is always executed (even if effectively empty) to avoid conditionals
      that would force compilation of branches with empty shapes.
    - Conjugation and block manipulation follow the conventions used for upper and lower
      sidebands in frequency-domain interferometer models.
    """
    # Insert optimized parameter values into the base parameter row.
    parameters = parameters.at[
        0, jnp.asarray(update_plan.optimized_parameter_indices, dtype=jnp.int32)
    ].set(
        optimized_parameters[jnp.asarray(update_plan.optimized_value_indices, dtype=jnp.int32)]
    )

    # Apply parameter linking (derived parameters).
    linked_outputs = []
    linked_targets = []
    for function, input_indices, target_indices in update_plan.linked_groups:
        inputs = parameters[0, jnp.asarray(input_indices, dtype=jnp.int32)]
        linked_outputs.append(vmap(function)(inputs))
        linked_targets.extend(target_indices)
    if linked_outputs:
        parameters = parameters.at[
            0, jnp.asarray(linked_targets, dtype=jnp.int32)
        ].set(jnp.concatenate(linked_outputs))

    # Quadrature-homodyne-detector phase factors (unit magnitude complex phase).
    qhd_phase_values = jnp.exp(
        1j * jnp.radians(parameters[0, jnp.asarray(update_plan.qhd_parameter_indices, dtype=jnp.int32)])
    )

    # --------------------
    # Carrier solve
    # --------------------
    parameters = expand_parameters(
        parameters,
        jnp.asarray(update_plan.carrier_changing_parameter_indices, dtype=jnp.int32),
        carrier_changing_values,
    )
    if parameters.shape[0] == 1:
        carrier_matrix = grouped_update(
            parameters[0], carrier_matrix, update_plan.carrier.static_groups
        )[None, :, :]
    else:
        carrier_matrix = vmap(
            lambda parameter: grouped_update(
                parameter, carrier_matrix, update_plan.carrier.static_groups
            )
        )(parameters)
    carrier = vmap(solve)(carrier_matrix[:, :, :-1], carrier_matrix[:, :, -1])

    # --------------------
    # Signal solve
    # --------------------
    parameters = expand_parameters(
        parameters,
        jnp.asarray(update_plan.signal_changing_parameter_indices, dtype=jnp.int32),
        signal_changing_values,
    )
    signal_base_matrix = grouped_update(
        parameters[0], signal_matrix, update_plan.signal.static_groups
    )
    carrier_size = carrier.shape[1]

    # Homodyne phases and carrier selections do not vary over the frequency
    # sweep, so build them once and feed one RHS block to each mapped solve.
    #
    # `lax.map` keeps only one dense signal matrix live at a time. Checkpointing
    # the per-frequency body rematerializes its intermediates in reverse mode,
    # which avoids the multi-gigabyte batched autodiff tape.
    carrier_rows = jnp.asarray(update_plan.signal_carrier_rows, dtype=jnp.int32)
    carrier_columns = jnp.asarray(update_plan.signal_carrier_columns, dtype=jnp.int32)
    carrier_indices = jnp.asarray(update_plan.signal_carrier_indices, dtype=jnp.int32)
    # Apply quadrature-homodyne-detector phases. Static placement indices keep
    # the scatter's index metadata out of the traced dataflow.
    noise_selection_vectors = lax.scatter_mul(
        noise_selection_vectors,
        jnp.asarray(update_plan.qhd_placing_indices, dtype=jnp.int32).reshape(-1, 3),
        qhd_phase_values,
        lax.ScatterDimensionNumbers(
            update_window_dims=(),
            inserted_window_dims=(0, 1, 2),
            scatter_dims_to_operand_dims=(0, 1, 2),
        ),
        unique_indices=True,
    )

    # Broadcast selection vectors across the carrier sweep dimension.
    noise_selection_vectors = jnp.tile(noise_selection_vectors, (1, carrier.shape[0], 1))

    # Select necessary carrier entries and build upper and lower sideband selections.
    upper_carrier_selections = noise_selection_vectors[:, :, : carrier.shape[1]] * carrier
    upper_carrier_selections = noise_selection_vectors.at[:, :, : carrier.shape[1]].set(upper_carrier_selections)

    lower_carrier_selections = jnp.conjugate(
        upper_carrier_selections[:, :, carrier.shape[1] : carrier.shape[1] * 2] * carrier
    )
    carrier_selections = upper_carrier_selections.at[
        :, :, carrier.shape[1] : carrier.shape[1] * 2
    ].set(lower_carrier_selections)

    # Broadcast only detector vectors, not dense matrices, over frequencies.
    carrier_selections = jnp.broadcast_to(
        carrier_selections,
        (carrier_selections.shape[0], parameters.shape[0], carrier_selections.shape[2]),
    )
    detector_rhs = jnp.transpose(carrier_selections, (1, 2, 0))

    # The noise-source matrix is frequency independent for the benchmark; keep
    # that matrix single-copy and support frequency-dependent custom setups too.
    if update_plan.noise.frequency_groups:
        noise_base_matrix = grouped_update(
            parameters[0], noise_matrix, update_plan.noise.static_groups
        )
    else:
        noise_base_matrix = grouped_update(
            parameters[0], noise_matrix, update_plan.noise.static_groups
        )

    def simulate_one_frequency(inputs):
        parameter, detector_rhs_single = inputs
        matrix = signal_base_matrix
        if update_plan.signal.frequency_groups:
            matrix = grouped_update(
                parameter, matrix, update_plan.signal.frequency_groups
            )

        # Scale signal connector entries by the carrier solution. The minus
        # sign matches the simulator's optical sign convention.
        signal_entries = (
            -matrix[carrier_rows, carrier_columns]
            * carrier[0, carrier_indices]
        )
        matrix = matrix.at[carrier_rows, carrier_columns].set(signal_entries)

        matrix = matrix.at[
            carrier_size : carrier_size * 2,
            carrier_size : carrier_size * 2,
        ].set(jnp.conjugate(matrix[
            carrier_size : carrier_size * 2,
            carrier_size : carrier_size * 2,
        ]))
        matrix = matrix.at[carrier_size * 2 :, :carrier_size].set(
            jnp.conjugate(matrix[carrier_size * 2 :, :carrier_size])
        )
        matrix = matrix.at[
            carrier_size : carrier_size * 2,
            carrier_size * 2 :,
        ].set(jnp.conjugate(matrix[
            carrier_size : carrier_size * 2,
            carrier_size * 2 :,
        ]))

        system = matrix[:, :-1]
        factors = jax.tree_util.tree_map(lax.stop_gradient, lu_factor(system))
        signal_solution = solve_factored(system, factors, matrix[:, -1])
        noise_weights = solve_factored(
            system, factors, detector_rhs_single, trans=2
        )
        if update_plan.noise.frequency_groups:
            per_frequency_noise = grouped_update(
                parameter, noise_base_matrix, update_plan.noise.frequency_groups
            )
        else:
            per_frequency_noise = noise_base_matrix
        noise_sources = per_frequency_noise @ noise_weights
        covariances = solve_factored(system, factors, noise_sources)
        projected = jnp.sum(
            jnp.real(covariances * jnp.conjugate(detector_rhs_single)), axis=0
        )
        noise_value = jnp.sqrt(
            2 * projected * UNIT_VACUUM * H_PLANCK * F0 * 0.25
        )
        return signal_solution, noise_value

    signal, noise_by_frequency = lax.map(
        jax.checkpoint(simulate_one_frequency),
        (parameters, detector_rhs),
    )
    noise = noise_by_frequency.T

    return carrier.T, signal.T, noise.squeeze()


def run_build_step(
    setup: Setup,
    changing_pairs: list | None = None,
    changing_values: jnp.ndarray | None = None,
    optimization_pairs: list | None = None,
    timeit: bool = False,
) -> tuple:
    """
    Build a `Setup` into simulation-ready arrays and matrices.

    Parameters
    ----------
    setup : Setup
        The optical setup to build.
    changing_pairs : list | None
        Each element is a (component_name, parameter_name) tuple describing which
        parameters are swept during simulation. If None, no sweep is applied.
    changing_values : jnp.ndarray | None
        Values for the parameters described by `changing_pairs`.
        Expected shape is (num_changing_parameters, num_values).
    optimization_pairs : list | None
        Each element is a (component_name, parameter_name) tuple describing which
        parameters are treated as optimized inputs (external values injected into
        the parameter vector before solving). If None, no optimized parameters exist.
    timeit : bool
        If True, return an additional timing measurement (seconds) for the build.

    Returns
    -------
    If `timeit` is False
        simulation_arrays : dict[str, jnp.ndarray]
            Mapping from `simulate` argument name to the array value.
        metadata : tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
            (detector_indices, mirror_indices, beamsplitter_indices, isolator_indices)

    If `timeit` is True
        build_seconds : float
            Time spent building and preparing arrays.
        (simulation_arrays, metadata) : tuple
            The same outputs as the non-timed case, packed into a tuple.

    Notes
    -----
    The returned dictionary keys are derived from the `simulate` function signature.
    """
    start = time()

    instructions, matrices, metadata = build(setup)
    arrays_to_prepare, carrier_arrays, signal_arrays, noise_arrays = pairs_to_arrays(
        *instructions, optimization_pairs, changing_pairs
    )
    prepared_arrays = prepare_arrays(*arrays_to_prepare, changing_values, parameters=matrices[0])
    update_plan = make_simulation_plan(
        carrier_arrays,
        signal_arrays,
        noise_arrays,
        matrices,
        tuple(LINKING_FUNCTIONS),
        prepared_arrays,
        arrays_to_prepare[3],
    )

    arg_count = simulate.__code__.co_argcount
    arg_names = simulate.__code__.co_varnames[:arg_count]

    all_arrays = (
        *prepared_arrays, *matrices, *carrier_arrays, *signal_arrays, *noise_arrays,
        update_plan,
    )
    if len(arg_names) != len(all_arrays):
        raise ValueError(
            "simulate signature does not match the number of prepared arrays and matrices. "
            f"simulate expects {len(arg_names)} arguments, but build produced {len(all_arrays)} arrays."
        )

    simulation_arrays = {arg_names[i]: array for i, array in enumerate(all_arrays)}

    if timeit:
        for name, array in simulation_arrays.items():
            if name != "update_plan":
                jax.block_until_ready(array)
        return time() - start, (simulation_arrays, metadata)

    return simulation_arrays, *metadata


def run_simulation_step(
    simulation_arrays: dict,
    jit_simulation: bool = False,
    timeit: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | tuple[float, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """
    Run `simulate` using a pre-built set of arrays.

    Parameters
    ----------
    simulation_arrays : dict[str, jnp.ndarray]
        Dictionary mapping argument names to arrays. This is the first return value
        of `run_build_step`.
    jit_simulation : bool
        If True, compile `simulate` using `jax.jit` and run one warmup call to
        trigger compilation before measuring runtime.
    timeit : bool
        If True, return the simulation runtime (seconds) alongside the results.

    Returns
    -------
    If `timeit` is False
        carrier : jnp.ndarray
        signal : jnp.ndarray
        noise : jnp.ndarray

    If `timeit` is True
        simulation_seconds : float
            Time spent executing the simulation (excluding warmup compilation).
        results : tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
            (carrier, signal, noise)

    Notes
    -----
    The warmup run uses the same inputs and forces compilation when `jit_simulation`
    is enabled.
    """
    simulation_function = simulate

    if jit_simulation:
        simulation_function = jit(simulate)
        _ = simulation_function(**simulation_arrays)  # warmup compilation

    start = time()
    results = simulation_function(**simulation_arrays)

    if timeit:
        for result in results:
            jax.block_until_ready(result)
        return time() - start, results

    return results


def run(
    setup: Setup,
    changing_pairs: list | None = None,
    changing_values: jnp.ndarray | None = None,
    optimization_pairs: list | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Convenience wrapper: build a setup and immediately run the simulation.

    Parameters
    ----------
    setup : Setup
        The setup to simulate.
    changing_pairs : list | None
        Sweep definition as (component_name, parameter_name) pairs.
    changing_values : jnp.ndarray | None
        Sweep values, shape (num_changing_parameters, num_values).
    optimization_pairs : list | None
        Optimized parameter definition as (component_name, parameter_name) pairs.

    Returns
    -------
    carrier : jnp.ndarray
        Carrier solution, shape (num_ports_carrier, num_carrier_values).
    signal : jnp.ndarray
        Signal solution, shape (num_ports_signal, num_signal_values).
    noise : jnp.ndarray
        Noise amplitude spectral density, shape (num_detectors, num_signal_values) after squeeze.
    detector_indices : jnp.ndarray
    mirror_indices : jnp.ndarray
    beamsplitter_indices : jnp.ndarray
    isolator_indices : jnp.ndarray
        Port indices in the order defined by the setup.
    """
    simulation_arrays, *metadata = run_build_step(setup, changing_pairs, changing_values, optimization_pairs)
    carrier, signal, noise = run_simulation_step(simulation_arrays)
    return carrier, signal, noise, *metadata


def run_setups(setups: list[Setup], frequencies: jnp.ndarray):
    """
    Run a list of setups over a shared frequency sweep.

    Parameters
    ----------
    setups : list[Setup]
        Setups to simulate.
    frequencies : jnp.ndarray
        Frequency values. This is passed as the sweep values for the pair ("f", "frequency").
        Expected shape is (1, num_frequencies) or broadcastable to the builder expectations.

    Returns
    -------
    results : list
        Each element is the return value of `run(setup, [("f", "frequency")], frequencies)`.
    """
    results = []
    for setup in setups:
        results.append(run(setup, [("f", "frequency")], frequencies))
    return results
