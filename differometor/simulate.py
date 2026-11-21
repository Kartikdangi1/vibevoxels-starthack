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
import jax.scipy.linalg
import numpy as np
from time import time
import jax.numpy as jnp
from jax import lax, vmap, jit

from differometor.setups import Setup
from differometor.build import build, pairs_to_arrays, prepare_arrays, LINKING_FUNCTIONS
from differometor.components import FUNCTIONS, UNIT_VACUUM, H_PLANCK, F0


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
    Both inputs are cast to complex128 to reduce numerical issues during solving.

    The function intentionally returns the raw complex solution rather than derived
    quantities (such as phase or amplitude), because some downstream transforms
    (notably `jnp.angle`) can produce invalid gradients in some configurations.
    """
    right_hand_side = right_hand_side.astype(jnp.complex128)
    matrix = matrix.astype(jnp.complex128)
    return jnp.linalg.solve(matrix, right_hand_side)


def solve_with_adjoint(
    matrix: jnp.ndarray, right_hand_side: jnp.ndarray, adjoint_right_hand_sides: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Solve ``M x = b`` and ``M^H y_d = s_d`` for one matrix using a single LU factorization.

    Parameters
    ----------
    matrix : jnp.ndarray
        System matrix M, shape (N, N).
    right_hand_side : jnp.ndarray
        Right-hand side b, shape (N,).
    adjoint_right_hand_sides : jnp.ndarray
        Right-hand sides s_d for the conjugate-transposed system, shape (D, N).

    Returns
    -------
    solution : jnp.ndarray
        x = M^{-1} b, shape (N,).
    adjoint_solutions : jnp.ndarray
        y_d = M^{-H} s_d, shape (D, N).

    Notes
    -----
    This mirrors how ``jnp.linalg.solve`` is implemented: the factorization is
    taken of ``stop_gradient(M)`` and ``lax.custom_linear_solve`` routes all
    derivatives through the (linear, cheap) matrix-vector product, so reverse
    mode reuses the same factors instead of differentiating through the LU.
    The only change is that one factorization now serves both systems, where
    previously each was factorized separately.
    """
    matrix = matrix.astype(jnp.complex128)
    right_hand_side = right_hand_side.astype(jnp.complex128)
    adjoint_right_hand_sides = adjoint_right_hand_sides.astype(jnp.complex128)

    lu, pivots, _ = lax.linalg.lu(lax.stop_gradient(matrix))
    factors = (lu, pivots)

    def lu_solve(vector, trans):
        return jax.scipy.linalg.lu_solve(factors, vector, trans=trans)

    # M x = b. Linear transpose of x -> M x is x -> M^T x (trans=1).
    solution = lax.custom_linear_solve(
        lambda x: matrix @ x,
        right_hand_side,
        solve=lambda _, b: lu_solve(b, 0),
        transpose_solve=lambda _, b: lu_solve(b, 1),
    )

    # M^H y = s. Linear transpose of y -> M^H y is y -> conj(M) y,
    # and conj(M) z = c  <=>  z = conj(M^{-1} conj(c)).
    def adjoint_solve(vector):
        return lax.custom_linear_solve(
            lambda y: jnp.conjugate(matrix).T @ y,
            vector,
            solve=lambda _, s: lu_solve(s, 2),
            transpose_solve=lambda _, c: jnp.conjugate(lu_solve(jnp.conjugate(c), 0)),
        )

    adjoint_solutions = vmap(adjoint_solve)(adjoint_right_hand_sides)
    return solution, adjoint_solutions


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
    outputs = evaluate_entries(parameters, function_input_indices, output_indices, function_indices)
    return matrix.at[matrix_indices[0], matrix_indices[1]].set(outputs)


def evaluate_entries(
    parameters: jnp.ndarray,
    function_input_indices: jnp.ndarray | tuple[jnp.ndarray, ...],
    output_indices: tuple[jnp.ndarray, jnp.ndarray],
    function_indices: jnp.ndarray,
) -> jnp.ndarray:
    """
    Evaluate component functions and return the values destined for matrix entries.

    Parameters
    ----------
    parameters : jnp.ndarray
        Parameter values for one simulation instance, shape (P,).
    function_input_indices : jnp.ndarray or tuple of jnp.ndarray
        Either the original layout, shape (K, I), paired with ``function_indices``
        selecting a function per row; or the grouped layout produced by
        ``group_by_function``: a tuple with one (K_f, I) index array per entry of
        ``FUNCTIONS``, rows ordered by function.
    output_indices : tuple[jnp.ndarray, jnp.ndarray]
        (row into the evaluated outputs, output slot), each shape (M,).
    function_indices : jnp.ndarray
        Function selector per row, shape (K,). Only used by the original layout.

    Returns
    -------
    outputs : jnp.ndarray
        Complex values for the M matrix entries, shape (M,).

    Notes
    -----
    The original layout evaluates ``lax.switch`` under ``vmap``, which turns into a
    select over *all* functions in ``FUNCTIONS`` for every row and puts every branch
    (and its derivative) into the compiled program. The grouped layout calls each
    function that the setup actually uses once, on its own batch of rows, which
    gives identical values with a much smaller program.
    """
    if isinstance(function_input_indices, tuple):
        output_matrix = jnp.concatenate(
            [
                vmap(FUNCTIONS[function_index])(parameters[indices])
                for function_index, indices in enumerate(function_input_indices)
                if indices.shape[0] > 0
            ]
        )
    else:
        function_inputs = parameters[function_input_indices]
        output_matrix = vmap(lambda i, x: lax.switch(i, FUNCTIONS, x))(function_indices, function_inputs)
    return output_matrix[output_indices[0], output_indices[1]]


def group_by_function(
    function_input_indices: jnp.ndarray,
    function_indices: jnp.ndarray,
    output_indices: jnp.ndarray,
) -> tuple[tuple[jnp.ndarray, ...], jnp.ndarray, jnp.ndarray]:
    """
    Reorder build-step function arrays so that rows using the same function are contiguous.

    Runs once at build time on concrete arrays. Returns the grouped input indices
    (one array per entry of ``FUNCTIONS``; empty for unused functions), the
    function indices in the new row order, and ``output_indices`` with its row
    references remapped to the new order. ``evaluate_entries`` then produces the
    exact same entry values as the original ``lax.switch`` path.
    """
    inputs = np.asarray(function_input_indices)
    functions = np.asarray(function_indices)
    outputs = np.asarray(output_indices)
    order = np.argsort(functions, kind="stable")
    new_row = np.empty_like(order)
    new_row[order] = np.arange(order.size)
    sorted_inputs, sorted_functions = inputs[order], functions[order]
    grouped = tuple(
        jnp.asarray(sorted_inputs[sorted_functions == function_index]) for function_index in range(len(FUNCTIONS))
    )
    remapped_outputs = jnp.asarray(np.stack((new_row[outputs[0]], outputs[1])))
    return grouped, jnp.asarray(sorted_functions), remapped_outputs


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
    parameters = parameters.at[[0], optimized_parameter_indices].set(
        optimized_parameters[optimized_value_indices]
    )

    # Apply parameter linking (derived parameters).
    inputs_to_link = parameters[0, linked_indices]
    outputs_to_link = vmap(lambda i, x: lax.switch(i, LINKING_FUNCTIONS, x))(
        linking_function_indices, inputs_to_link
    )
    parameters = parameters.at[[0], indices_to_link].set(outputs_to_link)

    # Quadrature-homodyne-detector phase factors (unit magnitude complex phase).
    qhd_phase_values = jnp.exp(1j * jnp.radians(parameters[0, qhd_parameter_indices]))

    # --------------------
    # Carrier solve
    # --------------------
    parameters = expand_parameters(parameters, carrier_changing_parameter_indices, carrier_changing_values)
    carrier_matrix = vmap(update, in_axes=(0, None, None, None, None, None))(
        parameters,
        carrier_matrix,
        carrier_function_input_indices,
        carrier_output_indices,
        carrier_matrix_indices,
        carrier_function_indices,
    )
    carrier = vmap(solve)(carrier_matrix[:, :, :-1], carrier_matrix[:, :, -1])

    # --------------------
    # Signal solve
    # --------------------
    parameters = expand_parameters(parameters, signal_changing_parameter_indices, signal_changing_values)
    signal_matrix = vmap(update, in_axes=(0, None, None, None, None, None))(
        parameters,
        signal_matrix,
        signal_function_input_indices,
        signal_output_indices,
        signal_matrix_indices,
        signal_function_indices,
    )

    # Scale signal connector entries by the carrier solution at the respective spaces.
    # The minus sign matches the sign convention used by the referenced modeling tools.
    signal_entries = (
        -signal_matrix[:, signal_carrier_placing_indices[0], signal_carrier_placing_indices[1]]
        * carrier[:, signal_carrier_indices.flatten()]
    )
    signal_matrix = signal_matrix.at[:, signal_carrier_placing_indices[0], signal_carrier_placing_indices[1]].set(
        signal_entries
    )

    # Apply conjugation rules to enforce upper and lower sideband symmetry.
    carrier_size = carrier.shape[1]
    signal_matrix = signal_matrix.at[:, carrier_size : carrier_size * 2, carrier_size : carrier_size * 2].set(
        jnp.conjugate(signal_matrix[:, carrier_size : carrier_size * 2, carrier_size : carrier_size * 2])
    )
    signal_matrix = signal_matrix.at[:, carrier_size * 2 :, :carrier_size].set(
        jnp.conjugate(signal_matrix[:, carrier_size * 2 :, :carrier_size])
    )
    signal_matrix = signal_matrix.at[:, carrier_size : carrier_size * 2, carrier_size * 2 :].set(
        jnp.conjugate(signal_matrix[:, carrier_size : carrier_size * 2, carrier_size * 2 :])
    )

    # --------------------
    # Quantum noise selection vectors (depend only on the carrier)
    # --------------------
    # Apply quadrature-homodyne-detector phases into the selection vectors.
    # `lax.scatter_mul` is used (rather than `.at[...].set`) so empty index arrays
    # remain valid without conditionals.
    noise_selection_vectors = lax.scatter_mul(
        noise_selection_vectors,
        qhd_placing_indices.reshape(-1, 3),
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

    # --------------------
    # Signal solve + detector back-propagation, sharing one LU per frequency
    # --------------------
    # signal        = M^{-1} b
    # noise_weights = M^{-H} s   (detector weights back-propagated through the signal system)
    carrier_selections = jnp.broadcast_to(
        carrier_selections,
        (carrier_selections.shape[0], signal_matrix.shape[0], carrier_selections.shape[2]),
    )
    signal, noise_weights = vmap(solve_with_adjoint, in_axes=(0, 0, 1), out_axes=(0, 1))(
        signal_matrix[:, :, :-1], signal_matrix[:, :, -1], carrier_selections
    )

    # --------------------
    # Quantum noise
    # --------------------

    if noise_matrix is None:
        # Sparse input noise (set by `run_build_step` when the base noise matrix is
        # all zeros and no entry is written twice): V_i has only the entries
        # (r_e, c_e, value_e), so w^H V_i w = sum_e conj(w[r_e]) * value_e * w[c_e].
        # This avoids materializing a dense (values, N, N) noise matrix.
        noise_values = vmap(evaluate_entries, in_axes=(0, None, None, None))(
            parameters, noise_function_input_indices, noise_output_indices, noise_function_indices
        )
        rows, columns = noise_system_matrix_indices
        temporary_sum = jnp.sum(
            jnp.real(jnp.conjugate(noise_weights[:, :, rows]) * noise_values * noise_weights[:, :, columns]), axis=2
        )
        noise = jnp.sqrt(2 * temporary_sum * UNIT_VACUUM * H_PLANCK * F0 * 0.25)
        return carrier.T, signal.T, noise.squeeze()

    # Update noise source matrix (may depend on current parameters).
    noise_matrix = vmap(update, in_axes=(0, None, None, None, None, None))(
        parameters,
        noise_matrix,
        noise_function_input_indices,
        noise_output_indices,
        noise_system_matrix_indices,
        noise_function_indices,
    )

    def matmul_single_batch(matrix_batch: jnp.ndarray, vector_batch: jnp.ndarray) -> jnp.ndarray:
        """
        Multiply a batch of matrices with a batch of vectors.

        Parameters
        ----------
        matrix_batch : jnp.ndarray
            Shape (B, N, N).
        vector_batch : jnp.ndarray
            Shape (B, N).

        Returns
        -------
        product : jnp.ndarray
            Shape (B, N).
        """
        return jnp.einsum("bij,bj->bi", matrix_batch, vector_batch)

    # Project noise sources into the system.
    noise_sources = vmap(matmul_single_batch, in_axes=(None, 0))(noise_matrix, noise_weights)

    # Output noise power s^H M^{-1} V_i M^{-H} s. Previously this solved
    # c = M^{-1} (V_i w) with a third factorization and formed Re(s^H c).
    # Since s^H M^{-1} = (M^{-H} s)^H = w^H, the same quantity is Re(w^H V_i w),
    # which needs no further solve.
    temporary_sum = jnp.sum(jnp.real(noise_sources * jnp.conjugate(noise_weights)), axis=2)
    noise = jnp.sqrt(2 * temporary_sum * UNIT_VACUUM * H_PLANCK * F0 * 0.25)

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

    arg_count = simulate.__code__.co_argcount
    arg_names = simulate.__code__.co_varnames[:arg_count]

    all_arrays = (*prepared_arrays, *matrices, *carrier_arrays, *signal_arrays, *noise_arrays)
    if len(arg_names) != len(all_arrays):
        raise ValueError(
            "simulate signature does not match the number of prepared arrays and matrices. "
            f"simulate expects {len(arg_names)} arguments, but build produced {len(all_arrays)} arrays."
        )

    simulation_arrays = {arg_names[i]: array for i, array in enumerate(all_arrays)}
    _optimize_layout(simulation_arrays)

    if timeit:
        for array in simulation_arrays.values():
            jax.block_until_ready(array)
        return time() - start, (simulation_arrays, metadata)

    return simulation_arrays, *metadata


def _optimize_layout(simulation_arrays: dict) -> None:
    """
    Rewrite build-step arrays (in place) into the layouts that `simulate` evaluates fastest.

    - Function arrays of the carrier, signal and noise systems are grouped by
      function (see `group_by_function`), removing `lax.switch` from the program.
    - If the base noise matrix is all zeros and no noise entry is written twice,
      `noise_matrix` is replaced by None so `simulate` uses the sparse noise path.
      Both conditions are checked here on concrete values; otherwise the dense
      path is kept.

    All rewrites are exact reorderings/representations of the same data.
    """
    for prefix in ("carrier", "signal", "noise"):
        grouped, functions, outputs = group_by_function(
            simulation_arrays[f"{prefix}_function_input_indices"],
            simulation_arrays[f"{prefix}_function_indices"],
            simulation_arrays[f"{prefix}_output_indices"],
        )
        simulation_arrays[f"{prefix}_function_input_indices"] = grouped
        simulation_arrays[f"{prefix}_function_indices"] = functions
        simulation_arrays[f"{prefix}_output_indices"] = outputs

    noise_positions = np.asarray(simulation_arrays["noise_system_matrix_indices"])
    unique_positions = np.unique(noise_positions, axis=1).shape[1]
    if unique_positions == noise_positions.shape[1] and not np.asarray(simulation_arrays["noise_matrix"]).any():
        simulation_arrays["noise_matrix"] = None


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
