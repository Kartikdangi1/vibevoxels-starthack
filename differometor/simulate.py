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

import os
import numpy as np
import jax
from time import time
import jax.numpy as jnp
from jax import lax, vmap, jit

from differometor.setups import Setup
from differometor.build import build, pairs_to_arrays, prepare_arrays, LINKING_FUNCTIONS
from differometor.components import (
    FUNCTIONS, UNIT_VACUUM, H_PLANCK, F0,
    laser, surface, space, space_lower, dummy_function, vacuum_quantum_noise, loss_quantum_noise, squeezer,
)
from differometor.sparse_solve import sparse_solve, sparse_system_solve, sparse_system_transpose_solve


jax.config.update("jax_enable_x64", True)


def _dense_solve(matrix: jnp.ndarray, right_hand_side: jnp.ndarray) -> jnp.ndarray:
    """
    Solve a dense linear system for each batch item. This is the original implementation,
    kept as a fallback/reference - see `solve` below for what actually runs by default.

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


# The signal/noise system matrices built by `differometor.build` are extremely sparse (measured
# on the official uifo_800_3000 fixture: 751x751, ~0.4% nonzero), yet a dense LU does full O(N^3)
# work on them. By default the carrier is solved by `sparse_solve` and the signal / quantum-noise
# systems by the entry-based sparse solvers in `differometor.sparse_solve` (one shared sparse LU per
# frequency, values-only assembly and VJPs, see docs/notes/eidothea_literature_verdicts.md).
# Set DIFFEROMETOR_SOLVER=dense to run the original dense assembly + `jnp.linalg.solve` path instead
# (e.g. to re-confirm a discrepancy is/isn't solver-related); it is memory hungry at 101 frequencies.
_DENSE = os.environ.get("DIFFEROMETOR_SOLVER") == "dense"
solve = _dense_solve if _DENSE else sparse_solve


# Verdict 5: `vmap(lax.switch(i, FUNCTIONS, x))` over a batched index lowers to "evaluate every branch
# on every row, then select", and JAX differentiates all of those branches again. That is what makes
# XLA compile times explode (tens of seconds on one CPU). `differometor.build` only ever assigns a
# small, fixed set of component functions to each of the three systems, so each call site dispatches
# over just its own branches (indices are remapped to positions in that branch list).
_CARRIER_FUNCTIONS = (laser, surface, space, dummy_function)
_NOISE_FUNCTIONS = (vacuum_quantum_noise, loss_quantum_noise, squeezer, dummy_function)
_SIGNAL_FUNCTIONS = tuple(f for f in FUNCTIONS if f not in (laser, vacuum_quantum_noise, loss_quantum_noise, squeezer))
# DIFFEROMETOR_DISPATCH=full restores the 19-way switch at every call site (original behaviour).
_FULL_DISPATCH = os.environ.get("DIFFEROMETOR_DISPATCH") == "full"


def _dispatcher(branches: tuple) -> tuple:
    """(branch list, global-function-index -> branch position table) for one call site."""
    if _FULL_DISPATCH:
        return tuple(FUNCTIONS), np.arange(len(FUNCTIONS))
    table = np.zeros(len(FUNCTIONS), dtype=np.int32)
    for position, function in enumerate(branches):
        table[FUNCTIONS.index(function)] = position
    return tuple(branches), table


_CARRIER_DISPATCH = _dispatcher(_CARRIER_FUNCTIONS)
_NOISE_DISPATCH = _dispatcher(_NOISE_FUNCTIONS)
_SIGNAL_DISPATCH = _dispatcher(_SIGNAL_FUNCTIONS)


def _evaluate(dispatch: tuple, function_indices: jnp.ndarray, function_inputs: jnp.ndarray) -> jnp.ndarray:
    """Evaluate component function `function_indices[k]` on `function_inputs[k]`; rows (K, 4)."""
    branches, table = dispatch
    local = jnp.asarray(table)[function_indices]
    return vmap(lambda i, x: lax.switch(i, branches, x))(local, function_inputs)


def update(
    parameters: jnp.ndarray,
    matrix: jnp.ndarray,
    function_input_indices: jnp.ndarray,
    output_indices: tuple[jnp.ndarray, jnp.ndarray],
    matrix_indices: tuple[jnp.ndarray, jnp.ndarray],
    function_indices: jnp.ndarray,
    dispatch: tuple | None = None,
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
    output_matrix = _evaluate(dispatch or (tuple(FUNCTIONS), np.arange(len(FUNCTIONS))), function_indices, function_inputs)
    outputs = output_matrix[output_indices[0], output_indices[1]]
    return matrix.at[matrix_indices[0], matrix_indices[1]].set(outputs)


def update_values(
    parameters: jnp.ndarray,
    function_input_indices: jnp.ndarray,
    output_indices: tuple[jnp.ndarray, jnp.ndarray],
    function_indices: jnp.ndarray,
    dispatch: tuple | None = None,
) -> jnp.ndarray:
    """
    Like `update`, but returns only the evaluated matrix entries, shape (M,), without scattering
    them into a dense matrix. Entry m belongs at position `matrix_indices[:, m]`.
    """
    function_inputs = parameters[function_input_indices]
    output_matrix = _evaluate(dispatch or (tuple(FUNCTIONS), np.arange(len(FUNCTIONS))), function_indices, function_inputs)
    return output_matrix[output_indices[0], output_indices[1]]


def _last_wins(linear_positions: jnp.ndarray) -> jnp.ndarray:
    """Boolean mask over entries: True where no later entry targets the same position, i.e. the
    entries that survive `.at[].set` (last write wins)."""
    order = jnp.argsort(linear_positions, stable=True)
    sorted_positions = linear_positions[order]
    is_last = jnp.concatenate([sorted_positions[1:] != sorted_positions[:-1], jnp.array([True])])
    return jnp.zeros(linear_positions.shape, dtype=bool).at[order].set(is_last)


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


def _carrier_selections(noise_selection_vectors, qhd_placing_indices, qhd_phase_values, carrier):
    """Detector selection vectors with QHD phases and carrier amplitudes applied, shape (D, SV, SN)."""
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
    return upper_carrier_selections.at[:, :, carrier.shape[1] : carrier.shape[1] * 2].set(lower_carrier_selections)


def _noise_density(quadratic: jnp.ndarray) -> jnp.ndarray:
    """Convert the detector quantum-noise quadratic form (D, SV) into amplitude spectral density."""
    return jnp.sqrt(2 * quadratic * UNIT_VACUUM * H_PLANCK * F0 * 0.25)


def _sideband_conjugation_mask(rows, cols, carrier_size):
    """Entries inside the lower-sideband blocks that must be conjugated (upper/lower sideband symmetry)."""
    c = carrier_size
    mid_row = (rows >= c) & (rows < 2 * c)
    mid_col = (cols >= c) & (cols < 2 * c)
    return (mid_row & mid_col) | ((rows >= 2 * c) & (cols < c)) | (mid_row & (cols >= 2 * c))


def _signal_and_noise_sparse(
    parameters, carrier, signal_matrix, noise_selection_vectors, qhd_placing_indices, qhd_phase_values,
    signal_function_input_indices, signal_function_indices, signal_output_indices, signal_matrix_indices,
    signal_carrier_indices, signal_carrier_placing_indices,
    noise_function_input_indices, noise_function_indices, noise_output_indices, noise_system_matrix_indices,
):
    """
    Signal system and quantum noise without ever building a dense (SV, SN, SN) matrix.

    Verdict 3: the signal matrix is the static `signal_matrix` base plus a COO list of per-frequency
    entries (component outputs, carrier-scaled connector entries); the sparse solvers take exactly
    that. Verdict 2: signal solve, noise solve and both reverse-mode solves share one sparse LU per
    frequency. Verdict 1: the detector quantum noise is the quadratic form w^H N w with
    A^H w = c, because c^H A^{-1} N w = (A^{-H} c)^H N w; the second solve of the old code is gone.
    Verdict 4: the noise source matrix N does not depend on the signal frequency, so it is evaluated
    once and contracted sparsely on its handful of nonzero entries.
    """
    n = signal_matrix.shape[0]
    carrier_size = carrier.shape[1]

    # Per-frequency matrix entries from the component functions: (SV, M).
    vals = vmap(update_values, in_axes=(0, None, None, None, None))(
        parameters, signal_function_input_indices, signal_output_indices, signal_function_indices,
        _SIGNAL_DISPATCH,
    )
    rows = signal_matrix_indices[0]
    cols = jnp.where(signal_matrix_indices[1] < 0, signal_matrix_indices[1] + n + 1, signal_matrix_indices[1])

    # Scale signal connector entries by the carrier solution at the respective spaces.
    # The minus sign matches the sign convention used by the referenced modeling tools.
    placing_rows = signal_carrier_placing_indices[0]
    placing_cols = jnp.where(
        signal_carrier_placing_indices[1] < 0, signal_carrier_placing_indices[1] + n + 1, signal_carrier_placing_indices[1]
    )
    entry_positions = rows * (n + 1) + cols
    order = jnp.argsort(entry_positions, stable=True)
    sorted_positions = entry_positions[order]
    wanted = placing_rows * (n + 1) + placing_cols
    found_at = jnp.clip(jnp.searchsorted(sorted_positions, wanted, side="right") - 1, 0, sorted_positions.shape[0] - 1)
    found = sorted_positions[found_at] == wanted
    current = jnp.where(found[None, :], vals[:, order[found_at]], signal_matrix[placing_rows, placing_cols][None, :])
    scaled = -current * carrier[:, signal_carrier_indices.flatten()]

    rows = jnp.concatenate([rows, placing_rows])
    cols = jnp.concatenate([cols, placing_cols])
    vals = jnp.concatenate([vals, scaled], axis=1)

    # Apply conjugation rules to enforce upper and lower sideband symmetry (base and entries).
    vals = jnp.where(_sideband_conjugation_mask(rows, cols, carrier_size)[None, :], jnp.conjugate(vals), vals)
    base = signal_matrix
    base = base.at[carrier_size : carrier_size * 2, carrier_size : carrier_size * 2].set(
        jnp.conjugate(base[carrier_size : carrier_size * 2, carrier_size : carrier_size * 2])
    )
    base = base.at[carrier_size * 2 :, :carrier_size].set(jnp.conjugate(base[carrier_size * 2 :, :carrier_size]))
    base = base.at[carrier_size : carrier_size * 2, carrier_size * 2 :].set(
        jnp.conjugate(base[carrier_size : carrier_size * 2, carrier_size * 2 :])
    )

    signal = sparse_system_solve(base, rows, cols, vals)

    # Back-propagate detector weights: w = A^{-H} c = conj(A^{-T} conj(c)) with A^T from the same LU.
    carrier_selections = _carrier_selections(noise_selection_vectors, qhd_placing_indices, qhd_phase_values, carrier)
    carrier_selections = jnp.broadcast_to(
        carrier_selections, (carrier_selections.shape[0], vals.shape[0], carrier_selections.shape[2])
    )
    weights = jnp.conjugate(
        sparse_system_transpose_solve(base, rows, cols, vals, jnp.conjugate(carrier_selections))
    )

    # Noise source matrix N: frequency independent (none of the noise functions read the swept
    # frequency), so evaluate it once; sum Re(conj(w_r) N_rc w_c) over its nonzero entries only.
    noise_values = update_values(
        parameters[0], noise_function_input_indices, noise_output_indices, noise_function_indices, _NOISE_DISPATCH
    )
    noise_rows, noise_cols = noise_system_matrix_indices
    noise_values = jnp.where(_last_wins(noise_rows * n + noise_cols), noise_values, 0)
    quadratic = jnp.sum(
        jnp.real(jnp.conjugate(weights[:, :, noise_rows]) * noise_values * weights[:, :, noise_cols]), axis=2
    )
    return signal, _noise_density(quadratic)


def _signal_and_noise_dense(
    parameters, carrier, signal_matrix, noise_matrix, noise_selection_vectors, qhd_placing_indices,
    qhd_phase_values, signal_function_input_indices, signal_function_indices, signal_output_indices,
    signal_matrix_indices, signal_carrier_indices, signal_carrier_placing_indices,
    noise_function_input_indices, noise_function_indices, noise_output_indices, noise_system_matrix_indices,
):
    """Original dense implementation (DIFFEROMETOR_SOLVER=dense): reference for the sparse path."""
    signal_matrix = vmap(update, in_axes=(0, None, None, None, None, None, None))(
        parameters,
        signal_matrix,
        signal_function_input_indices,
        signal_output_indices,
        signal_matrix_indices,
        signal_function_indices,
        _SIGNAL_DISPATCH,
    )
    signal_entries = (
        -signal_matrix[:, signal_carrier_placing_indices[0], signal_carrier_placing_indices[1]]
        * carrier[:, signal_carrier_indices.flatten()]
    )
    signal_matrix = signal_matrix.at[:, signal_carrier_placing_indices[0], signal_carrier_placing_indices[1]].set(
        signal_entries
    )
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
    signal = vmap(solve)(signal_matrix[:, :, :-1], signal_matrix[:, :, -1])

    carrier_selections = _carrier_selections(noise_selection_vectors, qhd_placing_indices, qhd_phase_values, carrier)
    carrier_selections = jnp.broadcast_to(
        carrier_selections, (carrier_selections.shape[0], signal_matrix.shape[0], carrier_selections.shape[2])
    )
    transposed_conjugated_signal_matrix = jnp.transpose(jnp.conjugate(signal_matrix[:, :, :-1]), (0, 2, 1))
    noise_weights = vmap(vmap(solve), in_axes=(None, 0))(transposed_conjugated_signal_matrix, carrier_selections)
    noise_matrix = vmap(update, in_axes=(0, None, None, None, None, None, None))(
        parameters,
        noise_matrix,
        noise_function_input_indices,
        noise_output_indices,
        noise_system_matrix_indices,
        noise_function_indices,
        _NOISE_DISPATCH,
    )
    noise_sources = vmap(lambda m, w: jnp.einsum("bij,bj->bi", m, w), in_axes=(None, 0))(noise_matrix, noise_weights)
    covariances = vmap(vmap(solve), in_axes=(None, 0))(signal_matrix[:, :, :-1], noise_sources)
    temporary_sum = jnp.sum(jnp.real(covariances * jnp.conjugate(carrier_selections)), axis=2)
    return signal, _noise_density(temporary_sum)


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
    carrier_matrix = vmap(update, in_axes=(0, None, None, None, None, None, None))(
        parameters,
        carrier_matrix,
        carrier_function_input_indices,
        carrier_output_indices,
        carrier_matrix_indices,
        carrier_function_indices,
        _CARRIER_DISPATCH,
    )
    carrier = vmap(solve)(carrier_matrix[:, :, :-1], carrier_matrix[:, :, -1])

    # --------------------
    # Signal solve + quantum noise
    # --------------------
    parameters = expand_parameters(parameters, signal_changing_parameter_indices, signal_changing_values)
    if _DENSE:
        signal, noise = _signal_and_noise_dense(
            parameters, carrier, signal_matrix, noise_matrix, noise_selection_vectors, qhd_placing_indices,
            qhd_phase_values, signal_function_input_indices, signal_function_indices, signal_output_indices,
            signal_matrix_indices, signal_carrier_indices, signal_carrier_placing_indices,
            noise_function_input_indices, noise_function_indices, noise_output_indices,
            noise_system_matrix_indices,
        )
    else:
        # The noise source matrix is assumed to start out empty (as `build` always creates it): all
        # its nonzeros are written by the noise component functions.
        signal, noise = _signal_and_noise_sparse(
            parameters, carrier, signal_matrix, noise_selection_vectors, qhd_placing_indices,
            qhd_phase_values, signal_function_input_indices, signal_function_indices, signal_output_indices,
            signal_matrix_indices, signal_carrier_indices, signal_carrier_placing_indices,
            noise_function_input_indices, noise_function_indices, noise_output_indices,
            noise_system_matrix_indices,
        )

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

    if timeit:
        for array in simulation_arrays.values():
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
