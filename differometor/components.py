"""
Optical simulation building blocks (JAX + NumPy).

This module provides small, composable component functions used to build
linear optical simulations with carrier fields, signal sidebands, quantum noise,
and simple optomechanics.

Conventions
-----------
- Angles are specified in **degrees** unless stated otherwise.
  (Several helper docstrings in older versions said "radians"; this module uses
  `jnp.radians(...)` consistently, so inputs are degrees.)
- Frequencies are in hertz.
- Lengths are in meters.
- Refractive indices are unitless.
- Most component functions return a fixed-length complex vector of length 4 so
  that downstream simulation code can stack results into a single matrix.

Output Standard
---------------
`standardize_output` pads shorter outputs with zeros up to length 4 and returns
a `jax.numpy.ndarray` with dtype `complex128`.

Indexing for parameter vectors
------------------------------
Many functions accept `parameters: jnp.ndarray` where each index has a fixed meaning.
Each function docstring specifies the expected indices.

Notes
-----
Some functions mirror Finesse internal implementations (referenced in docstrings)
to match phase conventions and signs.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


# -----------------------------------------------------------------------------
# Physical constants and simulation constants
# -----------------------------------------------------------------------------

EPSILON_0C = 1
"""Scaled permittivity constant used by the simulation.

This is not the physical vacuum permittivity. It is a simulation scaling
constant used consistently across field and power conversions in this code.
"""

C_LIGHT = 299_792_458.0
"""Speed of light in vacuum (meters per second)."""

LAMBDA = 1064e-9
"""Laser wavelength (meters)."""

UNIT_VACUUM = 1
"""Vacuum noise unit scaling used for quantum noise terms."""

# See finesse > simulations > base.pyx > ModelSettings > set_lambda0()
F0 = C_LIGHT / LAMBDA
"""Optical carrier frequency corresponding to `LAMBDA` (hertz)."""

F = 0
"""Offset frequency of the laser (hertz).

This is included for completeness. The current code uses `F0` for the carrier.
"""

H_PLANCK = 6.626_070_15e-34
"""Planck constant (joule seconds)."""

X_SCALE = 1e-09
"""Displacement scaling used in optomechanics (meters).

This typically maps a dimensionless mechanical coordinate to meters.
"""

DEFAULT_REFRACTIVE_INDEX = 1.0
"""Default refractive index for vacuum or air in this simplified model."""

SOFT_SIDE_POWER_THRESHOLD = 2e3
"""Power threshold used for soft-side checks (watts)."""

HARD_SIDE_POWER_THRESHOLD = 3.5e6
"""Power threshold used for hard-side checks (watts)."""

DETECTOR_POWER_THRESHOLD = 1e-2
"""Power threshold used for detector checks (watts)."""


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def standardize_output(output: list) -> jnp.ndarray:
    """
    Pad and cast a component output to the standard length and dtype.

    Parameters
    ----------
    output:
        Python list of scalars (real or complex). If shorter than 4, it is padded
        with zeros to length 4.

    Returns
    -------
    jax.numpy.ndarray
        A length-4 complex vector (`complex128`) suitable for stacking into a
        global simulation matrix.

    Rationale
    ---------
    Many component functions produce 1, 2, or 3 coefficients. Downstream code
    benefits from a uniform output length so that multiple components can be
    evaluated and stacked without shape-dependent branching.
    """
    output_length = len(output)
    if output_length < 4:
        output.extend([0] * (4 - output_length))
    return jnp.array(output, dtype=jnp.complex128)


# -----------------------------------------------------------------------------
# Default properties and parameter bounds
# -----------------------------------------------------------------------------

DEFAULT_PROPERTIES = {
    "frequency": {"frequency": 1},
    "laser": {"power": 1.0, "phase": 0.0},
    "squeezer": {"db": 0, "angle": 90},
    "mirror": {"loss": 5e-6, "reflectivity": 0.5, "tuning": 0.0},
    "beamsplitter": {"loss": 5e-6, "reflectivity": 0.5, "tuning": 0.0, "alpha": 45.0},
    "free_mass": {"mass": 40.0},
    "signal": {"amplitude": 1.0, "phase": 0.0},
    "space": {"length": 0.0, "refractive_index": 1.0},
    "detector": {},
    "qnoised": {},
    "qhd": {"phase": 180.0},
    "nothing": {},
    "directional_beamsplitter": {},
}
"""Default component properties used to populate missing parameters.

Each key corresponds to a component type and maps to a dictionary of parameter
names and default values.
"""

PARAMETER_BOUNDS = {
    "db": [0.01, 20],
    "angle": [-180, 180],
    "power": [0.01, 200],
    "loss": [5e-6, 0.999],
    "tuning": [0, 90],
    "mass": [0.01, 200],
    "length": [1, 4000],
    "reflectivity": [0, 1],
    "transmissivity": [0, 1],
    "phase": [-180, 180],
    "alpha": [-180, 180],
}
"""Numerical bounds for common parameters.

These bounds are intended for optimization or random sampling and should be
interpreted in the same units as the corresponding parameters.
"""


# -----------------------------------------------------------------------------
# Space (propagation)
# -----------------------------------------------------------------------------

def space(parameters: jnp.ndarray) -> jnp.ndarray:
    """
    Optical propagation through a uniform medium of finite length.

    Parameters
    ----------
    parameters:
        Vector with entries:
        - parameters[0]: frequency (hertz)
        - parameters[1]: length (meters)
        - parameters[2]: refractive index (unitless)

    Returns
    -------
    jax.numpy.ndarray
        Length-4 complex vector where:
        - output[0] is the propagation factor (complex phase)
        - output[1:4] are zeros

    Notes
    -----
    The returned factor is:
        phi = -exp(-i * 2π * f * L * n / c)

    The leading minus sign matches the referenced Finesse implementation:
    finesse > components > modal > space.pyx > space_fill_optical_2_optical
    """
    phi = -jnp.exp(-1j * 2 * jnp.pi * parameters[0] * parameters[1] * parameters[2] / C_LIGHT)
    return standardize_output([phi])


def space_lower(parameters: jnp.ndarray) -> jnp.ndarray:
    """
    Propagation for the lower sideband convention.

    This helper flips the sign of the frequency and calls `space`.

    Parameters
    ----------
    parameters:
        Vector with entries:
        - parameters[0]: frequency (hertz)
        - parameters[1]: length (meters)
        - parameters[2]: refractive index (unitless)

    Returns
    -------
    jax.numpy.ndarray
        Same as `space` but evaluated at negative frequency.
    """
    parameters = jnp.array([-parameters[0], parameters[1], parameters[2]])
    return space(parameters)


# -----------------------------------------------------------------------------
# Laser (carrier source)
# -----------------------------------------------------------------------------

def laser(parameters: jnp.ndarray) -> jnp.ndarray:
    """
    Laser carrier field source.

    Parameters
    ----------
    parameters:
        Vector with entries:
        - parameters[0]: power (watts)
        - parameters[1]: phase (degrees)

    Returns
    -------
    jax.numpy.ndarray
        Length-4 complex vector where:
        - output[0] is the complex carrier field amplitude
        - output[1:4] are zeros

    Notes
    -----
    The field amplitude follows:
        field = sqrt(2 * power / EPSILON_0C) * exp(i * phase)
    with `phase` interpreted in degrees.
    """
    field = jnp.sqrt(2 * parameters[0] / EPSILON_0C) * jnp.exp(1j * jnp.radians(parameters[1]))
    return standardize_output([field])


def laser_np(power: float = 1.0, phase: float = 0.0) -> np.ndarray:
    """
    NumPy version of `laser` field amplitude calculation.

    Parameters
    ----------
    power:
        Laser power (watts).
    phase:
        Laser phase (degrees).

    Returns
    -------
    numpy.ndarray
        Complex scalar field amplitude.
    """
    return np.sqrt(2 * power / EPSILON_0C) * np.exp(1j * np.radians(phase))


# -----------------------------------------------------------------------------
# Squeezer (quantum noise transformation)
# -----------------------------------------------------------------------------

def squeezer(parameters: jnp.ndarray) -> jnp.ndarray:
    """
    Quantum noise squeezing element.

    Parameters
    ----------
    parameters:
        Vector with entries:
        - parameters[0]: squeezing level in decibels (decibels)
        - parameters[1]: squeezing angle (degrees)

    Returns
    -------
    jax.numpy.ndarray
        Length-4 complex vector containing a 2x2 noise covariance-like block
        serialized as:
        - output[0]: upper diagonal term
        - output[1]: upper off-diagonal term
        - output[2]: lower diagonal term (complex conjugate of output[0])
        - output[3]: lower off-diagonal term (complex conjugate of output[1])

    Notes
    -----
    This matches the structure used by Finesse:
    finesse > components > modal > squeezer.pyx > c_squeezer_fill_qnoise
    """
    vacuum_unit = UNIT_VACUUM / 2
    squeezing_parameter = parameters[0] / (20 * jnp.log10(jnp.e))
    phase = jnp.exp(1j * 2 * jnp.radians(parameters[1]))

    upper_qn_diagonal = vacuum_unit * jnp.cosh(2 * squeezing_parameter)
    upper_qn_off_diagonal = vacuum_unit * jnp.sinh(2 * squeezing_parameter) * phase
    lower_qn_diagonal = jnp.conjugate(upper_qn_diagonal)
    lower_qn_off_diagonal = jnp.conjugate(upper_qn_off_diagonal)

    return standardize_output(
        [upper_qn_diagonal, upper_qn_off_diagonal, lower_qn_diagonal, lower_qn_off_diagonal]
    )


# -----------------------------------------------------------------------------
# Signals (sideband and modulation sources)
# -----------------------------------------------------------------------------

def signal_function(parameters: jnp.ndarray) -> jnp.ndarray:
    """
    Generic complex signal source.

    Parameters
    ----------
    parameters:
        Vector with entries:
        - parameters[0]: amplitude (arbitrary units)
        - parameters[1]: phase (degrees)

    Returns
    -------
    jax.numpy.ndarray
        Length-4 complex vector where output[0] is the complex signal phasor.
    """
    return standardize_output([parameters[0] * jnp.exp(1j * jnp.radians(parameters[1]))])


def laser_amplitude_modulation(parameters: jnp.ndarray) -> jnp.ndarray:
    """
    Laser amplitude modulation coupling factor.

    Parameters
    ----------
    parameters:
        Unused. Present for signature compatibility.

    Returns
    -------
    jax.numpy.ndarray
        Length-4 complex vector with output[0] equal to EPSILON_0C * 0.5.

    Notes
    -----
    The returned factor is multiplied with the carrier solution during signal
    simulation. It follows the Finesse amplitude modulation implementation:
    finesse > components > modal > laser.pyx > c_laser_fill_signal > SIGAMP_P1o
    """
    factor = EPSILON_0C * 0.5
    return standardize_output([factor])


def laser_frequency_modulation(parameters: jnp.ndarray) -> jnp.ndarray:
    """
    Laser frequency modulation coupling factor.

    Parameters
    ----------
    parameters:
        Vector with entries:
        - parameters[0]: signal frequency (hertz)

    Returns
    -------
    jax.numpy.ndarray
        Length-4 complex vector with output[0] equal to EPSILON_0C * 0.5 / f_signal.

    Notes
    -----
    The returned factor is multiplied with the carrier solution during signal
    simulation. It follows the Finesse frequency modulation implementation:
    finesse > components > modal > laser.pyx > c_laser_fill_signal > SIGFRQ_P1o
    """
    factor = EPSILON_0C * 0.5 / parameters[0]
    return standardize_output([factor])


def laser_frequency_modulation_lower(parameters: jnp.ndarray) -> jnp.ndarray:
    """
    Lower sideband version of `laser_frequency_modulation`.

    Parameters
    ----------
    parameters:
        Vector with entries:
        - parameters[0]: signal frequency (hertz)

    Returns
    -------
    jax.numpy.ndarray
        Same as `laser_frequency_modulation` but evaluated at negative frequency.
    """
    parameters = jnp.array([-parameters[0]])
    return laser_frequency_modulation(parameters)


def space_modulation(parameters: jnp.ndarray) -> jnp.ndarray:
    """
    Space (strain or length) modulation sideband factor.

    Parameters
    ----------
    parameters:
        Vector with entries:
        - parameters[0]: signal frequency (hertz)
        - parameters[1]: length (meters)
        - parameters[2]: refractive index (unitless)

    Returns
    -------
    jax.numpy.ndarray
        Length-4 complex vector where output[0] is the complex modulation factor.

    Notes
    -----
    This implements the plane-wave strain modulation model described in
    https://arxiv.org/abs/1306.6752 (equations referenced in comments).

    The returned factor excludes the signal amplitude and phase because those
    are applied later in the signal injection step.
    """
    # Actual modulation angular frequency (not an offset).
    w_g = 2 * jnp.pi * parameters[0]
    # Carrier angular frequency.
    w_0 = 2 * jnp.pi * F0

    # Equation 14 without the signal amplitude factor.
    m_g = -0.5 * w_0 / w_g * jnp.sin(w_g * parameters[1] * parameters[2] / 2 / C_LIGHT)

    # Equation 15 without the signal phase factor.
    phi_sb = -w_g * parameters[1] * parameters[2] / 2 / C_LIGHT

    # The imaginary unit corresponds to a pi/2 phase shift (equation 17).
    z = 1j * m_g * jnp.exp(1j * phi_sb)
    return standardize_output([z])


def space_modulation_lower(parameters: jnp.ndarray) -> jnp.ndarray:
    """
    Lower sideband version of `space_modulation`.

    Parameters
    ----------
    parameters:
        Vector with entries:
        - parameters[0]: signal frequency (hertz)
        - parameters[1]: length (meters)
        - parameters[2]: refractive index (unitless)

    Returns
    -------
    jax.numpy.ndarray
        Same as `space_modulation` but evaluated at negative frequency.
    """
    parameters = jnp.array([-parameters[0], parameters[1], parameters[2]])
    return space_modulation(parameters)


# -----------------------------------------------------------------------------
# Optomechanics
# -----------------------------------------------------------------------------

def susceptibility(parameters: jnp.ndarray) -> jnp.ndarray:
    """
    Free-mass mechanical susceptibility.

    Parameters
    ----------
    parameters:
        Vector with entries:
        - parameters[0]: signal frequency (hertz)
        - parameters[1]: mass (kilograms)

    Returns
    -------
    jax.numpy.ndarray
        Length-4 complex vector where output[0] is the susceptibility (meters per newton).

    Notes
    -----
    For a free mass:
        chi = -1 / (m * (2π f)^2)

    An extra minus sign is applied to match the sign convention used in the
    referenced implementation:
    finesse > components > mechanical.pyx > FreeMass > fill
    """
    factor = -1 / (parameters[1] * (2 * jnp.pi * parameters[0]) ** 2)
    return -standardize_output([factor])


def force_calculation_left(parameters: jnp.ndarray) -> jnp.ndarray:
    """
    Radiation pressure force conversion (left side).

    Parameters
    ----------
    parameters:
        Vector with entries:
        - parameters[0]: incidence angle alpha (degrees)

    Returns
    -------
    jax.numpy.ndarray
        Length-4 complex vector where output[0] is a scaling factor converting
        optical field power to force, including geometry and displacement scaling.

    Notes
    -----
    References:
    - finesse > components > modal > mirror.pyx > c_mirror_signal_mech_fill, ws.field_to_F
    - finesse > components > modal > beamsplitter.pyx > c_beamsplitter_signal_fill, ws.field1_to_F
    """
    return -standardize_output([incidence_angle_left(parameters[0]) / (C_LIGHT * X_SCALE)])


def force_calculation_right(parameters: jnp.ndarray) -> jnp.ndarray:
    """
    Radiation pressure force conversion (right side).

    Parameters
    ----------
    parameters:
        Vector with entries:
        - parameters[0]: incidence angle alpha (degrees)
        - parameters[1]: refractive index on the left side (unitless)
        - parameters[2]: refractive index on the right side (unitless)

    Returns
    -------
    jax.numpy.ndarray
        Length-4 complex vector where output[0] is a scaling factor converting
        optical field power to force.

    Notes
    -----
    References:
    - finesse > components > modal > mirror.pyx > c_mirror_signal_mech_fill, ws.field_to_F
    - finesse > components > modal > beamsplitter.pyx > c_beamsplitter_signal_fill, ws.field2_to_F
    """
    return standardize_output(
        [incidence_angle_right(parameters[0], parameters[1], parameters[2]) / (C_LIGHT * X_SCALE)]
    )


def optomechanical_phase_left(alpha: float) -> jnp.ndarray:
    """
    Optical phase response to displacement (left side).

    Parameters
    ----------
    alpha:
        Incidence angle (degrees).

    Returns
    -------
    jax.numpy.ndarray
        Complex scalar factor representing phase change per scaled displacement.

    Notes
    -----
    Uses:
        i * X_SCALE * (2π / lambda) * cos(alpha)
    matching:
    - finesse > components > modal > mirror.pyx > c_mirror_signal_mech_fill, ws.z_to_field
    - finesse > components > modal > beamsplitter.pyx > c_beamsplitter_signal_fill, ws.z_to_field1
    """
    return 1j * X_SCALE * 2 * jnp.pi / LAMBDA * incidence_angle_left(alpha)


def optomechanical_phase_right(
    alpha: float, refractive_index_left: float, refractive_index_right: float
) -> jnp.ndarray:
    """
    Optical phase response to displacement (right side).

    Parameters
    ----------
    alpha:
        Incidence angle (degrees).
    refractive_index_left:
        Refractive index on the left side (unitless).
    refractive_index_right:
        Refractive index on the right side (unitless).

    Returns
    -------
    jax.numpy.ndarray
        Complex scalar factor representing phase change per scaled displacement.

    Notes
    -----
    Uses Snell's law to compute the transmitted angle contribution.
    References:
    - finesse > components > modal > mirror.pyx > c_mirror_signal_mech_fill, ws.z_to_field
    - finesse > components > modal > beamsplitter.pyx > c_beamsplitter_signal_fill, ws.z_to_field2
    """
    return (
        1j
        * X_SCALE
        * 2
        * jnp.pi
        / LAMBDA
        * incidence_angle_right(alpha, refractive_index_left, refractive_index_right)
    )


def tuning_correction(parameters: jnp.ndarray) -> jnp.ndarray:
    """
    Frequency-dependent tuning correction for mechanical sidebands.

    Parameters
    ----------
    parameters:
        Vector with entries:
        - parameters[0]: signal frequency (hertz)
        - parameters[1]: tuning (degrees)

    Returns
    -------
    jax.numpy.ndarray
        Complex scalar correction factor.

    Notes
    -----
    Matches:
    finesse > components > modal > mirror.pyx > single_z_mechanical_frequency_signal_calc
    """
    return jnp.exp(1j * jnp.radians(parameters[1]) * parameters[0] / F0)


def corrected_optomechanical_phase_left(parameters: jnp.ndarray) -> jnp.ndarray:
    """
    Optomechanical phase coupling including tuning and reflectivity (left side).

    Parameters
    ----------
    parameters:
        Vector with entries:
        - parameters[0]: signal frequency (hertz)
        - parameters[1]: tuning (degrees)
        - parameters[2]: reflectivity fraction (unitless, between 0 and 1)
        - parameters[3]: loss (unitless, between 0 and 1)
        - parameters[4]: refractive index on the left side (unitless)
        - parameters[5]: incidence angle alpha (degrees)

    Returns
    -------
    jax.numpy.ndarray
        Length-4 complex vector where output[0] is the corrected coupling factor.

    Notes
    -----
    This includes:
    - tuning correction
    - optomechanical phase per displacement
    - surface reflectivity phase factor
    """
    absolute_reflectivity = (1 - parameters[3]) * parameters[2]
    factor = (
        tuning_correction(parameters)
        * optomechanical_phase_left(parameters[5])
        * reflectivity_left(absolute_reflectivity, parameters[1], parameters[0], parameters[4], parameters[5])
    )
    return -standardize_output([factor])


def corrected_optomechanical_phase_right(parameters: jnp.ndarray) -> jnp.ndarray:
    """
    Optomechanical phase coupling including tuning and reflectivity (right side).

    Parameters
    ----------
    parameters:
        Vector with entries:
        - parameters[0]: signal frequency (hertz)
        - parameters[1]: tuning (degrees)
        - parameters[2]: reflectivity fraction (unitless, between 0 and 1)
        - parameters[3]: loss (unitless, between 0 and 1)
        - parameters[4]: refractive index on the left side (unitless)
        - parameters[5]: refractive index on the right side (unitless)
        - parameters[6]: incidence angle alpha (degrees)

    Returns
    -------
    jax.numpy.ndarray
        Length-4 complex vector where output[0] is the corrected coupling factor.

    Notes
    -----
    The implementation uses:
    - complex conjugated tuning correction
    - an additional minus sign (matches referenced convention)
    - right-side optomechanical phase and reflectivity factor
    """
    absolute_reflectivity = (1 - parameters[3]) * parameters[2]
    factor = (
        jnp.conj(tuning_correction(parameters))
        * -1
        * optomechanical_phase_right(parameters[6], parameters[4], parameters[5])
        * reflectivity_right(
            absolute_reflectivity,
            parameters[1],
            parameters[0],
            parameters[4],
            parameters[5],
            parameters[6],
        )
    )
    return -standardize_output([factor])


# -----------------------------------------------------------------------------
# Quantum noise
# -----------------------------------------------------------------------------

def vacuum_quantum_noise(parameters: jnp.ndarray) -> jnp.ndarray:
    """
    Vacuum quantum noise source.

    Parameters
    ----------
    parameters:
        Unused. Present for signature compatibility.

    Returns
    -------
    jax.numpy.ndarray
        Length-4 complex vector where output[0] is UNIT_VACUUM / 2.

    Notes
    -----
    References:
    - finesse > components > modal > laser.pyx > c_laser_fill_qnoise
    - finesse > components > workspace.pyx > c_optical_quantum_noise_plane_wave
    """
    quantum_noise = UNIT_VACUUM / 2
    return standardize_output([quantum_noise])


def loss_quantum_noise(parameters: jnp.ndarray) -> jnp.ndarray:
    """
    Quantum noise injected by optical loss.

    Parameters
    ----------
    parameters:
        Vector with entries:
        - parameters[0]: loss (unitless, between 0 and 1)

    Returns
    -------
    jax.numpy.ndarray
        Length-4 complex vector where output[0] is loss / 2.

    Notes
    -----
    References:
    - finesse > components > modal > mirror.pyx > c_mirror_fill_qnoise
    - finesse > components > modal > beamsplitter.pyx > c_beamsplitter_fill_qnoise
    """
    quantum_noise = parameters[0] / 2
    return standardize_output([quantum_noise])


# -----------------------------------------------------------------------------
# Detectors and demodulation
# -----------------------------------------------------------------------------

def amplitude_detector(solution: jnp.ndarray) -> jnp.ndarray:
    """
    Convert complex field amplitude to amplitude readout.

    Parameters
    ----------
    solution:
        Complex field solution (any shape). The absolute value is taken elementwise.

    Returns
    -------
    jax.numpy.ndarray
        Amplitude readout with the same shape as `solution`.

    Notes
    -----
    Uses the scaling:
        sqrt(0.5 * EPSILON_0C) * |field|
    """
    return jnp.sqrt(0.5 * EPSILON_0C) * jnp.abs(solution)


def power_detector(solution: jnp.ndarray) -> jnp.ndarray:
    """
    Convert complex field amplitude to optical power.

    Parameters
    ----------
    solution:
        Complex field solution (any shape). The magnitude squared is taken elementwise.

    Returns
    -------
    jax.numpy.ndarray
        Power readout with the same shape as `solution`.

    Notes
    -----
    Uses the scaling:
        0.5 * EPSILON_0C * |field|^2
    """
    return 0.5 * EPSILON_0C * jnp.abs(solution) ** 2


def signal_detector(carrier: jnp.ndarray, signal: jnp.ndarray) -> jnp.ndarray:
    """
    Demodulate optical power for a single-frequency signal step.

    Parameters
    ----------
    carrier:
        Complex carrier field solution, shape (N,).
    signal:
        Complex signal sideband vector, shape (2N,). The convention is:
        - signal[:N]   are upper sidebands
        - signal[N:2N] are lower sidebands

    Returns
    -------
    jax.numpy.ndarray
        Complex demodulated signal, shape (N,).

    Notes
    -----
    Matches:
    finesse > detectors > compute > power.pyx > c_pd1_AC_output
    """
    upper_sideband = jnp.conj(carrier) * signal[: carrier.shape[0]]
    lower_sideband = carrier * signal[carrier.shape[0] : carrier.shape[0] * 2]
    return upper_sideband + lower_sideband


# -----------------------------------------------------------------------------
# Surface, mirror, and beamsplitter models
# -----------------------------------------------------------------------------
"""
Surface parameterization note
-----------------------------
Given `loss` and `reflectivity`:
- absolute_reflectivity = (1 - loss) * reflectivity
- absolute_transmissivity = (1 - loss) * (1 - reflectivity)

This couples reflectivity and transmissivity through loss.

For optimization, a more independent parameterization can be used:
- parameter A in [0, 1] sets the loss fraction
- parameter B in [0, 1] sets reflectivity fraction of the remaining power
Then:
    loss = A
    reflectivity = (1 - A) * B
    transmissivity = (1 - A) * (1 - B)
"""

def surface(parameters: jnp.ndarray) -> jnp.ndarray:
    """
    Optical surface coupling coefficients (reflection and transmission).

    Parameters
    ----------
    parameters:
        Vector with entries:
        - parameters[0]: loss (unitless, between 0 and 1)
        - parameters[1]: reflectivity fraction of remaining power (unitless, between 0 and 1)
        - parameters[2]: tuning (degrees)
        - parameters[3]: frequency (hertz)
        - parameters[4]: refractive index on the left side (unitless)
        - parameters[5]: refractive index on the right side (unitless)
        - parameters[6]: incidence angle alpha (degrees)

    Returns
    -------
    jax.numpy.ndarray
        Length-4 complex vector:
        - output[0]: reflection coefficient for left incidence
        - output[1]: transmission coefficient
        - output[2]: reflection coefficient for right incidence
        - output[3]: zero

    Notes
    -----
    This is a compact form used for assembling larger scattering matrices.
    """
    absolute_reflectivity = (1 - parameters[0]) * parameters[1]
    absolute_transmissivity = (1 - parameters[0]) * (1 - parameters[1])

    reflectivity_left_entry = reflectivity_left(
        absolute_reflectivity, parameters[2], parameters[3], parameters[4], parameters[6]
    )
    transmissivity_entry = transmissivity(
        absolute_transmissivity,
        parameters[2],
        parameters[3],
        parameters[4],
        parameters[5],
        parameters[6],
    )
    reflectivity_right_entry = reflectivity_right(
        absolute_reflectivity,
        parameters[2],
        parameters[3],
        parameters[4],
        parameters[5],
        parameters[6],
    )

    return standardize_output([reflectivity_left_entry, transmissivity_entry, reflectivity_right_entry])


# -----------------------------------------------------------------------------
# NumPy twins of the scalar surface formulas (used while *building* matrices)
# -----------------------------------------------------------------------------
# `build` evaluates these a few hundred times with plain Python floats. Eager `jnp` scalar ops cost
# a separate XLA executable compile per distinct primitive on first use (~2 s in total), so the
# build-time matrices use NumPy. Same formulas as `reflectivity_left` & co. above.

def _cos_left_np(alpha):
    return np.cos(np.radians(alpha))


def _cos_right_np(alpha, refractive_index_left, refractive_index_right):
    return np.cos(np.arcsin(refractive_index_left / refractive_index_right * np.sin(np.radians(alpha))))


def _phase_shift_np(tuning, frequency, refractive_index):
    return 2 * np.radians(tuning) * refractive_index * (1 + frequency / F0)


def _reflectivity_left_np(reflectivity, tuning, frequency, refractive_index_left, alpha=0.0):
    return -np.sqrt(reflectivity) * np.exp(
        1j * _phase_shift_np(tuning, frequency, refractive_index_left) * _cos_left_np(alpha)
    )


def _reflectivity_right_np(reflectivity, tuning, frequency, refractive_index_left, refractive_index_right=1.0, alpha=0.0):
    return -np.sqrt(reflectivity) * np.exp(
        -1j
        * _phase_shift_np(tuning, frequency, refractive_index_right)
        * _cos_right_np(alpha, refractive_index_right, refractive_index_left)
    )


def _transmissivity_np(transmissivity, tuning, frequency, refractive_index_left, refractive_index_right, alpha=0.0):
    left_term = _phase_shift_np(tuning, frequency, refractive_index_left) * _cos_left_np(alpha)
    right_term = _phase_shift_np(tuning, frequency, refractive_index_right) * _cos_right_np(
        alpha, refractive_index_left, refractive_index_right
    )
    return -np.sqrt(transmissivity) * np.exp(1j * (np.pi / 2 + 0.5 * (left_term - right_term)))


def space_entry_np(frequency, length, refractive_index):
    """`space(...)[0]` as a NumPy scalar: phi = -exp(-i * 2 pi f L n / c)."""
    return -np.exp(-1j * 2 * np.pi * frequency * length * refractive_index / C_LIGHT)


def mirror_matrix(
    loss: float = 0.0,
    reflectivity: float = 0.5,
    tuning: float = 0.0,
    frequency: float = 0.0,
    refractive_index_left: float = 1.0,
    refractive_index_right: float = 1.0,
) -> np.ndarray:
    """
    Build the 4x4 scattering matrix for a two-port mirror.

    Parameters
    ----------
    loss:
        Power loss (unitless, between 0 and 1).
    reflectivity:
        Reflectivity fraction of remaining power (unitless, between 0 and 1).
    tuning:
        Microscopic tuning phase (degrees).
    frequency:
        Frequency offset used in phase terms (hertz).
    refractive_index_left:
        Refractive index on the left side (unitless).
    refractive_index_right:
        Refractive index on the right side (unitless).

    Returns
    -------
    numpy.ndarray
        A 4x4 matrix mapping inputs to outputs in the ordering:
        [left input, left output, right input, right output].

    Notes
    -----
    The matrix layout matches the component ordering used by this simulation
    framework. Reflection and transmission coefficients follow the helper
    functions `reflectivity_left`, `reflectivity_right`, and `transmissivity`.
    """
    absolute_reflectivity = (1 - loss) * reflectivity
    absolute_transmissivity = (1 - loss) * (1 - reflectivity)

    reflectivity_left_to_left = _reflectivity_left_np(
        absolute_reflectivity, tuning, frequency, refractive_index_left
    )
    transmissivity_entry = _transmissivity_np(
        absolute_transmissivity, tuning, frequency, refractive_index_left, refractive_index_right
    )
    reflectivity_right_to_right = _reflectivity_right_np(
        absolute_reflectivity, tuning, frequency, refractive_index_left, refractive_index_right
    )

    return np.array(
        [
            [1, 0, 0, 0],  # left input
            [reflectivity_left_to_left, 1, transmissivity_entry, 0],  # left output
            [0, 0, 1, 0],  # right input
            [transmissivity_entry, 0, reflectivity_right_to_right, 1],  # right output
        ]
    )


def beamsplitter_matrix(
    loss: float = 0.0,
    reflectivity: float = 0.5,
    tuning: float = 0.0,
    frequency: float = 0.0,
    refractive_index_left: float = 1.0,
    refractive_index_right: float = 1.0,
    alpha: float = 0.0,
) -> np.ndarray:
    """
    Build the 8x8 scattering matrix for a four-port beamsplitter.

    Parameters
    ----------
    loss:
        Power loss (unitless, between 0 and 1).
    reflectivity:
        Reflectivity fraction of remaining power (unitless, between 0 and 1).
    tuning:
        Microscopic tuning phase (degrees).
    frequency:
        Frequency offset used in phase terms (hertz).
    refractive_index_left:
        Refractive index for the left and top sides (unitless).
    refractive_index_right:
        Refractive index for the right and bottom sides (unitless).
    alpha:
        Incidence angle (degrees) used for projection factors and phase terms.

    Returns
    -------
    numpy.ndarray
        An 8x8 matrix mapping inputs to outputs in the ordering:
        [left in, left out, top in, top out, right in, right out, bottom in, bottom out].

    Notes
    -----
    The coefficient placement matches the port mapping in this framework.
    """
    absolute_reflectivity = (1 - loss) * reflectivity
    absolute_transmissivity = (1 - loss) * (1 - reflectivity)

    reflectivity_left_entry = _reflectivity_left_np(
        absolute_reflectivity, tuning, frequency, refractive_index_left, alpha
    )
    transmissivity_entry = _transmissivity_np(
        absolute_transmissivity,
        tuning,
        frequency,
        refractive_index_left,
        refractive_index_right,
        alpha,
    )
    reflectivity_right_entry = _reflectivity_right_np(
        absolute_reflectivity,
        tuning,
        frequency,
        refractive_index_left,
        refractive_index_right,
        alpha,
    )

    return np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],  # left input
            [0, 1, reflectivity_left_entry, 0, transmissivity_entry, 0, 0, 0],  # left output
            [0, 0, 1, 0, 0, 0, 0, 0],  # top input
            [reflectivity_left_entry, 0, 0, 1, 0, 0, transmissivity_entry, 0],  # top output
            [0, 0, 0, 0, 1, 0, 0, 0],  # right input
            [transmissivity_entry, 0, 0, 0, 0, 1, reflectivity_right_entry, 0],  # right output
            [0, 0, 0, 0, 0, 0, 1, 0],  # bottom input
            [0, 0, transmissivity_entry, 0, reflectivity_right_entry, 0, 0, 1],  # bottom output
        ]
    )


def directional_beamsplitter_matrix() -> np.ndarray:
    """
    Build the 8x8 routing matrix for an ideal directional beamsplitter.

    Port mapping
    ------------
    - Left input  -> Right output
    - Top input   -> Left output
    - Right input -> Bottom output
    - Bottom input -> Top output

    Returns
    -------
    numpy.ndarray
        An 8x8 matrix implementing the above port routing with fixed sign
        conventions.
    """
    return np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],  # left input
            [0, 1, -1, 0, 0, 0, 0, 0],  # left output
            [0, 0, 1, 0, 0, 0, 0, 0],  # top input
            [0, 0, 0, 1, 0, 0, -1, 0],  # top output
            [0, 0, 0, 0, 1, 0, 0, 0],  # right input
            [-1, 0, 0, 0, 0, 1, 0, 0],  # right output
            [0, 0, 0, 0, 0, 0, 1, 0],  # bottom input
            [0, 0, 0, 0, -1, 0, 0, 1],  # bottom output
        ]
    )


def nothing_matrix() -> np.ndarray:
    """
    Build the 4x4 routing matrix for a direct connection (no optical element).

    Port mapping
    ------------
    - Left input  -> Right output
    - Right input -> Left output

    Returns
    -------
    numpy.ndarray
        A 4x4 matrix implementing the above routing with fixed sign conventions.
    """
    return np.array(
        [
            [1, 0, 0, 0],  # left input
            [0, 1, -1, 0],  # left output
            [0, 0, 1, 0],  # right input
            [-1, 0, 0, 1],  # right output
        ]
    )


# -----------------------------------------------------------------------------
# Helper functions (angles, phases, Fresnel-like coefficients)
# -----------------------------------------------------------------------------

def incidence_angle_left(alpha: float) -> jnp.ndarray:
    """
    Geometric projection factor for the left side.

    Parameters
    ----------
    alpha:
        Incidence angle (degrees).

    Returns
    -------
    jax.numpy.ndarray
        cos(alpha) as a scalar array.

    Notes
    -----
    Used for phase and radiation pressure scaling. Matches the plane-wave
    beamsplitter workspace convention in:
    finesse > components > modal > beamsplitter.pyx > BeamsplitterWorkspace
    """
    return jnp.cos(jnp.radians(alpha))


def incidence_angle_right(alpha: float, refractive_index_left: float, refractive_index_right: float) -> jnp.ndarray:
    """
    Geometric projection factor for the right side using Snell's law.

    Parameters
    ----------
    alpha:
        Incidence angle on the left side (degrees).
    refractive_index_left:
        Refractive index on the incident side (unitless).
    refractive_index_right:
        Refractive index on the transmitted side (unitless).

    Returns
    -------
    jax.numpy.ndarray
        cos(beta) where beta is the transmitted angle computed from Snell's law.

    Notes
    -----
    beta is obtained from:
        n_left * sin(alpha) = n_right * sin(beta)

    The cosine is returned because it is the projection used in phase and force
    terms.

    Related documentation:
    https://finesse.ifosim.org/docs/latest/physics/plane-waves/beam_splitter.html#beamsplitter-phase
    """
    return jnp.cos(
        jnp.arcsin(refractive_index_left / refractive_index_right * jnp.sin(jnp.radians(alpha)))
    )


def phase_shift(tuning: float, frequency: float, refractive_index: float) -> jnp.ndarray:
    """
    Microscopic phase shift used for surface coefficients.

    Parameters
    ----------
    tuning:
        Tuning phase (degrees).
    frequency:
        Frequency offset (hertz).
    refractive_index:
        Refractive index for the relevant side (unitless).

    Returns
    -------
    jax.numpy.ndarray
        Scalar phase shift in radians.

    Notes
    -----
    The returned value is:
        2 * radians(tuning) * n * (1 + frequency / F0)

    This matches the surface phase factor used by Finesse:
    - finesse > components > modal > mirror.pyx > mirror_fill_optical_2_optical
    - finesse > components > modal > beamsplitter.pyx > beamsplitter_fill_optical_2_optical
    """
    return 2 * jnp.radians(tuning) * refractive_index * (1 + frequency / F0)


def reflectivity_left(
    reflectivity: float,
    tuning: float,
    frequency: float,
    refractive_index_left: float,
    alpha: float = 0.0,
) -> jnp.ndarray:
    """
    Complex reflection coefficient for incidence from the left side.

    Parameters
    ----------
    reflectivity:
        Absolute reflectivity (unitless). This is already scaled by (1 - loss).
    tuning:
        Tuning phase (degrees).
    frequency:
        Frequency offset (hertz).
    refractive_index_left:
        Refractive index on the left side (unitless).
    alpha:
        Incidence angle (degrees).

    Returns
    -------
    jax.numpy.ndarray
        Complex scalar reflection coefficient.
    """
    return -jnp.sqrt(reflectivity) * jnp.exp(
        1j * phase_shift(tuning, frequency, refractive_index_left) * incidence_angle_left(alpha)
    )


def reflectivity_right(
    reflectivity: float,
    tuning: float,
    frequency: float,
    refractive_index_left: float,
    refractive_index_right: float = 1.0,
    alpha: float = 0.0,
) -> jnp.ndarray:
    """
    Complex reflection coefficient for incidence from the right side.

    Parameters
    ----------
    reflectivity:
        Absolute reflectivity (unitless). This is already scaled by (1 - loss).
    tuning:
        Tuning phase (degrees).
    frequency:
        Frequency offset (hertz).
    refractive_index_left:
        Refractive index on the left side (unitless).
    refractive_index_right:
        Refractive index on the right side (unitless).
    alpha:
        Incidence angle (degrees).

    Returns
    -------
    jax.numpy.ndarray
        Complex scalar reflection coefficient.

    Notes
    -----
    The right-side incidence angle projection is computed with Snell's law and
    applied to the phase.
    """
    return -jnp.sqrt(reflectivity) * jnp.exp(
        -1j
        * phase_shift(tuning, frequency, refractive_index_right)
        * incidence_angle_right(alpha, refractive_index_right, refractive_index_left)
    )


def transmissivity(
    transmissivity: float,
    tuning: float,
    frequency: float,
    refractive_index_left: float,
    refractive_index_right: float,
    alpha: float = 0.0,
) -> jnp.ndarray:
    """
    Complex transmission coefficient across a surface.

    Parameters
    ----------
    transmissivity:
        Absolute transmissivity (unitless). This is already scaled by (1 - loss).
    tuning:
        Tuning phase (degrees).
    frequency:
        Frequency offset (hertz).
    refractive_index_left:
        Refractive index on the left side (unitless).
    refractive_index_right:
        Refractive index on the right side (unitless).
    alpha:
        Incidence angle (degrees).

    Returns
    -------
    jax.numpy.ndarray
        Complex scalar transmission coefficient.

    Notes
    -----
    Includes a fixed quadrature phase (pi/2) and a differential phase term between
    left and right sides to match the Finesse plane-wave convention.
    """
    left_term = phase_shift(tuning, frequency, refractive_index_left) * incidence_angle_left(alpha)
    right_term = phase_shift(tuning, frequency, refractive_index_right) * incidence_angle_right(
        alpha, refractive_index_left, refractive_index_right
    )
    return -jnp.sqrt(transmissivity) * jnp.exp(1j * (jnp.pi / 2 + 0.5 * (left_term - right_term)))


# -----------------------------------------------------------------------------
# Miscellaneous
# -----------------------------------------------------------------------------

def dummy_function(parameters: jnp.ndarray) -> jnp.ndarray:
    """
    Placeholder component function.

    Parameters
    ----------
    parameters:
        Unused. Present for signature compatibility.

    Returns
    -------
    jax.numpy.ndarray
        Length-4 complex vector with output[0] equal to 1.
    """
    return standardize_output([1.0])


# -----------------------------------------------------------------------------
# Component function registry
# -----------------------------------------------------------------------------

FUNCTIONS = [
    force_calculation_right,
    laser,
    surface,
    space,
    space_modulation,
    signal_function,
    dummy_function,
    vacuum_quantum_noise,
    loss_quantum_noise,
    space_lower,
    laser_amplitude_modulation,
    laser_frequency_modulation,
    susceptibility,
    force_calculation_left,
    corrected_optomechanical_phase_left,
    corrected_optomechanical_phase_right,
    space_modulation_lower,
    squeezer,
    laser_frequency_modulation_lower,
]
"""Ordered list of component functions used by the simulation engine.

The order is important if the caller uses integer indices to reference a
component model.
"""


# -----------------------------------------------------------------------------
# Branch-free evaluation of all component functions
# -----------------------------------------------------------------------------

def _function_id(function) -> int:
    return FUNCTIONS.index(function)


def _mask_table(groups: list) -> np.ndarray:
    """Static (len(FUNCTIONS), len(groups)) bool table: table[f, g] = FUNCTIONS[f] in groups[g]."""
    table = np.zeros((len(FUNCTIONS), len(groups)), dtype=bool)
    for column, group in enumerate(groups):
        for function in group:
            table[_function_id(function), column] = True
    return table


def _select(function_indices: jnp.ndarray, cases: list, zero: jnp.ndarray) -> jnp.ndarray:
    """Per-row value chosen by function id: `cases` is a list of (functions, value-array) pairs;
    rows whose function is in no case get `zero`. One gather instead of a chain of compares/selects."""
    index_of_function = np.zeros(len(FUNCTIONS), dtype=np.int32)
    for position, (functions, _) in enumerate(cases, start=1):
        for function in functions:
            index_of_function[_function_id(function)] = position
    candidates = jnp.stack([zero] + [value for _, value in cases], axis=-1)
    chosen = jnp.asarray(index_of_function)[function_indices]
    return jnp.take_along_axis(candidates, chosen[:, None], axis=-1)[:, 0]


def evaluate_components(function_indices: jnp.ndarray, inputs: jnp.ndarray) -> jnp.ndarray:
    """
    Evaluate `FUNCTIONS[function_indices[k]](inputs[k])` for every row k, without `lax.switch`.

    Parameters
    ----------
    function_indices:
        Integer indices into `FUNCTIONS`, shape (K,). May be traced.
    inputs:
        Function input rows, shape (K, 7) (complex or real; the physical parameters are real, so
        only the real part is used).

    Returns
    -------
    jax.numpy.ndarray
        Complex128 array of shape (K, 4), identical to what the individual functions return.

    Notes
    -----
    `vmap(lax.switch(i, FUNCTIONS, x))` with a batched index makes XLA evaluate every branch on every
    row and select, and reverse-mode AD differentiates every branch again; on one CPU that is tens of
    seconds of XLA compile time. Every component function is a few real scalar formulas that end in
    `amplitude * exp(i * phase)` per output slot, so this evaluates one shared formula set, selects
    amplitude and phase per slot by function id (static lookup tables, one gather per slot), and takes
    ONE cos/sin per slot. All risky operations (sqrt, divisions, exp) only ever see benign values on
    rows that do not use them, so masked-out candidates cannot inject NaN/inf into reverse-mode
    gradients. Verified row-by-row against `FUNCTIONS` in tests/test_components.py.
    """
    p = jnp.real(inputs).astype(jnp.float64)
    fid = function_indices
    zero, one = jnp.zeros_like(p[:, 0]), jnp.ones_like(p[:, 0])
    half = 0.5 * one

    groups = [
        (surface,),
        (corrected_optomechanical_phase_left,),
        (corrected_optomechanical_phase_right,),
        (force_calculation_left,),
        (force_calculation_right,),
        (space, space_lower),
        (space_modulation, space_modulation_lower),
        (space_lower, space_modulation_lower, laser_frequency_modulation_lower),
        (laser,),
        (signal_function,),
        (laser_frequency_modulation, laser_frequency_modulation_lower),
        (susceptibility,),
        (squeezer,),
        (surface, corrected_optomechanical_phase_left, corrected_optomechanical_phase_right),  # sqrt(R) users
        (
            surface,
            corrected_optomechanical_phase_left,
            corrected_optomechanical_phase_right,
            force_calculation_left,
            force_calculation_right,
        ),  # mirror family
    ]
    masks = jnp.asarray(_mask_table(groups))[fid]  # (K, len(groups)): one gather for all masks
    (m_surface, m_left, m_right, m_force_l, m_force_r, m_space, m_space_mod, m_lower, m_laser, m_signal,
     m_fm, m_chi, m_squeezer, m_sqrt_r, m_family) = (masks[:, i] for i in range(len(groups)))

    def guard(mask, value, fallback):
        return jnp.where(mask, value, fallback)

    # --- canonical (loss, reflectivity, tuning, frequency, n_left, n_right, alpha) of the mirror family
    loss = jnp.where(m_surface, p[:, 0], jnp.where(m_left | m_right, p[:, 3], zero))
    reflectivity = jnp.where(m_surface, p[:, 1], jnp.where(m_left | m_right, p[:, 2], half))
    tuning = jnp.where(m_surface, p[:, 2], jnp.where(m_left | m_right, p[:, 1], zero))
    frequency = jnp.where(m_surface, p[:, 3], jnp.where(m_left | m_right, p[:, 0], one))
    n_left = jnp.where(m_surface | m_left | m_right, p[:, 4], jnp.where(m_force_r, p[:, 1], one))
    n_right = jnp.where(m_surface | m_right, p[:, 5], jnp.where(m_force_r, p[:, 2], one))
    alpha = jnp.where(
        m_surface | m_right, p[:, 6], jnp.where(m_left, p[:, 5], jnp.where(m_force_l | m_force_r, p[:, 0], zero))
    )
    n_left, n_right = guard(m_family, n_left, one), guard(m_family, n_right, one)
    alpha = guard(m_family, alpha, zero)
    loss, reflectivity = guard(m_family, loss, zero), guard(m_family, reflectivity, half)
    tuning, frequency = guard(m_family, tuning, zero), guard(m_family, frequency, one)

    sin_alpha = jnp.sin(jnp.radians(alpha))
    cos_left = jnp.cos(jnp.radians(alpha))
    # cos(arcsin(x)) == sqrt(1 - x^2): one sqrt instead of two expanded transcendentals.
    cos_right = jnp.sqrt(1 - (n_left / n_right * sin_alpha) ** 2)  # incidence_angle_right(alpha, nl, nr)
    cos_right_swapped = jnp.sqrt(1 - (n_right / n_left * sin_alpha) ** 2)  # (alpha, nr, nl)

    def shift(n):  # phase_shift(tuning, frequency, n)
        return 2 * jnp.radians(tuning) * n * (1 + frequency / F0)

    absolute_reflectivity = (1 - loss) * reflectivity
    absolute_transmissivity = (1 - loss) * (1 - reflectivity)
    sqrt_r = jnp.sqrt(guard(m_sqrt_r, absolute_reflectivity, one))
    sqrt_t = jnp.sqrt(guard(m_surface, absolute_transmissivity, one))
    tuning_phase = jnp.radians(tuning) * frequency / F0
    optomechanical = X_SCALE * 2 * jnp.pi / LAMBDA

    left_term = shift(n_left) * cos_left
    right_term = shift(n_right) * cos_right
    half_pi = jnp.pi / 2

    # --- space / modulation family: p = (f, length, refractive index)
    sign = jnp.where(m_lower, -1.0, 1.0)
    in_space = m_space | m_space_mod
    f_s = guard(in_space, p[:, 0], one)
    length_s = guard(in_space, p[:, 1], zero)
    index_s = guard(in_space, p[:, 2], one)
    space_phase = 2 * jnp.pi * f_s * length_s * index_s / C_LIGHT  # space: phi = -exp(-i * sign * space_phase)
    f_mod = guard(m_space_mod, p[:, 0], one)  # carrier-side `space` rows have f = 0: never divide by it
    w_g = sign * 2 * jnp.pi * f_mod
    path = w_g * length_s * index_s / 2 / C_LIGHT
    m_g = -0.5 * (2 * jnp.pi * F0) / w_g * jnp.sin(path)

    # --- sources and misc
    amplitude_laser = jnp.sqrt(guard(m_laser, 2 * p[:, 0] / EPSILON_0C, one))
    fm = 0.5 / guard(m_fm, sign * p[:, 0], one)
    chi = 1 / (guard(m_chi, p[:, 1], one) * (2 * jnp.pi * guard(m_chi, p[:, 0], one)) ** 2)
    squeeze = guard(m_squeezer, p[:, 0], zero) / (20 * jnp.log10(jnp.e))
    squeeze_angle = 2 * jnp.radians(p[:, 1])
    exp_squeeze = jnp.exp(2 * squeeze)  # cosh / sinh from a single exp
    cosh_squeeze = 0.5 * (exp_squeeze + 1 / exp_squeeze)
    sinh_squeeze = 0.5 * (exp_squeeze - 1 / exp_squeeze)

    # amplitude / phase for output slots 0..2 (slot 3 only exists for the squeezer, see below)
    a0 = _select(
        fid,
        [
            ((dummy_function,), one),
            ((vacuum_quantum_noise, laser_amplitude_modulation), half),
            ((loss_quantum_noise,), p[:, 0] / 2),
            ((laser_frequency_modulation, laser_frequency_modulation_lower), fm),
            ((susceptibility,), chi),
            ((laser,), amplitude_laser),
            ((signal_function,), p[:, 0]),
            ((space, space_lower), -one),
            ((space_modulation, space_modulation_lower), m_g),
            ((surface,), -sqrt_r),
            ((corrected_optomechanical_phase_left,), sqrt_r * optomechanical * cos_left),
            ((corrected_optomechanical_phase_right,), -sqrt_r * optomechanical * cos_right),
            ((force_calculation_left,), -cos_left / (C_LIGHT * X_SCALE)),
            ((force_calculation_right,), cos_right / (C_LIGHT * X_SCALE)),
            ((squeezer,), 0.5 * cosh_squeeze),
        ],
        zero,
    )
    t0 = _select(
        fid,
        [
            ((laser, signal_function), jnp.radians(p[:, 1])),
            ((space, space_lower), -sign * space_phase),
            ((space_modulation, space_modulation_lower), -path + half_pi),
            ((surface,), left_term),
            ((corrected_optomechanical_phase_left,), tuning_phase + left_term + half_pi),
            ((corrected_optomechanical_phase_right,), half_pi - tuning_phase - shift(n_right) * cos_right_swapped),
        ],
        zero,
    )
    a1 = _select(fid, [((surface,), -sqrt_t), ((squeezer,), 0.5 * sinh_squeeze)], zero)
    t1 = _select(fid, [((surface,), half_pi + 0.5 * (left_term - right_term)), ((squeezer,), squeeze_angle)], zero)
    a2 = _select(fid, [((surface,), -sqrt_r), ((squeezer,), 0.5 * cosh_squeeze)], zero)
    t2 = _select(fid, [((surface,), -shift(n_right) * cos_right_swapped)], zero)

    amplitudes = jnp.stack([a0, a1, a2], axis=-1)
    phases = jnp.stack([t0, t1, t2], axis=-1)
    slots = amplitudes * (jnp.cos(phases) + 1j * jnp.sin(phases))
    slot3 = jnp.where(m_squeezer, jnp.conjugate(slots[:, 1]), 0)  # squeezer: conjugate of slot 1
    return jnp.concatenate([slots, slot3[:, None]], axis=-1)
