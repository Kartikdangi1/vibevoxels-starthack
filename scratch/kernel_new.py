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
