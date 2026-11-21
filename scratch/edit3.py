p = 'differometor/components.py'
s = open(p, encoding='utf-8').read()

# numpy twins, inserted before mirror_matrix
twins = '''# -----------------------------------------------------------------------------
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


'''
a = s.index("def mirror_matrix(")
s = s[:a] + twins + s[a:]

# use twins inside mirror_matrix / beamsplitter_matrix only
ma = s.index("def mirror_matrix(")
mb = s.index("def directional_beamsplitter_matrix(")
seg = s[ma:mb]
seg = seg.replace("= reflectivity_left(", "= _reflectivity_left_np(").replace("= transmissivity(", "= _transmissivity_np(").replace("= reflectivity_right(", "= _reflectivity_right_np(")
s = s[:ma] + seg + s[mb:]
open(p, 'w', encoding='utf-8').write(s)

p = 'differometor/build.py'
b = open(p, encoding='utf-8').read()
old = 'space_entry = space(jnp.array([F, data["properties"]["length"], data["properties"]["refractive_index"]]))[0]'
assert old in b
b = b.replace(old, 'space_entry = space_entry_np(F, data["properties"]["length"], data["properties"]["refractive_index"])')
b = b.replace("    space,\n    laser_amplitude_modulation,", "    space,\n    space_entry_np,\n    laser_amplitude_modulation,", 1)
open(p, 'w', encoding='utf-8').write(b)
print("ok")
