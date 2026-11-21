exec(open('scratch/ablate3.py').read().split("def phases")[0])
n=751
rows=arrays['signal_matrix_indices'][0]; cols=arrays['signal_matrix_indices'][1]
pr=arrays['signal_carrier_placing_indices'][0]; pc=arrays['signal_carrier_placing_indices'][1]
vals=jnp.ones((101,rows.shape[0]),dtype=jnp.complex128)
def snippet(rows,cols,pr,pc,vals):
    entry_positions = rows * (n + 1) + cols
    order = jnp.argsort(entry_positions, stable=True)
    sorted_positions = entry_positions[order]
    wanted = pr * (n + 1) + pc
    found_at = jnp.clip(jnp.searchsorted(sorted_positions, wanted, side="right") - 1, 0, sorted_positions.shape[0] - 1)
    found = sorted_positions[found_at] == wanted
    cur = jnp.where(found[None, :], vals[:, order[found_at]], 0)
    lw = S._last_wins(pc*n+pr)
    return cur, lw
def t(name,fn,*a):
    t0=time.time(); l=jax.jit(fn).lower(*a); t1=time.time(); l.compile(); print(f"{name}: lower {t1-t0:.2f} compile {time.time()-t1:.2f}",flush=True)
t("sort snippet fwd",snippet,rows,cols,pr,pc,vals)
t("sort snippet vjp",lambda *a: jax.vjp(lambda v: snippet(rows,cols,pr,pc,v),a[-1]),rows,cols,pr,pc,vals)
