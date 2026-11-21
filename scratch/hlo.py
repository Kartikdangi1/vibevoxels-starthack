exec(open('scratch/ablate3.py').read().split("def phases")[0])
import collections, re
def cheap(fid,x):
    r=jnp.real(x).astype(jnp.float64); return (r[:, :4]*(1.0+0.1*fid[:,None])).astype(jnp.complex128)
S.evaluate_components=cheap
fwd=lambda p,arr: jax.vjp(lambda q: S.simulate(**dict(arr,optimized_parameters=q)),p)
txt=jax.jit(fwd).lower(p0,arrays).compile().as_text()
ops=collections.Counter(re.findall(r"= [\w\[\],\{\}\(\)\.:\s]*? (\w[\w-]*)\(",txt))
print(sum(ops.values()),"ops"); print(ops.most_common(25))
print("custom-call:", txt.count("custom-call"), "sort:", txt.count(" sort("), "scatter:", txt.count(" scatter("), "gather:", txt.count(" gather("), "while:", txt.count(" while("))
