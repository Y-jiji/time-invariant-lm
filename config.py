import jax

key = jax.random.key(15445)
def ini():
    global key
    key, subkey = jax.random.split(key)
    return subkey