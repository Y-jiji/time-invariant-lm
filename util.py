import jax

class KeyMan:
    def __init__(self):
        self.key = jax.random.key(64)
    def gen(self):
        self.key, subkey = jax.random.split(self.key)
        return subkey
