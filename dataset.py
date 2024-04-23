import pandas as pd
import jax, jax.numpy as jnp
from typing import *
import tqdm
from multiprocessing import *

class DataMan:
    """
    data manager
    """
    def __init__():
        pass
    def sample(self, task: str, mode: str, short: bool) -> Tuple[jax.Array, jax.Array, jax.Array]:
        pass

class WikiText:
    def __init__(self, device):
        with open('_data/wiki.txt', 'rb') as f:
            text = f.read()
        def load(x: bytes):
            x = jnp.frombuffer(x, dtype=jnp.uint8, count=len(x), offset=0)
            return jax.device_put(x, device)
        self.tx_tr = load(text[:3*len(text)//4])
        self.tx_te = load(text[3*len(text)//4:])
        self.progress = 0
        self.device = device
        self.extras = lambda: tuple()

    def sample(self, task: str, mode: str, batch: int, short: bool) -> Tuple[jax.Array, jax.Array, jax.Array]:
        tx = self.tx_tr if mode == 'tr' else\
             self.tx_te if mode == 'te' else\
             self.tx_te if mode == 'va' else\
             None
        if tx is None: raise "unknown mode"
        sentences = tx[self.progress:self.progress + 1024*batch]
        self.progress += batch
        xcat = sentences[:-1]
        ycat = sentences[1:]
        xsep = jnp.zeros_like(xcat)
        if short:
            xsep = xsep.at[jnp.arange(0, xcat.shape[0], 128)].set(1)
        xsep = xsep.cumsum()
        if task == "forward":
            return xcat, xsep, ycat, *self.extras()
        else:
            return ycat, xsep, xcat, *self.extras()

class TinyShake:
    def __init__(self, device):
        with open('_data/tinyshake.txt', 'rb') as f:
            text = f.read()
        def load(x: bytes):
            x = jnp.frombuffer(x, dtype=jnp.uint8, count=len(x), offset=0)
            return jax.device_put(x, device)
        self.tx_tr = load(text[:3*len(text)//4])
        self.tx_te = load(text[3*len(text)//4:])
        self.progress = 0
        self.device = device
        self.extras = lambda: tuple()

    def sample(self, task: str, mode: str, batch: int, short: bool) -> Tuple[jax.Array, jax.Array, jax.Array]:
        tx = self.tx_tr if mode == 'tr' else\
             self.tx_te if mode == 'te' else\
             self.tx_te if mode == 'va' else\
             None
        if tx is None: raise "unknown mode"
        sentences = tx[self.progress:self.progress + 1024*batch]
        self.progress += batch
        xcat = sentences[:-1]
        ycat = sentences[1:]
        xsep = jnp.zeros_like(xcat)
        if short:
            xsep = xsep.at[jnp.arange(0, xcat.shape[0], 128)].set(1)
        xsep = xsep.cumsum()
        if task == "forward":
            return xcat, xsep, ycat, *self.extras()
        else:
            return ycat, xsep, xcat, *self.extras()

if __name__ == '__main__':
    dataset = WikiText(jax.devices('cuda')[0])
    for i in tqdm.tqdm(range(100)):
        dataset.sample('forward', 'tr', 8, short=False)