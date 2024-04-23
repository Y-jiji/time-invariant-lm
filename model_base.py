import flax.linen as nn
import jax.numpy as jnp
import jax
from jax.tree_util import *
from typing import *
import math

# Decoder Layer with Flash Attention
class BaseLayer(nn.Module):
    D: int
    H: int
    
    def setup(self):
        self.qkv = nn.Sequential([
            nn.LayerNorm(reduction_axes=(-2), feature_axes=(-1), use_bias=False, use_scale=False),
            nn.Dense(3*self.D)
        ])
        self.ffn = nn.Sequential([
            nn.LayerNorm(reduction_axes=(-2), feature_axes=(-1), use_bias=False, use_scale=False),
            nn.Dense(self.H),
            nn.relu,
            nn.Dense(self.D)
        ])

    def __call__(self, xcat: jax.Array, xsep: jax.Array):
        """
        xcat: [sum{i} L(i), D]
        xsep: [sum{i} L(i)]
        """
        L = xcat.shape[0]
        qkv = self.qkv(xcat)
        q,k,v = qkv[..., 0:self.D], qkv[..., self.D:self.D*2], qkv[..., self.D*2:self.D*3]
        atten = nn.softmax(jnp.einsum("ij, kj -> ik", q, k), axis=-1)
        atten = atten * (xsep == xsep.reshape(L, 1)) * (jnp.arange(L) <= jnp.arange(L).reshape(L, 1))
        atten = atten / (atten.sum(axis=-1, keepdims=True) + 1e-10)
        return xcat + self.ffn(jnp.matmul(atten, xcat)), xsep

class BaseModel(nn.Module):
    D: int
    H: int
    VOCAB: int
    HEIGHT: int

    def setup(self):
        self.embed = nn.Embed(self.VOCAB, self.D)
        self.sandwich = nn.Sequential([
            BaseLayer(self.D, self.H)
            for i in range(self.HEIGHT)
        ])
        self.output = nn.Sequential([
            nn.Dense(self.VOCAB),
            nn.log_softmax
        ])
    
    def __call__(self, xcat: jax.Array, xsep: jax.Array):
        xcat = self.embed(xcat)
        xcat, xsep = self.sandwich(xcat, xsep)
        return self.output(xcat)

if __name__ == '__main__':
    # test basic functionalities
    import math
    import tqdm
    from util import *
    key = KeyMan()
    device = jax.devices("cuda")[0]
    D = 17
    H = 128
    VOCAB = 13
    HEIGHT = 6
    ALPHA = 0.8
    DECAY = 0.2
    HEIGHT = 6
    # generate a model frame
    model = BaseModel(D, H, VOCAB, HEIGHT)
    # initialize model
    state = model.init(key.gen(), 
        jax.random.randint(key.gen(), (64, ), 0, VOCAB), 
        jax.random.randint(key.gen(), (64, ), 0, 2).cumsum(axis=0), 
    )
    state = jax.device_put(state, device)
    apply = jax.jit(model.apply)
    # use the model once
    for i in tqdm.tqdm(range(100)):
        apply(state, 
            jax.random.randint(key.gen(), (1024*16, ), 0, VOCAB), 
            jax.random.randint(key.gen(), (1024*16, ), 0, 2).cumsum(axis=0),
        )
