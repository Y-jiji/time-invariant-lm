import flax.linen as nn
import jax.numpy as jnp
import jax
from typing import *
import math

class TiLayer(nn.Module):
    # constants
    D: int
    H: int
    VOCAB: int
    DECAY: float
    LOG_ALPHA: float

    def setup(self):
        # first token mix for time delta
        self.delta = self.param('delta', nn.initializers.normal(1e-1), (self.D, 1))
        # second token mix for modifying xdiv by probability
        self.slice = self.param('slice', nn.initializers.normal(1e-1), (self.D, 1))
        # elements for time mix
        self.a = self.param('a', nn.initializers.normal(1e-1), (self.D, 1))
        self.b = self.param('b', nn.initializers.normal(1e-1), (self.D, self.D))
        self.ffn = nn.Sequential([
            nn.LayerNorm(reduction_axes=(-2), feature_axes=(-1), use_bias=False, use_scale=False),
            nn.Dense(self.H),
            nn.relu,
            nn.Dense(self.D)
        ])
        self.ln2 = nn.LayerNorm(reduction_axes=(-2), feature_axes=(-1), use_bias=False, use_scale=False)

    def __call__(self, xcat: jax.Array, xsep: jax.Array, xdiv: jax.Array, esti: jax.Array, output: jax.Array, key: jax.random.PRNGKey)\
        -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.random.PRNGKey]\
    :
        """
        xcat: [sum{i} L(i), D]
        xsep: [sum{i} L(i)]
        xdiv: [sum{i} L(i)]
        esti: [2, sum{i} L(i), VOCAB], probability esitmated in middle layer that directly used as output, and its last layer if it exists
        """
        # if the block boundary should be preserved
        key, subkey = jax.random.split(key)
        xdiv = xdiv * (jax.random.uniform(subkey, xdiv.shape) > self.DECAY)
        t = nn.log_sigmoid(jnp.matmul(xcat, self.delta))
        # right-to-left aggregation operator
        @jax.jit
        def op(a, b, x, y):
            (x, xt, xsep, xdiv), (y, yt, ysep, ydiv) = x, y
            if x.shape[0] == 0: return (x, xt, xsep, xdiv)
            dt = yt - xt # time difference
            dt = dt.reshape(*dt.shape, 1)
            a_ = jnp.exp(dt * a) * b
            return (y + (xsep == ysep) * (xdiv == ydiv) * jnp.einsum("ij, ijk -> ik", x, a_), yt, ysep, ydiv)
        # compute a group-wise suffix-sum
        xnew, _, _, _ = jax.lax.associative_scan(
            jax.tree_util.Partial(op, self.a, self.b), # first fill in self.a as shared params
            (xcat, t, xsep.reshape(*xsep.shape, 1), xdiv.cumsum(0).reshape(*xdiv.shape, 1)),
            reverse=True, axis=0
        )
        xnew = self.ffn(xnew) + xcat
        # generate estimation from each group suffix sum
        xest = nn.log_softmax(jnp.matmul(self.ln2(xnew), output))
        gt   = esti[0].max(-1, keepdims=True) > self.LOG_ALPHA
        esti = jnp.stack([
            gt * esti[0] + (1-gt) * xest,
            gt * esti[1] + (1-gt) * esti[0]
        ], axis=0)
        # xnew = gt * xcat + (1-gt) * xnew
        # compute ffn + residual for output
        return xnew, xsep, xdiv, esti, output, key

class TiModel(nn.Module):
    # constants
    D: int                          # forwarding dimension
    H: int                          # hidden dimension
    VOCAB: int                      # dictionary size
    ALPHA: float                    # the confidence threshold
    DECAY: float                    # the probability to remove a boundary
    HEIGHT: int                     # depth of this model
    EPSILON: float                  # 

    def setup(self):
        self.embed  = nn.Embed(self.VOCAB, self.D)
        self.output = self.param('output', nn.initializers.normal(1e-1), (self.D, self.VOCAB))
        self.sandwich = nn.Sequential([
            TiLayer(self.D, self.H, self.VOCAB, self.DECAY, math.log(self.ALPHA))
            for i in range(self.HEIGHT)
        ])

    def __call__(self, xcat: jax.Array, xsep: jax.Array, key: jax.random.PRNGKey)\
        -> jax.Array\
    :
        """
        xcat: [sum{i} L(i), D]
        xsep: [sum{i} L(i)]
        """
        # concatenated embedding for each sentence
        xcat = self.embed(xcat)
        # esti: [2, sum{i} L(i), VOCAB], probability esitmated in middle layer that directly used as output, and its last layer if it exists
        esti = jnp.log(jnp.ones((2, xcat.shape[0], self.VOCAB)) / self.VOCAB)
        # the output output combined with higher layer output
        xcat, xsep, xdiv, esti, _, key = \
            self.sandwich(xcat, xsep, jnp.ones(xcat.shape[:-1]), esti, self.output, key)
        return esti[0] * (1-self.EPSILON) + esti[1] * self.EPSILON

if __name__ == '__main__':
    import math
    import tqdm
    from util import *
    key = KeyMan()
    device = jax.devices("cuda")[0]
    D = 17
    H = 128
    VOCAB = 13
    ALPHA = 0.8
    DECAY = 0.2
    HEIGHT = 6
    EPSILON = 0.01
    # generate a model frame
    model = TiModel(D, H, VOCAB, ALPHA, DECAY, HEIGHT, EPSILON)
    # initialize model
    state = model.init(key.gen(), 
        jax.random.randint(key.gen(), (64, ), 0, VOCAB), 
        jax.random.randint(key.gen(), (64, ), 0, 2).cumsum(axis=0), 
        key.gen(), train=False,
    )
    state = jax.device_put(state, device)
    apply = jax.jit(model.apply)
    # use the model once
    for i in tqdm.tqdm(range(1000)):
        apply(state, 
            jax.random.randint(key.gen(), (1024*16, ), 0, VOCAB), 
            jax.random.randint(key.gen(), (1024*16, ), 0, 2).cumsum(axis=0),
            key.gen(), train=True
        )
