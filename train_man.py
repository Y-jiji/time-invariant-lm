from flax.training.train_state import TrainState
import jax, jax.numpy as jnp
import optax
import flax.linen as nn
from typing import *
from dataset import *
import tqdm

class TrainMan:
    """
    manager of training process
    """
    def __init__(self, model: nn.Module, dataset: DataMan, task: str, batch: int, init: Tuple[Any]):
        self.model = model
        self.optim = optax.adam(1e-4)
        self.state = TrainState.create(
            apply_fn=model.apply,
            params=model.init(*init),
            tx=self.optim
        )
        self.dataset = dataset
        self.task  = task
        self.batch = batch

    def train(self, epoch: int, extrapolation: bool):
        def loss_fn(param, xcat, xsep, label, *extras):
            L = xcat.shape[0]
            ALIGN = 8192
            xcat = jnp.concat([xcat, jnp.zeros(ALIGN - L%ALIGN, int)])
            xsep = jnp.concat([xsep, jnp.ones(ALIGN - L%ALIGN, int) * 1000000000])
            out = self.state.apply_fn(param, xcat, xsep, *extras)
            return -out[jnp.arange(0, label.shape[0]), label].mean()
        grad_fn = jax.jit(jax.grad(loss_fn))
        loss_fn = jax.jit(loss_fn)
        for e in range(epoch):
            if e != 0:
                # training for 4096 samples
                for i in tqdm.tqdm(range(4096//self.batch), desc=f'train {e}'):
                    xcat, xsep, label, *extras = self.dataset.sample(self.task, 'tr', self.batch, extrapolation)
                    grads = grad_fn(self.state.params, xcat, xsep, label, *extras)
                    self.state = self.state.apply_gradients(grads=grads)
            losses = []
            # validating for 128 samples
            for i in tqdm.tqdm(range(128//self.batch), desc=f'validate {e}'):
                xcat, xsep, label, *extras = self.dataset.sample(self.task, 'va', self.batch, False)
                losses.append(loss_fn(self.state.params, xcat, xsep, label, *extras).mean().item())
            print(sum(losses) / len(losses))
