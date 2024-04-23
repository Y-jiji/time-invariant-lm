from train_man import *
from model_base import *
from model_ti import *
from dataset import *
import jax
from util import *

if __name__ == '__main__':
    BATCH = 8
    H = 256
    D = 128
    VOCAB = 256
    HEIGHT = 3
    ALPHA = 0.9
    DECAY = 0.6
    EPSILON = 0.01
    keyman  = KeyMan()
    for extrapolation in [False]:
        extrapolation_text = ', extraploation' if extrapolation else ''
        # print(f'tiny shakespeare, base model{extrapolation_text}')
        # dataset = TinyShake(jax.devices('cuda')[0])
        # model   = BaseModel(D, H, VOCAB, HEIGHT)
        # train_man = TrainMan(model, dataset, "forward", BATCH, init=(
        #     keyman.gen(), 
        #     jax.random.randint(keyman.gen(), (64, ), 0, VOCAB), 
        #     jax.random.randint(keyman.gen(), (64, ), 0, 2).cumsum(axis=0), 
        # ))
        # train_man.train(5, extrapolation)
        print(f'tiny shakespeare, tilm{extrapolation_text}')
        dataset = TinyShake(jax.devices('cuda')[0])
        dataset.extras = lambda: (keyman.gen(), )
        model   = TiModel(D, H, VOCAB, ALPHA, DECAY, HEIGHT, EPSILON)
        train_man = TrainMan(model, dataset, "backward", BATCH, init=(
            keyman.gen(), 
            jax.random.randint(keyman.gen(), (1024*16, ), 0, VOCAB), 
            jax.random.randint(keyman.gen(), (1024*16, ), 0, 2).cumsum(axis=0),
            keyman.gen()
        ))
        train_man.train(5, extrapolation)
        print(f'wikitext, base model{extrapolation_text}')
        dataset = WikiText(jax.devices('cuda')[0])
        model   = BaseModel(D, H, VOCAB, HEIGHT)
        train_man = TrainMan(model, dataset, "forward", BATCH, init=(
            keyman.gen(), 
            jax.random.randint(keyman.gen(), (64, ), 0, VOCAB), 
            jax.random.randint(keyman.gen(), (64, ), 0, 2).cumsum(axis=0), 
        ))
        train_man.train(5, extrapolation)
        print(f'wikitext, tilm{extrapolation_text}')
        dataset = WikiText(jax.devices('cuda')[0])
        dataset.extras = lambda: (keyman.gen(), )
        model   = TiModel(D, H, VOCAB, ALPHA, DECAY, HEIGHT, EPSILON)
        train_man = TrainMan(model, dataset, "backward", BATCH, init=(
            keyman.gen(), 
            jax.random.randint(keyman.gen(), (1024*16, ), 0, VOCAB), 
            jax.random.randint(keyman.gen(), (1024*16, ), 0, 2).cumsum(axis=0),
            keyman.gen()
        ))
        train_man.train(5, extrapolation)