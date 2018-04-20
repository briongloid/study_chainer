# https://github.com/chainer/chainer/blob/master/examples/dcgan/visualize.py

import os
import warnings

import numpy as np

import chainer
from chainer.dataset import concat_examples
from chainer.training import extension
from chainer.training import trigger as trigger_module

try:
    from PIL import Image
    _available = True
except (ImportError, TypeError):
    _available = False

def _check_available():
    if not _available:
        warnings.warn(
            '$ pip install pillow'
        )
    
class GenerateImage(extension.Extension):
    
    def __init__(self, generate_func, data_func,
                 generator=None,
                 trigger=(1, 'epoch'),
                 file_name='previwe/{.updater.iteration:0>8}.png',
                 rows=5, cols=5, seed=0
                ):
        self._generate_func = generate_func
        self._data_func = data_func
        self._generator = generator
        self._trigger = trigger_module.get_trigger(trigger)
        self._file_name = file_name
        self._rows = rows
        self._cols = cols
        self._seed = seed
    
    def __call__(self, trainer):
        if _available:
            from PIL import Image
        else:
            return
        
        if not self._trigger(trainer):
            return
        
        np.random.seed(self._seed)
        rows = self._rows
        cols = self._cols
        n_images = rows * cols
        
        if hasattr(self._generate_func, 'xp'):
            xp = self._generate_func.xp
        else:
            xp = self._generator.xp
        
        x = xp.asarray([self._data_func(i) for i in range(n_images)])
        x = concat_examples(x)
        with chainer.using_config('train', False):
            t = self._generate_func(x)
        t = chainer.cuda.to_cpu(t.data)
        np.random.seed()
        
        t = np.asarray(np.clip(t * 255, 0.0, 255.0), dtype=np.uint8)
        _, _, H, W = t.shape
        t = t.reshape((rows, cols, 3, H, W))
        t = t.transpose(0, 3, 1, 4, 2)
        t = t.reshape((rows * H, cols * W, 3))
        
        file_name = self._file_name.format(trainer)
        file_path = os.path.abspath(file_name)
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        Image.fromarray(t).save(file_path)
        return
    
    