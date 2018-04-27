import numpy as np

import chainer
from chainer import functions as F
from chainer import links as L
from chainer import initializers
from chainer.backends import cuda

def add_noise(h, sigma=0.2):
    xp = cuda.get_array_module(h)
    if chainer.config.train:
        return h + sigma * xp.random.randn(*h.shape).astype(xp.float32)
    else:
        return h

def label2onehot(label, num_class):
    xp = cuda.get_array_module(label)
    return xp.eye(num_class)[label]

class Generator(chainer.Chain):
    
    def __init__(self, n_noise, n_class, bottom_width=4, ch=512, initialW=initializers.Normal(0.02)):
        super(Generator, self).__init__()
        self.n_noise = n_noise
        self.n_class = n_class
        self.ch = ch
        self.bottom_width = bottom_width
        
        with self.init_scope():
            self.l0 = L.Linear(n_noise+n_class, bottom_width*bottom_width*ch,
                               initialW=initialW)
            self.dc1 = L.Deconvolution2D(ch, ch//2, 4, 2, 1, initialW=initialW)
            self.dc2 = L.Deconvolution2D(ch//2, ch//4, 4, 2, 1, initialW=initialW)
            self.dc3 = L.Deconvolution2D(ch//4, ch//8, 4, 2, 1, initialW=initialW)
            self.dc4 = L.Deconvolution2D(ch//8, 3, 3, 1, 1, initialW=initialW)
            self.bn0 = L.BatchNormalization(bottom_width*bottom_width*ch)
            self.bn1 = L.BatchNormalization(ch//2)
            self.bn2 = L.BatchNormalization(ch//4)
            self.bn3 = L.BatchNormalization(ch//8)
            
    def __call__(self, z, label):
        xp = cuda.get_array_module(z)
        label = label2onehot(label, self.n_class).astype(xp.float32)
        h = F.concat([z, label], axis=1)
        h = F.relu(self.bn0(self.l0(h)))
        h = F.reshape(h, (len(h), self.ch, self.bottom_width, self.bottom_width))
        h = F.relu(self.bn1(self.dc1(h)))
        h = F.relu(self.bn2(self.dc2(h)))
        h = F.relu(self.bn3(self.dc3(h)))
        x = F.sigmoid(self.dc4(h))
        return x
    
    def make_noise(self, batchsize=None):
        size = (self.n_noise,)
        if batchsize is not None:
            size = (batchsize,) + size
        return np.random.normal(0, 1, size=size).astype(np.float32)

class Discriminator(chainer.Chain):
    
    def __init__(self,n_class, bottom_width=4, ch=512, 
                 initialW=initializers.Normal(0.02)):
        super(Discriminator, self).__init__()
        self.n_class = n_class
        
        with self.init_scope():
            self.c0_0 = L.Convolution2D(None, ch//8, 3, 1, 1, initialW=initialW)
            self.c0_1 = L.Convolution2D(ch//8, ch//4, 4, 2, 1, initialW=initialW)
            self.c1_0 = L.Convolution2D(ch//4, ch//4, 3, 1, 1, initialW=initialW)
            self.c1_1 = L.Convolution2D(ch//4, ch//2, 4, 2, 1, initialW=initialW)
            self.c2_0 = L.Convolution2D(ch//2, ch//2, 3, 1, 1, initialW=initialW)
            self.c2_1 = L.Convolution2D(ch//2, ch//1, 4, 2, 1, initialW=initialW)
            self.c3_0 = L.Convolution2D(ch//1, ch//1, 3, 1, 1, initialW=initialW)
            self.l4 = L.Linear(bottom_width*bottom_width*ch, 1, initialW=initialW)
            self.bn0_1 = L.BatchNormalization(ch//4, use_gamma=False)
            self.bn1_0 = L.BatchNormalization(ch//4, use_gamma=False)
            self.bn1_1 = L.BatchNormalization(ch//2, use_gamma=False)
            self.bn2_0 = L.BatchNormalization(ch//2, use_gamma=False)
            self.bn2_1 = L.BatchNormalization(ch//1, use_gamma=False)
            self.bn3_0 = L.BatchNormalization(ch//1, use_gamma=False)
    
    def __call__(self, x, label):
        batchsize = len(x)
        xp = cuda.get_array_module(x)
        label = label2onehot(label, self.n_class)[:,:,None,None]
        label = label * xp.ones((batchsize, self.n_class) + x.shape[2:])
        label = label.astype(xp.float32)
        h = F.concat([x, label], axis=1)
        
        h = add_noise(h)
        h = F.leaky_relu(add_noise(self.c0_0(h)))
        h = F.leaky_relu(add_noise(self.bn0_1(self.c0_1(h))))
        h = F.leaky_relu(add_noise(self.bn1_0(self.c1_0(h))))
        h = F.leaky_relu(add_noise(self.bn1_1(self.c1_1(h))))
        h = F.leaky_relu(add_noise(self.bn2_0(self.c2_0(h))))
        h = F.leaky_relu(add_noise(self.bn2_1(self.c2_1(h))))
        h = F.leaky_relu(add_noise(self.bn3_0(self.c3_0(h))))
        return self.l4(h)