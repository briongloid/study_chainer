import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Chain, ChainList

class BottomBlock(Chain):
    def __init__(self, in_channels, out_channels):
        super(BottomBlock, self).__init__()
        w = chainer.initializers.Normal(0.02)
        with self.init_scope():
            self.c0 = L.Convolution2D(in_channels, out_channels, 3, 1, 1, initialW=w)
            self.c1 = L.Convolution2D(out_channels, out_channels, 3, 1, 1, initialW=w)
    
    def __call__(self, x):
        h = F.relu(self.c0(x))
        h = F.relu(self.c1(h))
        return h
    
class DownBlock(BottomBlock):
    
    def __call__(self, x):
        encoded = super(DownBlock, self).__call__(x)
        h = F.max_pooling_2d(encoded, ksize=2)
        return (h, encoded)

class UpBlock(BottomBlock):
    
    def __call__(self, x, encoded):
        h = F.unpooling_2d(x, ksize=2, outsize=encoded.shape[2:])
        h = F.concat([h, encoded], axis=1)
        h = super(UpBlock, self).__call__(h)
        return h
    
class UNet(Chain):
    
    def __init__(self, down_list, bottle_link, up_list, out_link=None):
        super(UNet, self).__init__()
        assert len(down_list) == len(up_list)
        
        with self.init_scope():
            self.down_list = chainer.ChainList(*down_list)
            self.bottle_link = bottle_link
            self.up_list = chainer.ChainList(*up_list)
            if out_link is not None:
                self.out_link = out_link
            else:
                self.out_link = None
        
    
    def __call__(self, x):
        h = x
        _encoded = []
        
        for i in range(len(self.down_list)):
            h = self.down_list[i](h)
            if isinstance(h, tuple):
                _encoded.append(h[1])
                h = h[0]
            else:
                _encoded.append(h)
        
        h = self.bottle_link(h)
        
        for i in range(len(self.up_list))[::-1]:
            h = self.up_list[i](h, _encoded.pop())
        
        if self.out_link is not None:
            h = self.out_link(h)
        
        return h
        