import chainer
import chainer.functions as F
import chainer.backends.cuda as cuda

def dis_loss_func(dis, y_fake, y_real):
    batchsize = len(y_fake)
    L1 = F.sum(F.softplus(-y_real)) / batchsize
    L2 = F.sum(F.softplus(y_fake)) / batchsize
    loss = L1 + L2
    chainer.report({'loss': loss}, dis)
    return loss

def gen_loss_func(gen, y_fake):
    batchsize = len(y_fake)
    loss = F.sum(F.softplus(-y_fake)) / batchsize
    chainer.report({'loss': loss}, gen)
    return loss

class DCGANUpdater(chainer.training.StandardUpdater):
    
    def __init__(self, gen_loss_func=gen_loss_func, dis_loss_func=dis_loss_func, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        self.gen_loss_func = gen_loss_func
        self.dis_loss_func = dis_loss_func
        super(DCGANUpdater, self).__init__(*args, **kwargs)
    
    def update_core(self):
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')
        
        batch = next(self.get_iterator('main'))
        z, x_real = self.converter(batch, self.device)
        gen, dis = self.gen, self.dis
        
        y_real = dis(x_real)
        
        x_fake = gen(z)
        y_fake = dis(x_fake)
        
        dis_optimizer.update(self.dis_loss_func, dis, y_fake, y_real)
        gen_optimizer.update(self.gen_loss_func, gen, y_fake)
        