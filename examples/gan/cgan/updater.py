import sys
sys.path.append('../../..')
from study_chainer.training.updaters import DCGANUpdater

class CGANUpdater(DCGANUpdater):
    
    def update_core(self):
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')
        gen, dis = self.gen, self.dis

        batch = next(self.get_iterator('main'))
        batchsize = len(batch)
        z, x_real, label = self.converter(batch, self.device)
        
        y_real = dis(x_real, label)
        
        x_fake = gen(z, label)
        y_fake = dis(x_fake, label)

        dis_optimizer.update(self.dis_loss_func, dis, y_fake, y_real)
        gen_optimizer.update(self.gen_loss_func, gen, y_fake)