import argparse
import os

import chainer
from chainer.backends import cuda
from chainer import training
from chainer.training import extensions

from net import Generator
from net import Discriminator
from net import make_z
from updater import DCGANUpdater
from dataset import DCGANDataset
from visualize import out_generated_image

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=50)
    parser.add_argument('--epoch', '-e', type=int, default=1000)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--out', '-o', default='')
    parser.add_argument('--resume', '-r', default='')
    parser.add_argument('--n_hidden', '-n', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--snapshot_interval', type=int, default=100000)
    parser.add_argument('--display_interval', type=int, default=100)
    args = parser.parse_args()

    out_dir = 'result'
    if args.out != '':
        out_dir = '{}/{}'.format(out, args.out)
    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# n_hidden: {}'.format(args.n_hidden))
    print('# epoch: {}'.format(args.epoch))
    print('# out: {}'.format(out_dir))
    print('')

    gen = Generator(n_hidden=args.n_hidden)
    dis = Discriminator()

    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu).use()
        gen.to_gpu()
        dis.to_gpu()

    def make_optimizer(model, alpha=0.0002, beta1=0.5):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        optimizer.add_hook(
            chainer.optimizer.WeightDecay(0.0001), 'hook_dec')
        return optimizer
    gen_optimizer = make_optimizer(gen)
    dis_optimizer = make_optimizer(dis)

    train, _ = chainer.datasets.get_cifar10(withlabel=False)
    train = DCGANDataset(make_z=make_z(args.n_hidden), dataset=train)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)

    updater = DCGANUpdater(
        models=(gen, dis),
        iterator=train_iter,
        optimizer={'gen': gen_optimizer, 'dis': dis_optimizer},
        device=args.gpu
    )

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=out_dir)

    snapshot_interval = (args.snapshot_interval, 'iteration')
    generateimage_interval = (args.snapshot_interval//100, 'iteration')
    display_interval = (args.display_interval, 'iteration')
    
    trainer.extend(
        extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        gen, 'gen_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        dis, 'dis_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'gen/loss', 'dis/loss',
    ]), trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(
        out_generated_image(
            gen, dis,
            10, 10, args.seed, out_dir, make_z(args.n_hidden)),
        trigger=generateimage_interval)

    trainer.run()

if __name__ == '__main__':
    main()