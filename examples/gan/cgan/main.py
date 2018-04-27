import argparse
import os

import chainer
from chainer import training
from chainer.training import extensions
from chainer.backends import cuda

from net import Generator, Discriminator
from updater import CGANUpdater

import sys
sys.path.append('../../..')
from study_chainer.training.extensions.generate_image import GenerateImage

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=50)
    parser.add_argument('--epoch', '-e', type=int, default=1000)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--out', '-o', default='')
    parser.add_argument('--resume', '-r', default='')
    parser.add_argument('--n_noise', '-n', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--snapshot_interval', type=int, default=1000)
    parser.add_argument('--display_interval', type=int, default=100)
    args = parser.parse_args()
    
    out_dir = 'result'
    if args.out != '':
        out_dir = '{}/{}'.format(out_dir, args.out)
    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# n_hidden: {}'.format(args.n_noise))
    print('# epoch: {}'.format(args.epoch))
    print('# out: {}'.format(out_dir))
    print('')
    
    gen = Generator(n_noise=args.n_noise, n_class=10)
    dis = Discriminator(n_class=10)

    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu)
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

    train, _ = chainer.datasets.get_cifar10(withlabel=True)
    transformer = lambda data: (gen.make_noise(),) + data
    train = chainer.datasets.TransformDataset(train, transformer)
    
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)

    updater = CGANUpdater(
        models=(gen, dis),
        iterator=train_iter,
        optimizer={'gen': gen_optimizer, 'dis': dis_optimizer},
        device=args.gpu
    )

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=out_dir)

    snapshot_interval = (args.snapshot_interval, 'iteration')
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
    trainer.extend(extensions.PlotReport(
        ['gen/loss', 'dis/loss'], x_key='iteration', trigger=display_interval))
    trainer.extend(extensions.ProgressBar(update_interval=10))
    
    gen_func = lambda data: gen(data[0], data[1])
    def data_func(gen):
        def _data_func(index):
            return (gen.make_noise(), index//10)
        return _data_func
    
    trainer.extend(
        GenerateImage(
            gen_func, data_func(gen),
            file_name='{}/{}'.format(out_dir, 'preview/{.updater.iteration:0>8}.png'),
            rows=10, cols=10, seed=800, device=args.gpu,
            trigger=snapshot_interval))

    trainer.run()
    
if __name__ == '__main__':
    main()