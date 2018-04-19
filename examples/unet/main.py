import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

from net import UNet, BottomBlock, DownBlock, UpBlock
from dataset import get_facade, FacadeDataset
from dataset import transfrom_images, transform_labels

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

    bottom_ch = 512
    unet = UNet([
        DownBlock(None, bottom_ch // 8),
        DownBlock(None, bottom_ch // 4),
        DownBlock(None, bottom_ch // 2)
    ],
    BottomBlock(None, bottom_ch),
    [
        UpBlock(None, bottom_ch // 2),
        UpBlock(None, bottom_ch // 4),
        UpBlock(None, bottom_ch // 8)
    ],
    L.Convolution2D(None, 12, 3, 1, 1))

    model = L.Classifier(unet)

    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    print('Loading Data...')
    images, labels = get_facade()
    print('Transforming Images...')
    images = transfrom_images(images)
    print('Transforming Labels...')
    labels = transform_labels(labels)

    train, test = (labels[:300], images[:300]), (labels[300:], images[300:])
    train, test = FacadeDataset(train[1], train[0]), FacadeDataset(test[1], test[0])
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                repeat=False, shuffle=False)

    snapshot_interval = (args.snapshot_interval, 'iteration')
    generateimage_interval = (args.snapshot_interval//100, 'iteration')
    display_interval = (args.display_interval, 'iteration')

    print('Setting trainer...')
    updater = training.updater.StandardUpdater(train_iter, optimizer,
                                              device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=out_dir)
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'main/loss', 'main/accuracy'
    ]), trigger=display_interval)
    trainer.extend(extensions.LogReport())
    if extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport(
            ['main/loss', 'validation/main/loss'],
        'epoch', file_name='loss.png'))
        trainer.extend(extensions.PlotReport(
            ['main/accuracy', 'validation/main/accuracy'],
        'epoch', file_name='accuracy.png'))
    trainer.extend(extensions.ProgressBar(update_interval=20))

    print('RUN')
    trainer.run()

if __name__ == '__main__':
    main()