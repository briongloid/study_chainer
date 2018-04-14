import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

class NN(chainer.Chain):
    
    def __init__(self, n_units, n_out):
        super(NN, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_units)
            self.l2 = L.Linear(None, n_units)
            self.l3 = L.Linear(None, n_out)
    
    def __call__(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        return self.l3(h)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=100)
    parser.add_argument('--epoch', '-e', type=int, default=20)
    parser.add_argument('--frequency', '-f', type=int, default=-1)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--out', '-o', default='result')
    parser.add_argument('--unit', '-u', type=int, default=1000)
    args = parser.parse_args()
    
    model = L.Classifier(NN(args.unit, 10))
    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
    
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    
    train, test = chainer.datasets.get_mnist()
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                repeat=False, shuffle=False)
    
    updater = training.updater.StandardUpdater(train_iter, optimizer,
                                              device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    
    trainer.extend(extensions.dump_graph('main/loss'))
    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    
    trainer.extend(extensions.LogReport())
    
    if extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport(
            ['main/loss', 'validation/main/loss'],
        'epoch', file_name='loss.png'))
        trainer.extend(extensions.PlotReport(
            ['main/accuracy', 'validation/main/accuracy'],
        'epoch', file_name='accuracy.png'))
    
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
        'main/accuracy', 'validation/main/accuracy', 'elapsed_time']
    ))
    
    trainer.extend(extensions.ProgressBar(update_interval=10))
    
    trainer.run()
    
if __name__ == '__main__':
    main()