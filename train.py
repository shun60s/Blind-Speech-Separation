#coding:utf-8

# Descripton: a trainer of U-Net, singing voice separation model
#             to separate speech from mixed signal of music and speech
# Date: 2018.6.3
#
# UNet is based on 
#      network.py
#      Created on Wed Nov  1 00:27:08 2017
#      @author: wuyiming
#      which is an implement of "Singing Voice Separation with Deep U-Net Convolutional Networks" by A.Jansson et al
#
#      util.py
#      Created on Wed Nov  1 11:47:06 2017
#      @author: wuyiming
# 
# 'unet.model' is used as initial value of the U-Net model.
# Pls download it from following.
#
# UNet-VocalSeparation-Chainer
# <https://github.com/Xiao-Ming/UNet-VocalSeparation-Chainer>
#--------------------------------------
#
# Chainer version 3.2 (use version 3.x)
#
# This is based on <https://raw.githubusercontent.com/chainer/chainer/v3/examples/mnist/train_mnist.py>
#
#---------------------------------------
# Check version
#  Python 3.6.4 on win32 (Windows 10)
#  Chainer 3.2.0
#  numpy 1.14.0 

from __future__ import print_function

import argparse
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import  training, cuda
from chainer.training import extensions
from chainer import reporter
from TM_dataset import *


class UNet(chainer.Chain):
    def __init__(self):
        super(UNet, self).__init__()
        with self.init_scope():
            # L.convolution2D (in channels=1, Out channels=16, 
            #  SizeOfFilter=4, Stride=2, Pad=1)
            self.conv1 = L.Convolution2D(1, 16, 4, 2, 1)
            self.norm1 = L.BatchNormalization(16)
            self.conv2 = L.Convolution2D(16, 32, 4, 2, 1)
            self.norm2 = L.BatchNormalization(32)
            self.conv3 = L.Convolution2D(32, 64, 4, 2, 1)
            self.norm3 = L.BatchNormalization(64)
            self.conv4 = L.Convolution2D(64, 128, 4, 2, 1)
            self.norm4 = L.BatchNormalization(128)
            self.conv5 = L.Convolution2D(128, 256, 4, 2, 1)
            self.norm5 = L.BatchNormalization(256)
            self.conv6 = L.Convolution2D(256, 512, 4, 2, 1)
            self.norm6 = L.BatchNormalization(512)
            self.deconv1 = L.Deconvolution2D(512, 256, 4, 2, 1)
            self.denorm1 = L.BatchNormalization(256)
            self.deconv2 = L.Deconvolution2D(512, 128, 4, 2, 1)
            self.denorm2 = L.BatchNormalization(128)
            self.deconv3 = L.Deconvolution2D(256, 64, 4, 2, 1)
            self.denorm3 = L.BatchNormalization(64)
            self.deconv4 = L.Deconvolution2D(128, 32, 4, 2, 1)
            self.denorm4 = L.BatchNormalization(32)
            self.deconv5 = L.Deconvolution2D(64, 16, 4, 2, 1)
            self.denorm5 = L.BatchNormalization(16)
            self.deconv6 = L.Deconvolution2D(32, 1, 4, 2, 1)

    def __call__(self, X):
        #  print ('X.shape', X.shape)
        h1 = F.leaky_relu(self.norm1(self.conv1(X)))
        h2 = F.leaky_relu(self.norm2(self.conv2(h1)))
        h3 = F.leaky_relu(self.norm3(self.conv3(h2)))
        h4 = F.leaky_relu(self.norm4(self.conv4(h3)))
        h5 = F.leaky_relu(self.norm5(self.conv5(h4)))
        h6 = F.leaky_relu(self.norm6(self.conv6(h5)))
        # Only 3 are  Dropout 
        dh = F.relu(F.dropout(self.denorm1(self.deconv1(h6))))
        dh = F.relu(F.dropout(self.denorm2(self.deconv2(F.concat((dh, h5))))))
        dh = F.relu(F.dropout(self.denorm3(self.deconv3(F.concat((dh, h4))))))
        dh = F.relu(self.denorm4(self.deconv4(F.concat((dh, h3)))))
        dh = F.relu(self.denorm5(self.deconv5(F.concat((dh, h2)))))
        # last active function is sigmod
        dh = F.sigmoid(self.deconv6(F.concat((dh, h1))))
        return dh

class UNetTrainmodel(chainer.Chain):
    def __init__(self, unet):
        super(UNetTrainmodel, self).__init__()
        with self.init_scope():
            self.unet = unet
            self.loss = None

    def __call__(self, X, Y):
        self.loss= None
        O = self.unet(X)
        self.loss = F.mean_absolute_error(X*O, Y)  # O is mask
        reporter.report({'loss': self.loss}, self)
        return self.loss

def main():
    parser = argparse.ArgumentParser(description='U-Net Speech(Vocal) Separation')
    parser.add_argument('--stft', '-t', default='stft',
                        help='Prefix Directory Name to input as dataset')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of track in each mini-batch')
    parser.add_argument('--patchlength', '-l', type=int, default=128,
                        help='length of input frames in one track')
    parser.add_argument('--epoch', '-e', type=int, default=30,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=1,
                        help='Frequency of taking a snapshot')
                        # default=-1 only last, default=1 every epoch, write out snapshot
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Prefix Directory Name to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# patch-length: {}'.format(args.patchlength))
    print('# epoch: {}'.format(args.epoch))
    print('# result directory: ', args.out)

    # Load dataset
    train = TM_DatsSet(args.batchsize, args.patchlength, args.stft)

    # Set up a neural network to train
    # Classifier reports mean_absolute/squared_error loss and accuracy at everypha=
    # iteration, which will be used by the PrintReport extension below.
    
    unet = UNet()
    model = UNetTrainmodel(unet)
    model.compute_accuracy= False  # no need compute accuracy
    
    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam(alpha=0.001) #alpha=0.0001) 
    optimizer.setup(model)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot and a model for each specified epoch
    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))
    trainer.extend(extensions.snapshot_object(unet,'model_{.updater.iteration}'), trigger=(frequency, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Save  plot image to the result dir
    if extensions.PlotReport.available():
        trainer.extend( extensions.PlotReport(['main/loss'], 'epoch', file_name='loss.png'))

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar(update_interval=1))  # default: update_interval=100

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)
    else:
        # Use model data made by wuyiming san
        chainer.serializers.load_npz("unet.model",unet)
        print ('# load unet.model')

    # Run the training
    trainer.run()
    

if __name__ == '__main__':
    main()


