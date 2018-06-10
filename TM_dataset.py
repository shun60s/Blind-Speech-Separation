#coding:utf-8

"""
Description: load dataset of each spectrogram of music and (vocal)speech
Date:  2018.6.3

Reference: network.py and util.py
           by wuyiming
           of UNet-VocalSeparation-Chainer-master
           <https://github.com/Xiao-Ming/UNet-VocalSeparation-Chainer>
"""

import chainer
import numpy as np
from librosa.util import find_files

# check version
# python 3.6.4 (64bit) win32
# windows 10 (64bit)
# chainer (3.2.0)
# librosa (0.6.0)
# numpy (1.14.0)

def LoadDataset( data_path, target="vocal"):
    filelist_fft = find_files( data_path, ext="npz", case_sensitive=True)  #[:200] Why was limit to 200 ?
    Xlist = []
    Ylist = []
    for file_fft in filelist_fft:
        dat = np.load(file_fft)
        Xlist.append(dat["mix"])
        if target == "vocal":
            assert(dat["mix"].shape == dat["vocal"].shape)
            Ylist.append(dat["vocal"])
        else:
            assert(dat["mix"].shape == dat["inst"].shape)
            Ylist.append(dat["inst"])
    return Xlist, Ylist


class TM_DatsSet(chainer.dataset.DatasetMixin):
    def __init__(self, batchsize, patchlength, path_stft, subepoch_mag=1):  # original: subepoch_mag=4
        self.batchsize=batchsize
        self.patchlength=patchlength
        self.subepoch_mag=subepoch_mag
        
        # load train data
        self.Xlist, self.Ylist= LoadDataset( path_stft)
        self.len= len(self.Xlist)
        self.itemlength=[x.shape[1] for x in self.Xlist]
        self.subepoch = (sum(self.itemlength) // self.patchlength // self.batchsize)  * self.subepoch_mag
        print ('# subepoch ', self.subepoch)
        print ('# number of patterns', self.__len__())
        
    def __len__(self):
        # batchsize を　subepoch分　回す。
        return self.batchsize * self.subepoch
    
    def get_example(self, i):
        X = np.zeros((1, 512, self.patchlength), dtype="float32")
        Y = np.zeros((1, 512, self.patchlength), dtype="float32")
        # インデックスをデータセットの総数内に変換する。
        i0= i % self.len
        # ランダムに　patch length 長の連続フレームをとりだす。
        randidx = np.random.randint( self.itemlength[i0]-self.patchlength-1)
        X[0, :, :] = self.Xlist[i0][1:, randidx:randidx + self.patchlength]
        Y[0, :, :] = self.Ylist[i0][1:, randidx:randidx + self.patchlength]
        return X,Y

if __name__ == '__main__':
    tm0=TM_DatsSet(64,128)
    print( tm0.__len__())
    X,Y=tm0.get_example(0)
    print( X.shape, Y.shape)

