#coding:utf-8

"""
Description: separate speech from mixed signal of music and speech
Date:  2018.6.3

Reference:  const.py, DoExperiment.py and util.py
            by wuyiming
            in UNet-VocalSeparation-Chainer
            <https://github.com/Xiao-Ming/UNet-VocalSeparation-Chainer>
"""
import argparse
import os
import numpy as np
from librosa.util import find_files
from librosa.core import load, stft, istft, resample, to_mono
from librosa.output import write_wav
from scipy.io.wavfile import read
import train
import chainer
from chainer import config

# check version
# python 3.6.4 (64bit) win32
# windows 10 (64bit)
# Chainer 3.2.0
# librosa (0.6.0)
# scipy (1.0.0)
# numpy (1.14.0)


def separate(PATH_INPUT, PATH_OUTPUT, MODEL, SR=16000, FFT_SIZE = 1024, H = 512):
    
    if os.path.isdir( PATH_INPUT):
        # 入力がディレクトリーの場合、ファイルリストをつくる
        filelist_mixdown = find_files(PATH_INPUT, ext="wav", case_sensitive=True)
    else:
    	# 入力が単一ファイルの場合
        filelist_mixdown=[PATH_INPUT]
    print ('number of mixdown file', len(filelist_mixdown))
    
    # 出力用のディレクトリーがない場合は　作成する。
    _, path_output_ext = os.path.splitext(PATH_OUTPUT)
    print ('path_output_ext',path_output_ext)
    if len(path_output_ext)==0  and  not os.path.exists(PATH_OUTPUT):
        os.mkdir(PATH_OUTPUT)
    
    # モデルの読み込み
    unet = train.UNet()
    chainer.serializers.load_npz( MODEL,unet)
    config.train = False
    config.enable_backprop = False
    
    # ミックスされたものを読み込み、vocal(speech)の分離を試みる
    for fmixdown in filelist_mixdown:
        # audioread でエラーが発生した場合は、scipyを使う。
        try:
            y_mixdown, _ = load(fmixdown,  sr=SR, mono=True)
        except:
            sr_mixdown, y_mixdown = read(fmixdown)
            if not sr_mixdown == SR:
                y_mixdown = resample(y_mixdown, sr_mixdown, SR)
        
        # 入力の短時間スペクトラムを計算して、正規化する。
        spec = stft(y_mixdown, n_fft=FFT_SIZE, hop_length=H, win_length=FFT_SIZE)
        mag = np.abs(spec)
        mag /= np.max(mag)
        phase = np.exp(1.j*np.angle(spec))
        print ('mag.shape', mag.shape)  
        start = 0
        end = 128 * (mag.shape[1] // 128)  # 入力のフレーム数以下で、networkの定義に依存して　適切な値を選ぶこと。
        # speech(vocal)を分離するためのマスクを求める
        mask = unet(mag[:, start:end][np.newaxis, np.newaxis, 1:, :]).data[0, 0, :, :]
        mask = np.vstack((np.zeros(mask.shape[1], dtype="float32"), mask))
        # 入力の短時間スペクトラムにマスクを掛けて、逆FFTで波形を合成する。
        mag2=mag[:, start:end]*mask 
        phase2=phase[:, start:end]
        y = istft(mag2*phase2, hop_length=H, win_length=FFT_SIZE)
        
        # 分離した speech(vocal)を出力ファイルとして保存する。
        if len(path_output_ext)==0:
            # ディレクトリーへ出力
            foutname, _ = os.path.splitext( os.path.basename(fmixdown) )
            fname= os.path.join(PATH_OUTPUT, (foutname + '.wav'))
        else:
            # 指定されたファイルへ出力
            fname= PATH_OUTPUT
        print ('saving... ', fname)
        write_wav(fname, y, SR, norm=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Speech(Vocal) Separation by U-Net')
    parser.add_argument('--input', '-i', default='mixdown',
                        help='Prefix Directory Name Or the file name (ex: xxx.wav) to input as mixed signal')
    parser.add_argument('--out', '-o', default='separate',
                        help='Prefix Directory Name Or the file name (ex: xxx.wav) to output as separated signal')
    parser.add_argument('--model', '-m', default='result/model_420',
                        help='Specify model (ex: result/model_iterationNumber Or unet.model)')
    args = parser.parse_args()
    
    separate( args.input, args.out, args.model)

