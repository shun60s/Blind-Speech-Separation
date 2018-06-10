#coding:utf-8

"""
Description: make dataset of Spectrogram from music_speech for U-Net train
Date:  2018.6.3

Reference: const.py, ProcessDSD.py, ProcessIKALA.py, ProcessIMAS.py, and ProcessMedleyDB.py
           by wuyiming
           in UNet-VocalSeparation-Chainer
           <https://github.com/Xiao-Ming/UNet-VocalSeparation-Chainer>
------------------------------------------------------------------------

This will use following  MARSYAS(music analysis, retrieval and synthesis for audio singnal) Data sets

Music Speech

A similar dataset which was collected for the purposes of music/speech discrimination. 
The dataset consists of 120 tracks, each 30 seconds long. 
Each class (music/speech) has 60 examples. 
The tracks are all 22050Hz Mono 16-bit audio files in .wav format. 
<http://marsyasweb.appspot.com/download/data_sets/>

Download the GTZAN music/speech collection, music_speech.tar.gz and expand it into music_speech directory.

"""
import argparse
import os
import numpy as np
from librosa.util import find_files
from librosa.core import load, stft
from librosa.output import write_wav

# check version
# python 3.6.4 (64bit) win32
# windows 10 (64bit)
# librosa (0.6.0)
# numpy (1.14.0)

def mixdown( PATH_music_wav, PATH_speech_wav, PATH_stft, PATH_MIXDOWN,  mix_ratio,  loop_count,
             SR=16000, FFT_SIZE = 1024, H = 512):
    #  mix_ratioは　music と　speechを加算する割合、重み。
    #  SRは　sampling frequency [Hz]
    #  FFT_SIZEは　FFTの長さ
    #  H は　shift frame (hopping) of Short-time Fourier transform
    
    #　music と　speech のファイルのリストを作る
    filelist_music = find_files(PATH_music_wav, ext="wav", case_sensitive=True)
    filelist_speech = find_files(PATH_speech_wav, ext="wav", case_sensitive=True)
    print ('number of music file', len(filelist_music))
    print ('number of speech file', len(filelist_speech))
    # 出力用のディレクトリーがない場合は　作成する。
    if not os.path.exists(PATH_stft):
        os.mkdir(PATH_stft)
    if not os.path.exists(PATH_MIXDOWN):
        os.mkdir(PATH_MIXDOWN)
    
    # loop: 訓練データを増やすため　music と speech のミックスする組み合わせを　1個つづ更新する
    b=min(loop_count, len(filelist_speech) )
    print ('boost loop count', b)
    for loop in range (b):
        # music と　speech を読み込んで　ミックスする
        for (fmusic,fspeech) in zip (filelist_music, filelist_speech):
            y_music, _ = load(fmusic,  sr=SR, mono=True)
            y_speech, _= load(fspeech, sr=SR, mono=True)
            minsize = min( [y_music.size, y_speech.size] ) 
            # 短い時間の方に合わせて加算する（混ぜる）
            y_mixdown = y_music[:minsize]  + y_speech[:minsize] * mix_ratio
            # 短時間FFTスペクトルの計算
            Spec_mixdown  = np.abs(stft(y_mixdown, n_fft=FFT_SIZE, hop_length=H)).astype(np.float32) # window='hann'
            Spec_speech   = np.abs(stft(y_speech[:minsize],  n_fft=FFT_SIZE, hop_length=H)).astype(np.float32) # window='hann'
            Spec_music    = np.abs(stft(y_music[:minsize],   n_fft=FFT_SIZE, hop_length=H)).astype(np.float32) # window='hann'
            # スペクトルの最大値で正規化する
            norm = Spec_mixdown.max()
            Spec_mixdown /= norm
            Spec_speech /= norm
            Spec_music /= norm
            # スペクトルを npzファイルとして保存する
            path_fmusic, _ = os.path.splitext( os.path.basename(fmusic) )
            path_fspeech, _ = os.path.splitext( os.path.basename(fspeech) )
            foutname= path_fspeech + '_' + path_fmusic
            print ('saving... ', foutname)
            # UNet-VocalSeparation-Chainerと互換性を保つため、 Speech の npzの中の名前は vocalとする。
            np.savez(os.path.join(PATH_stft, foutname+".npz"), mix=Spec_mixdown, vocal=Spec_speech, inst=Spec_music)
            # ミックダウンしたwavを保存する
            fname= os.path.join(PATH_MIXDOWN, foutname + '.wav')
            write_wav(fname, y_mixdown , SR,  norm=True)  # 最大値が１になるように正規化する。
            
        # speech file list　の要素を1個分シフトする
        filelist_speech = [filelist_speech[-1]]+ filelist_speech[:-1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare dataset for U-Net train')
    parser.add_argument('--music', '-m', default='music_speech/music_wav',
                        help='Prefix Directory Name to input music')
    parser.add_argument('--speech', '-s', default='music_speech/speech_wav',
                        help='Prefix Directory Name to input speech')
    parser.add_argument('--stft', '-t', default='stft',
                        help='Prefix Directory Name to output stfr as dataset')
    parser.add_argument('--mixdown', '-o', default='mixdown',
                        help='Prefix Directory Name to output mixdown')
    parser.add_argument('--mixratio', '-r', type=float, default=1.0,
                        help='mix down ratio of speech to music')
    parser.add_argument('--loop', '-b', type=int, default=2,
                        help='boost loop count to mix')
    args = parser.parse_args()

    mixdown(args.music, args.speech, args.stft, args.mixdown,  args.mixratio, args.loop)

