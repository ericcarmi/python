# DSP functions

import librosa
from scipy.io.wavfile import read,write
from scipy.signal import butter,freqz,lfilter
from scipy.signal import TransferFunction as tf
from pylab import *
import os

fs=44100
ts=1.0/fs
t=linspace(0,1-ts,fs)

def playSound(x,fs):
    outname = 'blank.wav'
    x = x - mean(x)
    write(outname, fs, asarray(x/max(abs(x)) * 2**14.0, 'int16'))
    os.system('play %s' %outname)
    os.system('rm   %s' %outname)


def playSequence(f0,fs,length,ratios):
    t = linspace(0,length-1.0/fs,fs*length)
    x = sin(2*pi*t*f0)
    L = 0
    for k in range(1,len(ratios)):
        x = append(x,sin(2*pi*t*f0*ratios[k] + arcsin(L)))
        L = sin(2*pi*t*f0*ratios[k])[-1]
    playSound(x,fs)
    return asarray(x)

# Chords
def playSequence2(f0,f1,fs,length,ratios):
    t = linspace(0,length-1.0/fs,fs*length)
    x = sin(2*pi*t*f0)
    y = sin(2*pi*t*f0)
    for k in range(1,len(ratios)):
        x = append(x,sin(2*pi*t*f0*ratios[k]))
        y = append(y,sin(2*pi*t*f1*ratios[k]))

    z = (x+y)/2.0
    playSound(z,fs)
    return

def expfreqsweep(w1,w2,T):
    fs = 44100
    ts = 1.0/fs
    t = linspace(0,T-ts, fs*T)
    return sin(w1*T/log(w2/w1)*(exp(t/T*log(w2/w1))-1))

def linfreqsweep(w1,w2,T):
    fs = 44100
    ts = 1.0/fs
    t = linspace(0,T-ts, fs*T)
    F = linspace(w1,w2,len(t))
    return sin(t*F/2)



def impulsegen(T,z1,z2):
    return [0]*z1 + [1]*T + [0]*z2



def unitstep(t,T):
    y = 0*t
    y[T:] = 1
    return y

def triangleWave(x,freq):
    y = 0*x
    for k in range(len(x)):
        A = x[k] % (F)
        if(A>(F/2.)):
            y[k] = 4*A/F - 3
        else:
            y[k] =  -4*A/F + 1
    return y


def squareWave(x,F):
    y = 0*x
    for k in range(len(x)):
        A = x[k] % (F)
        if(A>(F/2)):
            y[k] = 1
        else:
            y[k] = -1
    return y

def sawWave(x,F):
    return (x % (2*pi*F)) / pi - 1
