# Audio compressor
from pylab import *
from scipy.io.wavfile import read,write

fs,audiofile = read("/home/eric/Music/Filters.wav")
x=audiofile[:,0]/2**15.
ts=1.0/fs
L = 44100*10
thresh = -30.0 # Threshold in dB
width = 10
ratio = 5 # Compression ratio
attack = 0.001
release = 0.0001

currentRatio = 1
currentRatio_1 = 1
ratioError = 0
ratioError_1 = 0

r=[]
N = 2048
E = zeros(L)
xbuff = zeros((N))
rI = 0
y=0*x
dpwr = 0
xPWR_1 = 0
Q=[]
T=[]
dT=0
rs=[]
smoothed = 1
smoothed_1 = 1
for k in range(L):
    # Moving measurement of energy in signal
    E[k] = ( E[k-1] + ( abs(x[k]**2) - xbuff[-1] ) * 1.0/N )
    xbuff = roll(xbuff,1)
    xbuff[0] = abs(x[k]**2)
    xPWR = 20*log10(abs(E[k]))
    dpwr = xPWR - xPWR_1

    if(xPWR < (thresh - width/2)):
        ratioError = 1 - currentRatio # Target is ratio of 1
        dT=0
    elif(xPWR > (thresh + width/2)):
        ratioError = ratio - currentRatio # Target is compression ratio
        dT=0
    else:
        ratioError = 1/exp(-log(ratio)) - currentRatio


    # Attack and release are based on velocity
    if(dpwr > 0):
        currentRatio = currentRatio + attack*(ratioError + ratioError_1)
    else:
        currentRatio = currentRatio + release*(ratioError + ratioError_1)

    smoothed = 0.9*smoothed_1 + 0.1*currentRatio

    r.append(currentRatio)
    rs.append(smoothed)
    Q.append(xPWR)
    T.append(dT)
    y[k] = x[k] / currentRatio

    ratioError_1 = ratioError
    xPWR_1 = xPWR
    currentRatio_1 = currentRatio
    smoothed_1 = smoothed

plt.plot(r)
plt.plot(rs)
show()


# #Testing
# currentRatio = 1
# # Loop for control system/gain scheduling of ratio
# r = []
# rI = 0
# Ki = 0.01
# Kp = 0.9
# ratioError_1 = 0
# for k in range(200):
#     ratioError = ratio - currentRatio # Target is ratio of 1
#     rI = rI + ts*(ratioError + ratioError_1)
#     currentRatio = currentRatio + 0.1*(ratioError + ratioError_1)
#     r.append(currentRatio)
#     ratioError_1 = ratioError
#
#
# plt.plot(r)
# show()
#
# x=linspace(-5,5,1000)
# y=tanh(x)
# z=exp(-x**2)
# # Plot the compression curve piecewise
# # Index where function is equal to T is difference/step
# N = 10000
# x = linspace(-80,0,N)
# g=0*x
# step = 80.0/N
# T = -20
# W = 10
# R = 3
# beta = (1-1.0/R)/(1.0/R*(T+W/2)**2)
# # Need to transform back to dB
# #plt.plot(z,G)
# #show()
# x1 = T - W/2
# x2 = T + W/2
# q1 = int((80 + x1)/step)
# q2 = int((80 + x2)/step)
# y2 = 20*log10(1.0/R)
# #beta = (1.0+(x2-x1)**2)/R
# alpha = sqrt(R-1)
# z = linspace(0,alpha,q2-q1)
# G = 1/(1+z**2)
# b2 = y2 + x2/R
# y = []
# M=int(len(x)-q2)
# for k in range(M):
#     y.append(x2 + k*1.0/R*step)
# y=array(y)
# g[0:q1] = x[0:q1]
# h=cumsum(G)
# g[q1:q2] = (h-1)/h[-1] * (x2-x1) + x1
# g[q2:] = y
# plt.plot(g)
# show()
