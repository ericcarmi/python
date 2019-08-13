#!/usr/bin/env python3
# Iterated Function System with Inverse Transform Sampling for arbitrary pdfs

from pylab import *
import ITS
from IFSstructures import *

A,T,p = getIFS("tree") # See IFSstructures for other fractals

zx=[]
zy=[]
x = linspace(0,1,301)
y = linspace(0,1,301)

for m in range(len(x)):
    for n in range(len(y)):
        z0 = array(( [x[m]], [y[m]] ))
        for k in range(10):
            r = ITS.getSample(p)
            z0 = A[r] @ z0 + T[r]
        zx.append(z0[0])
        zy.append(z0[1])

plt.scatter(zx,zy,s=0.1)
show()



def augment(A,T):
    rA, cA = A.shape
    rT, cT = T.shape
    if(rT != rA):
        print("Size mismatch")
        return
    Q = zeros((rA+1,cA+1))
    Q[0:rA, 0:cA] = A
    Q[0:rA, -1] = T.T
    Q[-1,-1] = 1
    return Q

def getScale(A):
    N = len(A)
    s=0
    for k in range(N):
        s += det(A[k])

    return sqrt(s/N)
