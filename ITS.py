from numpy import random,abs, cumsum, array, argmax
from random import uniform

def getSample(pdf):
    pdf = array(pdf)
    cdf = cumsum(pdf/sum(pdf))
    n = argmax(cdf >= uniform(0,1))
    return n
