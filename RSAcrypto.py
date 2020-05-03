#!/usr/bin/python
'''
This is meant to be educational. This does not use BigInts, so it is not secure.
Create public and private key using RSA
'''
from math import gcd
import random
import math
from numpy import floor, mod, array
from numpy.random import randint
from tqdm import tqdm

def Euclid(n, p, q, b):
    n0 = n
    b0 = b
    t0 = 0
    t = 1
    q = floor(n0/b0)
    r = n0 - q*b0
    while( r > 0 ):
        temp = t0 - q*t
        if( temp >= 0):
            temp = mod(temp,n)
        elif( temp < 0 ):
            temp = n - mod(-temp,n)
        t0 = t
        t = temp
        n0 = b0
        b0 = r
        q = floor(n0/b0)
        r = n0 - q*b0

    if( b0 != 1):
        print("No inverse exists")
        return
    else:
	    return mod(t,n)

def is_prime(n):
    if n == 2:
        return True
    if n % 2 == 0 or n <= 1:
        return False

    sqr = int(math.sqrt(n)) + 1

    for divisor in range(3, sqr, 2):
        if n % divisor == 0:
            return False
    return True

# Input prime numbers p,q to generate a key pair
def RSAkeygen(p, q):
    if( (is_prime(p) & is_prime(q)) == False):
        print("Must provide prime numbers")
        return
    n = p*q
    phi = (p-1)*(q-1)
    b = randint(2,phi+1)
    while( gcd(b,phi) != 1):
        b = randint(2,phi+1)
        a = Euclid(phi,p,q,b)
    return long(a), long(b)

# Calculate (x ** y) % z efficiently
def modexp(x, y, z):
    number = 1
    while y:
        if y & 1:
            number = number * x % z
        y >>= 1
        x = x * x % z
    return number

# Generate primes by counting...not fastest, but works
def gen_primes(N):
    p=[]
    for k in range(2,N):
        if( is_prime(k) ):
            p.append(k)
    return array(p)


'''
# TEST

primes=gen_primes(10000)
s = 'hello world' # Message to encode
Ls = len(s)
M = [0]*Ls # Message, encoded as ascii
C = [0]*Ls # Cipher, encoded with public key
D = [0]*Ls # Decoded cipher, with private key

p = 997
q = 443
n = p*q
phi = (p-1)*(q-1)
b = 8343535 # Public key
a = int(Euclid(phi, p, q, b)) # Private key, calculated quickly with primes known

x = ''
for k in range(0,len(M)):
    M[k] = ord(s[k])

for k in range(0,len(M)):
    C[k] = modexp(M[k], b, n)
    D[k] = modexp(C[k], a, n)
    x += chr(D[k])

print("Original Message: %s" %s)
print("Encoded message: %s" % ', '.join(map(str,M)))
print("Encoded Cipher: %s" % ', '.join(map(str,C)))
print("Decrypted cipher: %s" % ', '.join(map(str,D)))
print("Decrpyted message: %s" %x)
'''
