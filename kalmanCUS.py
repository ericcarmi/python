# Complex Kalman filter
# Complex coefficient means oscillations in a first order system
# Unscented transform is used for observations: H(x) = x*conj(x)

from pylab import *
from scipy.signal import freqz,butter,lfilter
from scipy.io.wavfile import read,write

# Create time variables and signals
fs = 44100
ts = 1.0/fs
L = 1
t = linspace(0,L-ts,fs*L)
noisevar = 1.0
noise = normal(0,noisevar,fs*L)
# Make noise complex? In the model at least, not measurement
freq = 440
w0=2*pi*freq
x = sin(w0*t)
y = x + noise

# Create and initialize Kalman parameters
f_est = freq   # Expected frequency
z = exp(1j*2*pi*f_est*ts)
F = z   # Coefficients match frequency of expected frequency
H = 1
H2 = 0 # For y(k-1). Has a filtering effect on measurement
Q = 0.001 +0.015j # Value of Q effects stability of the output
R = zeros((fs*L)) + .1+1j
R2 = zeros((fs*L)) + 0.1+1j
Ppre = zeros((fs*L)) + 1 + 1j
Ppost = zeros((fs*L)) + 1 + 1j
Py    = zeros((fs*L)) + 1 + 1j
Pxy   = zeros((fs*L)) + 1 + 1j
Kgain = zeros((fs*L)) + 1j
xpre  = zeros((fs*L))*1j
xpost  = zeros((fs*L))*1j

# Unscented variables
nsig = 4
xsig_1 = zeros((2*nsig))*1j
xsig = zeros((2*nsig))*1j
ysig = zeros((2*nsig))*1j
yhat  = zeros((fs*L))*1j

H = lambda x: x * conj(x)

for k in range(1,len(y)):
    R[k] = noisevar
    # Calculate sigma points with last post-measurement estimate
    for m in range(1,2*nsig+1):
        xsig_1[m-1] = xpost[k-1] + Ppost[k-1]*exp(1j*2*pi*m/(2*nsig))
        xsig[m-1] = F * xsig_1[m-1]

    xpre[k] = 1.0/(2*nsig) * sum(xsig)
    # Calculate a priori error covariance
    s = 0
    for m in range(2*nsig):
        s += (xsig[m] - xpre[k])**2
    Ppre[k] = 1.0/(2*nsig) * s + Q

    # Do it again for the pre-measurement estimate
    for m in range(1,nsig+1):
        xsig_1[m-1] = xpre[k] + Ppre[k]*exp(1j*2*pi*m/(2*nsig))
        ysig[m-1] = H(xsig_1[m-1])

    yhat[k] = 1.0/(2*nsig) * sum(ysig)

    # Estimate covariance of predicted measurement
    s = 0
    s2 =0
    for m in range(2*nsig):
        s += (ysig[m] - yhat[k])**2
        s2 += (xsig[m] - xpre[k])*(ysig[m] - yhat[k])
    Py[k] = 1.0/(2*nsig) * s + R[k]
    Pxy[k] = 1.0/(2*nsig) * s2

    # Update gain and post-measurement estimates
    Kgain[k] = Pxy[k] * (Py[k])**-1
    xpost[k] = xpre[k] + Kgain[k]*(y[k] - H(yhat[k]) )
    Ppost[k] = Ppre[k] - Kgain[k] * Py[k] * conj(Kgain[k])


Y = fft(y)
plt.plot(abs(Y))
show()
plt.plot(abs(fft(imag(xpost))))
show()
filtered = imag(xpost)
plt.plot(filtered)
show()
write('noisysine.wav',fs,asarray(y/max(abs(y))*2**14,'int16'))
write('Unscentedsine.wav',fs,asarray(filtered/max(abs(filtered))*2**14,'int16'))




# Complex unscented Kalman as a function
def CUSKalman(y,fs,f_est):
    # Create time variables and signals
    ts = 1.0/fs
    t = linspace(0,len(x)*ts-ts,fs*len(y))
    # Create Kalman parameters
    #f_est = freq
    z = exp(1j*2*pi*f_est*ts)
    F = z   # Coefficients match frequency of expected frequency
    H = 1
    H2 = 0
    Q = 0.001
    R = zeros((fs)) + .1+0j
    R2 = zeros((fs)) + 0.1+0j
    Rperiod = int(fs/freq) # Interval for updating R
    ybuff = zeros((100))*1j
    Ppre = zeros((fs))+1 + 1j
    Ppost = zeros((fs))+1 + 1j
    Py    = zeros((fs))+1 + 1j
    Pxy   = zeros((fs))+ 1 + 1j
    Kgain = zeros((fs)) + 1j
    xpre  = zeros((fs))*1j
    xpost  = zeros((fs))*1j
    yhat  = zeros((fs))*1j

    nsig = 4
    xsig_1 = zeros((2*nsig))*1j
    xsig = zeros((2*nsig))*1j
    ysig = zeros((2*nsig))*1j

    H = lambda x: x * conj(x)
    r=1
    for k in range(1,len(y)):
        R[k] = r
        # Calculate sigma points with last post-estimate
        for m in range(1,2*nsig+1):
            xsig_1[m-1] = xpost[k-1] + Ppost[k-1]*exp(1j*2*pi*m/(2*nsig))
            xsig[m-1] = F * xsig_1[m-1]

        xpre[k] = 1.0/(2*nsig) * sum(xsig)
        # Calculate a prior error covariance
        s = 0
        for m in range(2*nsig):
            s += (xsig[m] - xpre[k])**2
        Ppre[k] = 1.0/(2*nsig) * s + Q

        # Do it again for the pre-measurement estimate
        for m in range(1,nsig+1):
            xsig_1[m-1] = xpre[k] + Ppre[k]*exp(1j*2*pi*m/(2*nsig))
            ysig[m-1] = H(xsig_1[m-1])

        yhat[k] = 1.0/(2*nsig) * sum(ysig)

        # Estimate covariance of predicted measurement
        s = 0
        s2 =0
        for m in range(2*nsig):
            s += (ysig[m] - yhat[k])**2
            s2 += (xsig[m] - xpre[k])*(ysig[m] - yhat[k])
        Py[k] = 1.0/(2*nsig) * s + R[k]
        Pxy[k] = 1.0/(2*nsig) * s2


        Kgain[k] = Pxy[k] * (Py[k])**-1
        xpost[k] = xpre[k] + Kgain[k]*(y[k] - H(yhat[k]) )
        Ppost[k] = Ppre[k] - Kgain[k] * Py[k] * conj(Kgain[k])


    return imag(xpost)
