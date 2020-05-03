from pylab import *
from bpproject import *

# Data is organized in the following way:
# data[0] is one trial
# data[0][0] contains the PID gains used for the first trial
# data[0][1] contains the measurements (Tr, PO, SSE)
data = load("trainingset.npy")

bp = backprop()

Lm = [3, 4, 4, 3]
bp.setNNlayout(Lm)
bp.numTrials = len(data)
bp.setactivation('logsig')
bp.xx = data[:,0] # PID gains

bp.activate[1] =  lambda x: x
bp.diffactivate[1] = lambda x: 1
# Set target function and input range with notes and targets, not functions
# Input range is the FFT of the notes
bp.initSave(True)
bp.alpha = 0.9
#bp.momentumFlag = True
#bp.mu = 0.01

print bp.W[0]
bp.w_init=0.1
bp.b_init=0.1
bp.train()


bp.plotWeights()
#print bp.W[0][0]
E = bp.run()
print E[0],E[1]
print E[-1]
for k in range(len(E)):
    plt.plot(k,E[k][0],'r')
    plt.plot(k,E[k][1],'g')
    plt.plot(k,E[k][2],'b')

plt.show()
