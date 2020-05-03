# Backpropagation class for python
'''
Outline of algorithm

Forward
    n = wx + b
    y = f(n)
Backward
    s_M = -2 G(n(M)) E
    s_m = -2 G(n(m)) * w * s_m+1


'''
from numpy import zeros, exp, log, sin, cos, ones, pi, matmul, dot, linspace
from numpy import array,eye,sign,tanh,load
from numpy.random import random,uniform
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

measured_data = load("trainingset.npy")


class backprop():
    def __init__(self):
        self.numNeurons = []          # Number of neurons in n-th layer
        self.numLayers = 0           # Number of layers
        self.w = []
        self.b = []
        self.w_1 = 0*self.w
        self.b_1 = 0*self.b
        self.dw = 0*self.w
        self.db = 0*self.b

        self.w_init = 1.0
        self.b_init = 1.0

        self.n = 0
        self.y = 0
        self.yhat = 0
        self.t = 0
        self.numTrials = 0
        self.sensitivities = 0
        self.Gmatrix = 0
        self.targetfunc = None

        self.alpha = 0.1
        self.mu = 0.0
        self.dalpha = 0
        self.dmu = 0
        self.momentumFlag = False
        self.saveFlag = False

        self.pLow = 0
        self.pHigh = 1
        self.xx = 0
        self.yy = 0
        self.zz = 0


    def perturbation(self,x):
        r = random.uniform(0,1)
        if r > 0.5:
            return x
        return -1*x

    def initSave(self, x):
        self.saveFlag = x
        self.W = []
        self.B = []
        for k in range(len(self.w)):
            r = self.w[k].shape[0]
            c = self.w[k].shape[1]
            self.W.append( zeros((self.numTrials, r, c)) )
            self.B.append( zeros((self.numTrials, r, c)) )

    def setNNlayout(self,neuronlist):
        self.numNeurons = neuronlist
        self.numLayers = len(self.numNeurons)
        self.w = []
        self.b = []
        self.y=[]
        self.n=[]
        self.Gmatrix = []
        self.sensitivities = []
        self.y.append(zeros((self.numNeurons[0], 1)))
        self.w_1 = []
        self.b_1 = []
        self.dw = []
        self.db = []
        for k in range(1,self.numLayers):
            self.w.append( self.w_init*(2*(random(( self.numNeurons[k], self.numNeurons[k-1])))-1) )
            self.b.append( self.b_init*(2*(random(( self.numNeurons[k], 1)))-1) )
            self.y.append(zeros((self.numNeurons[k], 1)))
            self.n.append(zeros((self.numNeurons[k], 1)))
            self.Gmatrix.append( zeros((self.numNeurons[k], self.numNeurons[k])))
            self.sensitivities.append(self.y[k]*0)
            self.w_1.append(0*self.w[k-1])
            self.b_1.append(0*self.b[k-1])
            self.dw.append(0*self.w[k-1])
            self.db.append(0*self.b[k-1])

    def setnumTrials(self,x):
        self.numTrials = x


    def setactivation(self,func):
        # Might want different activation functions for each layer
        self.activate = []
        self.diffactivate = []

        for k in range(len(self.n)):
            if(func == 'logsig'):
                self.activate.append( lambda x: 1.0/(1.0+exp(-x)))
                self.diffactivate.append( lambda x : exp(-x)/((1.0+exp(-x))**2))

            elif(func == 'linear'):
                self.activate.append( lambda x: x)
                self.diffactivate.append( lambda x: eye(1))

            elif(func == 'hardlimit'):
                self.activate.append( lambda x: (x > 0) + 0)
                self.diffactivate.append( lambda x: (sign(x)+1)/2)

            elif(func == 'tanh'):
                self.activate.append( lambda x: tanh(x))
                self.diffactivate.append( lambda x: (1 - tanh(x)**2))

    def settargetfunc(self,func):
        if self.numNeurons[0] == 1:
            self.targetfunc = eval("lambda x:" + func)
        else:
            self.targetfunc = eval(func)

    def setinputrange(self,*arg):
        self.pLow = arg[0]
        self.pHigh = arg[1]
        self.xx = linspace(self.pLow,self.pHigh,self.numTrials)
        try:
            self.pLowy = arg[2]
            self.pHighy = arg[3]
            self.yy = linspace(self.pLowy,self.pHighy,self.numTrials)
            self.pLowz = arg[4]
            self.pHighz = arg[5]
            self.zz = linspace(self.pLowz, self.pHighz, self.numTrials)
            self.xx = array([[self.xx,self.yy,self.zz]])
            self.xx = self.xx.T
        except IndexError:
            pass

    def setalphaend(self,x):
        self.dalpha = (x - self.alpha) / self.N

    def setmuend(self,x):
        self.dmu = (x - self.mu) / self.N

    def train(self):
        self.setNNlayout(self.numNeurons)
        for k in range(0,self.numTrials):

            for m in range(self.numNeurons[0]):
                 #self.y[0][m] = array([[uniform(self.pLow,self.pHigh)]])
                 self.y[0][m] = self.xx[k][m]

            self.y[0] = self.y[0].reshape((3,1))
            #self.y[0] = self.xx[k]
            #self.t = self.targetfunc(self.y[0])
            self.t = measured_data[k][1]
            self.t = self.t.reshape((3,1))

            # Forward propagation
            for m in range(self.numLayers-1):
                self.n[m] = self.w[m].dot(self.y[m]) + self.b[m]
                self.y[m+1] = self.activate[m](self.n[m])

            # Get the error for this trial
            E = self.t - self.y[-1]

            # Last Gmatrix and sensitivies
            self.Gmatrix[-1] = self.diffactivate[m](self.n[-1])
            self.sensitivities[-1] = -2*self.Gmatrix[-1] * E
            # Backpropagation of the sensitivities
            if ~self.momentumFlag:
                for m in reversed(range(self.numLayers,1)):
                    for j in range(self.numNeurons[m]):
                        # Just the diagonal elements of G are non-zero
                        self.Gmatrix[m-1][j][j] = self.diffactivate[m](self.n[m-1][j][0])
                    self.sensitivities[m-1] = self.Gmatrix[m-1].dot( (self.w[m].T).dot(self.sensitivities[m]))

                for m in range(self.numLayers-1):
                    self.w[m] = self.w[m] - self.alpha * self.sensitivities[m].dot(self.y[m].T)
                    self.b[m] = self.b[m] - self.alpha * self.sensitivities[m]
            else:
                for m in range(self.numLayers):
                    for m in reversed(range(self.numLayers-1)):
                        for j in range(self.numNeurons[m]):
                            self.Gmatrix[m-1][j][j] = self.diffactivate[m](self.n[m-1][j][0])
                        self.sensitivities[m-1] = self.Gmatrix[m-1].dot( (self.w[m].T).dot(self.sensitivities[m]))

                    for m in range(self.numLayers-1):
                        self.w[m] = self.w[m] - self.alpha * self.sensitivities[m].dot(self.y[m].T) - self.mu*self.w_1[m]
                        self.b[m] = self.b[m] - self.alpha * self.sensitivities[m] - self.mu*self.b_1[m]
                    self.w_1[m] = self.w[m]
                    self.b_1[m] = self.b[m]

            if self.saveFlag:
                for m in range(len(self.w)):
                    self.W[m][k] = self.w[m]
                    self.B[m][k] = self.b[m]

            self.alpha += self.dalpha
            self.mu += self.dmu


    def run(self):

        self.fnn = []
        self.fnnerr = []
        MSE = 0

        for k in range(self.numTrials):

            for m in range(self.numNeurons[0]):
                 #self.y[0][m] = array([[uniform(self.pLow,self.pHigh)]])
                 self.y[0][m] = self.xx[k][m]


            self.y[0] = self.y[0].reshape((3,1))

            # Forward propagation
            for m in range(self.numLayers-1):
                self.n[m] = self.w[m].dot(self.y[m]) + self.b[m]
                self.y[m+1] = self.activate[m](self.n[m])


            self.fnn.append(self.y[-1])
            self.fnnerr.append(self.t - self.y[-1])
            MSE += self.fnnerr[k]**2
        MSE = MSE / self.numTrials


        return self.fnn




    def plotWeights(self):
        leg = [] # Initialize the plot's legend
        for k in range(len(self.W)):
            for m in range(self.numNeurons[k]):
                plt.plot(self.W[k][:,0,m],'*')
                leg.append('w%i_%i' %(k,m))
        for k in range(len(self.B)):
            for m in range(self.numNeurons[k]):
                plt.plot(self.B[k][:,0,m],'^')
                leg.append('b%i_%i' %(k,m))
        plt.legend(leg)
        try:
            plt.show()
        except KeyboardInterrupt:
            plt.close()
