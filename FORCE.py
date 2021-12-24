import numpy as np
from sprandn import sprandn
import matplotlib.pyplot as plt
import math
import os
import signal 
import sys
from generatePlots import plotInterface


# Class defines 
class FORCE:
    def __init__(self, param, init_cond, learningPrequsits, posOut=False, dale=True, trueDale=False, dynplot=True, outweight=True, path="./run"):
        """
        constructor of FORCE class 

        Parameters
        ----------
        param:
            dict of all parameters needed by the simulation
        init_cond:
            dict of most initial state of the network.
        learningPrequsits:
            dict containing all information about the learning process itself.
            (simtime, population vectors usw.).
        posOut:
            declares if the output should be forced to be positive
        dale:
            boolean that sets is dale's principle should be enforced
        trueDale:
            enforces dale on population scale (as in the biological setting).
            Rather than on unit scale
        dynplot:
            boolean, if true a self updating plot is generated which shows 
            the simulation progress
        outweight:
            boolean, if false force is not allowed to change output weights 
            to learn a particular training function, only connection strength 
            are adaptable
        path:
            string, tells FORCE where to save results
        """
        # first of get everything out of the dictionary

        # number of neurons in the network
        self.N = param['N']
        self.p = param['p']
        self.x = init_cond['x']
        # weight for the output
        self.wo = init_cond['wo']
        self.dw = init_cond['dw']
        self.z = init_cond['z']
        self.InhibStrRatio = init_cond['InhibStrRatio']
        self.outStr = param['outStr']

        # connection matrix for the network
        self.M = init_cond['M']

        

        # g is a parameter sets how chaotic the network behaves before force is started
        # generally g ~ 1.5 is a good start
        self.g = param['g']

        # how many neurons in the network should be connected to the read out units
        self.nRec2Out = param['nRec2Out']
        self.alpha = param['alpha']
        self.dt = param['dt']
        self.simtime = learningPrequsits['simtime']
        self.stimOnTime = param['stimOnTime']
        self.stimOffTime = param['stimOffTime']
        
        # Istim is an array of the same shape as simtime and holds external stimuli to the network
        self.Istim = learningPrequsits['Istim']
        
        # array of populations receiving stimulation. Has structure [[stimulated Population, stim strength], ...]
        self.stimulatedPops =  learningPrequsits['stimulatedPops']

        # population vectors: define which neurons in the network should be grouped to learn a training function
        self.popV = learningPrequsits['popV']

        # physical distance between layers 
        #  self.dist = init_cond['dist']

        #  self.M = np.multiply(self.M,self.dist)
        # ft is the set of learning functions
        self.ft = learningPrequsits['ft']

        # how often is a training stimulus presented?
        self.reps = learningPrequsits['reps']

        # If true learing will be stopped but testing will still be conducted
        self.abort = False

        self.outweight = outweight

        self.steps = len(self.simtime)
        self.dynplot = dynplot
        self.r = np.tanh(self.x)
        self.ti = -1
        self.path=path

        # controll if there are enough trainingsfunctions for popVec
        assert self.ft.shape[0] == self.popV.shape[0]
        # controll if popV have correct dim
        assert self.popV.shape[1] == self.N

        # initialize all other values
        self.P = (1/self.alpha)*np.mat(np.eye(self.nRec2Out))

        # will hold the output during learning
        self.zt = np.zeros((len(self.popV),len(self.simtime)))

        # will hold the output during testing
        self.zpt = np.zeros((len(self.popV),len(self.simtime)))

        # number of neurons in external network that is responsible for Istim
        self.Nin = 1
        # connection probability from this extenal network to the force network
        self.Min = np.random.rand(self.N,self.Nin)
        self.I = np.mat(np.ones((self.Nin,1)))

        self.learn_every = 2
        self.posOut = posOut
        self.dale = dale
        self.lenRep = len(self.simtime)//self.reps
        self.wt= np.zeros(len(self.simtime)//self.learn_every)

        # define dales principle here
        if self.dale:
            self.Msign = np.ones((self.N,self.N))
            #  pdb.set_trace()
            
            for i in range(1,self.Msign.shape[0], 2):
                self.Msign[:,i].fill(-1)
                self.M[:,i] = self.M[:,i]*self.InhibStrRatio
            self.M = np.abs(self.M)
            self.M = np.multiply(self.M, self.Msign)

        if trueDale:
            #  alternative population based:
            for k in range(1,len(self.popV),2):
                vec = self.popV[k]
                for i in range(len(vec)):
                    if vec[i]==1:
                        self.Msign[:,i].fill(-1)
                        self.M[:,i] = self.M[:,i]*1.3
            self.M = np.abs(self.M)
            self.M = np.multiply(self.M, self.Msign)

        # here all initial plotting settings happen
        self.OUT = plotInterface(self, dynplot)
        np.savez(os.path.join(path, "initial_conditions"), M=self.M, wo=self.wo, dw =self.dw, x = self.x, ft = self.ft,Istim=self.Istim)
        np.savez(os.path.join(path, "param"), N=self.N, p=self.p, g=self.g,nRec2Out=self.nRec2Out,alpha=self.alpha,dt=self.dt)
        signal.signal(signal.SIGTSTP, self.exit_handler)

    def exit_handler(self,signal, frame):
        """
        This function is used to prematurely stop an simulation and start testing. 
        Used incase that FORCE learning time was picked way to long.
        """
        print('exiting simultation!')
        self.abort = True

    # currently not implemented
    def reshuffleNeurons(self):
        #  for pop in self.popV:
            #  for i in range(len(pop)):
                #  if pop[i]==1 and Msign>0:
        return "not working!"
          
    def rNext(self):
        '''
        calcultes r(x), in a function here because it is used multiple times 
        and sometimes changes
        '''
        #  self.r = (np.tanh(self.x)+1)/2.
        self.r = np.tanh(self.x)

    def learn(self):
        '''
        learning after FORCE, see "Generating Coherent Patterns of Activity from Chaotic Neural Networks"
        '''
        # update inverse correlation matrix
        k = np.array(self.P * self.r)
        rPr = np.ndarray.item(self.r.T * k)
        c = 1.0/(1.0 + rPr)
        self.P = self.P - k * (k.T * c)

        # update the error for the linear readout
        e = np.array(self.zt[:,self.ti] - self.ft[:,self.ti])

        # update the output weights
        self.dw = [-c*e[i]*self.P*self.r for i in range(len(self.popV))]

        #  self.wo = np.log(1 + np.exp( 15*( self.wo + np.sum(np.multiply(self.dw,self.popV),axis=0) ) ) )/15
        #  self.wo = np.abs(self.wo + np.sum(np.multiply(self.dw, self.popV), axis=0))
        self.wo = self.wo + np.sum(np.multiply(self.dw,self.popV),axis=0)
        #  self.wo[self.wo<0] = 0
        
        # update the internal weight matrix using the output's error
        #  self.M = self.M + np.multiply(self.dist, np.sum( [self.dw[i]*self.popV[i].T for i in range(len(self.popV))],axis=0).T)
        self.M = self.M + np.sum( [self.dw[i]*self.popV[i].T for i in range(len(self.popV))],axis=0).T

        if self.dale:
            self.M[np.sign(self.Msign) != np.sign(self.M)] = 0

    def test(self):
        '''
        testing of parameters. Generates output for a second time span,
        as long as the wanted learning time 
        '''
        print("begin testing phase...")
        for ti in range(self.steps):
            self.ti = ti
            z = self.timestep()
            self.zpt[:,self.ti] = z
            if ti % (self.lenRep//2) == 0:
                self.OUT.plot(self.zpt)
                if ti % self.lenRep == 0 and ti != 0: 
                    self.OUT.plotLast(self.zpt)
        return 0

    def timestep(self):
        '''
        Advances Network by one time increment,
        see "Generating Coherent Patterns of Activity from Chaotic Neural Networks"
        '''
        
        # sim, so x(t) and r(t) are created.
        #  self.x = (1.0-self.dt)*self.x + np.multiply(self.M,self.dist)*(self.r*self.dt) + self.Istim[self.ti]*np.sum([ np.multiply(self.Min, self.popV[i]) for i in self.stimulatedPops], axis=0)*self.I
        self.x = (1.0-self.dt)*self.x + self.M*(self.r*self.dt) + self.Istim[self.ti]*np.sum([ pop[1] * np.multiply(self.Min, self.popV[int(pop[0])]) for pop in self.stimulatedPops], axis=0)*self.I
        self.rNext()
        if self.outweight:
            z = np.asarray([self.wo.T * np.multiply(self.r,vec) for vec in self.popV]).flatten()
        else:
            z = np.asarray([np.full_like(self.wo.T,self.outStr) * np.multiply(self.r,vec) for vec in self.popV]).flatten()

        # if posOut==True softcut the result
        if self.posOut:
            z = np.log(1+np.exp(15*z))/15
        return z

    def save(self):
        '''
        saves all necessary variables so that in theory the network could be loaded and run from the saved timestep.
        The initial conditions are saved indepently
        '''
        np.savez(os.path.join(self.path, "netOut"), M=self.M, wo=self.wo, x=self.x, p=self.P, zt=self.zt, zpt=self.zpt, wt = self.wt)
        print("ok, all done!")
        plt.ioff()
        plt.savefig(os.path.join(self.path, "final_plot"))
    def load(self, fname):
        '''
        should initialize and give back a FORCE network with previously an
        previously saved state. Currently not implemented
        '''
        netParam = np.load(fname)
        self.M = netParam['M']
        self.wo = netParam['wo']
        self.x = netParam['x']

    def run(self):
        '''
        conductes a whole FORCE learning run on the given parameters. 
        This includes a timespan where FORCE is manipulating output weights
        and network connection strength to make the network output similar 
        to the given training functions (self.ft) and a testing timespan
        where the found weights are tested.
        '''
        for self.ti in range(self.steps):
            z = self.timestep()
            self.zt[:,self.ti] = z
            if self.ti % self.lenRep == 0 and self.ti != 0:
                self.OUT.plot(self.zt)
                self.OUT.plotLast(self.zt)
                self.OUT.plotGrad()
                self.OUT.printWeigth()
                if self.abort:
                    break
            if self.ti % self.learn_every == 0:
                self.learn()
                self.wt[self.ti//self.learn_every]  = np.sqrt(self.wo.T * self.wo)
        self.ti = 0
        self.test()
        self.save()

   
