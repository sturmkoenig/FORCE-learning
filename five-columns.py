#!/usr/bin/python3
import numpy as np
from sprandn import sprandn
import matplotlib.pyplot as plt
import math
import os
import sys
from FORCE import FORCE

# This py-file initalizes parameters and then runs the FORCE-learning process
dat = np.load('data/data_alpha0.57.npy')/250

# command-line argument to set output-dir
index = int(sys.argv[1])
# learning repetitions of the FORCE-learning process
reps = 40
# Number of units in FORCE network
N = 1600
# Number of units from FORCE-network connected to the read-out units
nRec2Out = N
# time-step of learning process
dt = 0.12
# seconds in recording data time (two time-steps in CCM-simulation process are 0.0005 seconds apart)
nsecs = int(0.0005*reps*len(dat))
# Factor setting the initial activity in the FORCE-network (prior to learning)
# for further information see the explanation of g in the publication 
# of schwalger et. Al (Generating Coherent Patterns of Activity from Chaotic Neural Networks)
g = 1.5
# connection probability of units in FORCE-network c
p = 1
# This factor is used to scale the initial connection strength to the total number of connections 
scale = 1/math.sqrt(p*N)
# The number of FORCE-sub networks (#Number of learning functions)
# One cortical column has 8 populations therefore here it is 8
pop_count = dat.shape[1]

simtime = np.linspace(0,nsecs,num=(len(dat)*reps))

# popV = population vector. Each population has one popV with entries that are either 1 if unit is assigned to that population
popV = np.zeros((pop_count,N,1))

for i in range(pop_count): 
    popV[i,(N//pop_count)*i:(N//pop_count)*(i+1)] = np.ones_like(popV[i,(N//pop_count)*i:(N//pop_count)*(i+1)])

Istim = np.zeros(len(simtime))
ft = np.array([np.tile(dat[:,i],reps) for i in range(pop_count)])

# time in seconds of stimulation onset
stimOnTime = 0.1
# time in seconds of stimulation off
stimOffTime = 0.16

for i in range(len(simtime)):
    if int(stimOnTime/0.0005) < i%len(dat)<int(stimOffTime/0.0005):
        Istim[i] = 0.12


# strength of read out vector (if outweight=False)
outStr = 0.06

dist = np.ones((N,N))
M = sprandn(N,N,p)*g*scale
M = np.mat(M.todense())

dirname = "run/index_{0}".format(index)
print(dirname)

try:
    os.mkdir(dirname)
except OSError:
    print("dir already exists")
param = {
        "N": N,
        "p": p,
        "g": g ,
        "nRec2Out": nRec2Out,
        "alpha": 1.0,
        "dt": dt,
        "outStr": outStr,
        "stimOnTime": stimOnTime,
        "stimOffTime": stimOffTime
        }

init_cond = {
        "x": 0.5*np.mat(np.random.randn(N,1)),
        "wo": np.mat(np.zeros((nRec2Out,1))),
        "dw": np.mat(np.zeros((nRec2Out,1))),
        "z": np.zeros(ft.shape[0]),
        "dist": dist,
        "M": M,
        "InhibStrRatio": 0.5
        }

learningPrequsits = {
        "simtime": simtime,
        "Istim": Istim,
        "stimulatedPops": np.array([[18,0.0983],[19,0.0619],[22,0.0512],[23,0.0196]]),
        "popV": popV,
        "ft": ft,
        "reps": reps
        }

simpleNet = FORCE(param, init_cond,learningPrequsits, posOut=True, dale=True, trueDale=False, dynplot=False,outweight=False, path=dirname)
simpleNet.run()
