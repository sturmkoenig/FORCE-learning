#!/usr/bin/python3
import numpy as np
from sprandn import sprandn
import matplotlib.pyplot as plt
import math
import os
from FORCE import FORCE
import sys


dat = np.load("./data/200samplesMeanAndDeviation.npz")['mean']
dat0 = dat[8000:8600,18]/250
reps = 50

N = 1000
nRec2Out = N
dt = 0.12
nsecs = int(dt*reps*len(dat0))
g = 1.5
p = 0.5
scale = 1/math.sqrt(p*N)

simtime = np.linspace(0,nsecs,num=(len(dat0)*reps))

vec1 = np.ones((N,1))

ft = np.array([np.tile(dat0, reps)])
popV = np.array([vec1])

stimOnTime = 0.1
stimOffTime = 0.16
Istim = np.zeros(len(simtime))
for i in range(len(simtime)):
    if int(stimOnTime/0.0005) < i%len(dat0)<int(stimOffTime/0.0005):
        Istim[i] = 0.12

for i in range(10):
	M = sprandn(N,N,p)*g*scale
	M = np.mat(M.todense())
	dirname = 'run/timestep_'+str(dt)+'_try_'+str(i)
	try:
	    os.mkdir(dirname)
	except OSError:
	    print("dir already exists")
	init_cond = {
	        "x": 0.5*np.mat(np.random.randn(N,1)),
	        "wo": np.mat(np.zeros((nRec2Out,1))),
	        "dw": np.mat(np.zeros((nRec2Out,1))),
	        "z": np.zeros(ft.shape[0]),
	        "M": M,
		"dist": np.full((N,N),1.0),
		"InhibStrRatio": 1.3
	        } 
	
	learningPrequsits = {
	        "simtime": simtime,
	        "Istim": Istim,
	        "stimulatedPops": np.array([[0,0.1]]),
	        "popV": popV,
	        "ft": ft,
	        "reps": reps
	        }
	
	# holds everything thats constant
	param = {
	        "N": N,
	        "p": p,
	        "g": g ,
	        "nRec2Out": nRec2Out,
	        "alpha": 1.0,
	        "dt": dt,
		"outStr": 0.06,
        	"stimOnTime": stimOnTime,
        	"stimOffTime": stimOffTime
	        }
	
	simpleNet = FORCE(param, init_cond,learningPrequsits, posOut=False, dale=True, trueDale=False, dynplot=False, outweight=False, path=dirname)
	simpleNet.run()
