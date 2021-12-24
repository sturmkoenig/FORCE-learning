#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 22:54:41 2018

@author: mina
"""

from pylab import *
import matplotlib.ticker as mtick
import os,sys,inspect
from numpy.fft import fft

import multipop16_2 as mp
import multiprocessing as multproc
import time
from potjans import *
from tempfile import TemporaryFile
from scipy.linalg import block_diag
import random
from sklearn.preprocessing import StandardScaler


#
# simulation parameters.
#

thalamic_input = 'ON' #'ON' if 'ON' only principal/central column (PC) is stimulated with step simulation
optogenetic_inhibiton = 'ON' #'OFF' if on the number of inhibitory neurons is reduced by 40% in L23 and L5 
barrel_cortex = 'OFF'

ncc = 5 #number of cortical columns always odd for symmetry
cur_dir = os.getcwd()
B = 19.
pconn = np.load(cur_dir + '/example_connectome.npy' )

start_time = time.time()

if barrel_cortex == 'ON': #Compile the right Makefile before!
    N_full = { #Lefort C2 Mouse Barrel Cortex
          'L23': {'E':1691, 'I':230},
          'L4' : {'E':1656, 'I':140},
          'L5' : {'E':1095,  'I':221},
          'L6' : {'E':1288, 'I':127}}

    import multipop as mp
else:
    import multipop16_2 as mp     
    
#simulation time
t0 = 5.   #time for relaxation of initial transient
tend = t0 + 0.16
Ntrials_pop=100
nproc=4

#set parameters of step stimulation
tstart = t0 + 0.06
duration=0.06

dt=0.0005
dtpop=0.0005
dtbin=0.0005
dtbinpop=dtbin

dtrec= 0.0005
mode='glif'

seed=np.random.rand()


Tha_Drive_Vec = np.zeros(ncc*8)
midd_cc = int(float(ncc)/2.+0.5)

if thalamic_input == 'ON':
    Tha_Drive_Vec[(midd_cc-1)*8:midd_cc*8]=array(tha_drive_vec)

# network weight matrix
W = np.tile(weight_matrix,(ncc,ncc))

#%%
K = ncc*8
# parameter values of mu for non-adapting neurons obtained from separate fitting script to match rates of original Potjans & Diesmann model 
DeltaV = 5. * np.ones(K)
mu = np.array([ 19.14942021,  20.36247192,  30.80455243,  28.06868188, 29.43680143,  29.32954448,  34.93225564,  32.08123423])
mu = np.tile(mu,ncc)

rates = np.array([ 0.9736411 ,  2.86068612,  4.67324687,  5.64983048,  8.1409982 , 9.01281818,  0.98813263,  7.53034027])
rates = np.tile(rates,ncc)

#add adaptation (absent in original Potjans-Diesmann model)
tau_theta = [[1.],[0.5]]
tau_theta = tau_theta*4*ncc

J_theta =  [[1.],[0.]] 
J_theta = J_theta*4*ncc
J_theta = np.array(J_theta)

#calcalate mean adapataion current and compensate
mu_a = sum(J_theta,axis=1) * rates
mu += mu_a

#%%
c = np.ones(K, dtype=float) * 10.
Vreset = [neuron_params['v_reset'] - neuron_params['v_rest'] for k in range(K)]
Vth = [neuron_params['v_thresh'] - neuron_params['v_rest'] for k in range(K)]
delay = d_mean['E'] * 1e-3
t_ref = neuron_params['tau_refrac'] *1e-3

import ipdb; ipdb.set_trace()

N = np.array([ N_full['L23']['E'], N_full['L23']['I'],\
               N_full['L4']['E'], N_full['L4']['I'],\
               N_full['L5']['E'], N_full['L5']['I'],\
               N_full['L6']['E'], N_full['L6']['I'] ])
N = np.tile(N,ncc)

# turn psc amplitudes into psp amplitudes ---post synaptic current  -> post synaptic  potential
Js = W.copy()   

for i in range(len(W)):
    if i%2==0:
        Js[:,i] *= (neuron_params['tau_syn_E'] * 1e-3) / neuron_params['cm'] * 1e3
    else:        
        Js[:,i] *= (neuron_params['tau_syn_I'] * 1e-3)/ neuron_params['cm'] * 1e3

#%%
taus1_ = [neuron_params['tau_syn_E'] * 1e-3, neuron_params['tau_syn_I'] * 1e-3]
taus1_ = taus1_ + taus1_ + taus1_ + taus1_
taus1 = [taus1_ for k in range(K)]

taum = np.array([neuron_params['tau_m']*1e-3 for k in range(K)])

i0=int(t0/dtrec) #transit iarray_1d_int = npct.ndpointer(dtype=np.intc, ndim=1, flags='CONTIGUOUS')n time steps
step = np.hstack(( np.reshape( Tha_Drive_Vec, (ncc*8,1) ), np.zeros((ncc*8,1)) ))
if step.max()!=0:
    step = step / step.max() * B 
tstep=np.array([[tstart,tstart+duration] for k in range(K)])
                                         
#=======
if optogenetic_inhibiton == 'ON':
    N_Opto = N.copy()
    L23I = list(range(1,ncc*8,8))
    L5I = list(range(5,ncc*8,8))
    for i in range((midd_cc-1)*8,midd_cc*8): ##    
       
        if i in L23I:
            N_Opto[i]-=0.4*N_full['L23']['I']   
        if i in L5I:
            N_Opto[i]-=0.4*N_full['L5']['I']
    Nstep = np.hstack((np.reshape(N_Opto, (ncc*8,1) ),np.reshape(N, (ncc*8,1) )))
else:
    Nstep = np.hstack((np.reshape(N, (ncc*8,1) ), np.reshape(N, (ncc*8,1) )))
Ntstep = tstep
#######################################################################
    
def trajec_pop_c(seed):
    #simulation function to use for multiprocessing
    p2 = mp.MultiPop(dt=dtpop, rho_0=c, tau_m=taum, tau_sfa=tau_theta, J_syn=Js,
                     taus1=taus1, delay=np.ones((K,K))*delay, t_ref= np.ones(K)*t_ref,
                     V_reset=np.array(Vreset), J_a=J_theta, pconn=pconn, mu= np.array(mu),
                     delta_u= DeltaV*np.ones(K), V_th= Vth*np.ones(K),sigma=np.zeros(K),
                     mode=mode)
    p2.dt_rec=dtrec
    p2.build_network_tilo_populations()
    p2.simulate(tend,step=step,tstep=tstep, Nstep=Nstep, Ntstep=Ntstep, ncc=ncc, seed=seed)    #Nstep=Nstep, Ntstep=tstep,
   
    return {'h':p2.sim_h_tot,'A' :p2.sim_A}


#build file name 
p1 = mp.MultiPop(dt=dt, rho_0= c, tau_m = taum, tau_sfa=tau_theta, J_syn=Js,
                 taus1=taus1,delay=np.ones((K,K))*delay, t_ref= np.ones(K)*t_ref,
                 V_reset= np.array(Vreset), J_a=J_theta, pconn=pconn, mu= np.array(mu),
                 delta_u= DeltaV*np.ones(K), V_th= Vth*np.ones(K),sigma=np.zeros(K), mode=mode)
p1.dt_rec=dtrec
m=np.argmax(step)
indx=np.unravel_index(m,step.shape)
p1.build_network_tilo_neurons()

datastr = cur_dir + '/' + 'Tha%s_Op%s'%(thalamic_input,optogenetic_inhibiton) +\
          '_B%g_%g_tstep%g_T%g'%(B,step[indx],tstep[indx], tend) +\
          '_dt%g_dtbin%g_'%(dtpop,int(dtbinpop/dtpop)*dtpop) +\
          'Ntrials%d'%(Ntrials_pop) 

if not os.path.exists(datastr):
    os.makedirs(datastr) 

fname2 = datastr + '/'+ 'nn_test' + '.npz' 
   
print("results are will be writen to: ",fname2)

m = int(Ntrials_pop / nproc) #trials per processor
AH_part=[]
AH_all=[]
h_all=[]


#
# main loop
#
for n in range(m):
    print('part ', n+1)
    pool=multproc.Pool(processes=nproc)
    AH_part=pool.map(trajec_pop_c,np.arange(nproc)+n*nproc+1)
    pool.close()
    pool.join()
 

    AH_all += AH_part
        
#
# evaluate results
#
pop_A=np.zeros((Ntrials_pop,int(tend/dtrec),ncc*8))
pop_h=np.zeros((Ntrials_pop,int(tend/dtrec),ncc*8))

for i in range(Ntrials_pop):
    pop_Ap= np.array(AH_all[i]['A'])
    pop_hp = np.array(AH_all[i]['h'])
    pop_A[i] = pop_Ap
    pop_h[i] = pop_hp

L=pop_A.shape[1]
pop_t = dtrec * arange(L)


np.savez(fname2,t=pop_t,A=pop_A,h=pop_h)

end = time.time()
print("time elapsed: %f minutes" % ((end - start_time)/60))

