#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 22:54:41 2018

@author: mina
"""

import numpy as np
import pylab as pl
import os,sys,inspect
import argparse
from numpy.fft import fft

import multiprocessing as multproc
import time
from tempfile import TemporaryFile
import random

import multipop16_2 as mp
from potjans import tha_drive_vec, weight_matrix, d_mean, neuron_params
from read_data import get_data_distribution

from parameters import *


def get_default_pconn():

    return np.load(os.getcwd() + '/example_connectome.npy' )


def run_column_model(pconn_hyper=None, seed=None, verbose=False, workdir=None, infile=None, save_params=True):
    """
    runs a complete simulation and returns network activations.

    Parameters
    ----------
    pconn_hyper : 
        start value for pconn?
    seed :
        seed for random number generator
    verbose : bool, optional
        if True prints progressbar to show completition status
        default is True
    workdir : str, optional
        sets the directory where the column modul should be executed
        if None the current directory is taken
        default is None
    infile : str, optional
        if not None hyper parameters are read from file workdir/infile
        important can't be definied when pconn_hyper is defined
        default is None
    save_params
        if True the following parameters are saved at workdir/parms.npz
        pconn=pconn, thalamic_input,
        optogenetic_inhibiton, B, step,
        tstep, tend, dtpop,
        dtbin=int(dtbinpop/dtpop)*dtpop, Ntrials_pop, seed
        default is True
    """

    cur_dir = os.getcwd()

    pconn_default = get_default_pconn()

    if workdir is None:
        workdir = cur_dir

    if infile is not None:
        print("rading hyper parameters from file: "+workdir+"/"+infile)
        assert(pconn_hyper is None)
        data = np.load(os.path.join(workdir,infile))
        if isinstance(data,np.ndarray):
            pconn = data
        else:
            pconn = data['pconn']
    elif pconn_hyper is not None:
        if pconn_hyper.shape==(8,8):
            pconn = pconn_default # np.zeros([8*ncc]*2)
            for n in range(ncc):
                pconn[n*npp:(n+1)*npp,n*npp:(n+1)*npp] = pconn_hyper
        else:
            assert(pconn_hyper.shape==(40,40))
            pconn = pconn_hyper
    else:
        pconn = pconn_default # np.load(cur_dir + '/example_connectome.npy' )

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

        N_full = { # Numbers of neurons in full-scale model as in potjans.py
                'L23': {'E':20683, 'I':5834},
                'L4' : {'E':21915, 'I':5479},
                'L5' : {'E':4850,  'I':1065},
                'L6' : {'E':14395, 'I':2948}}

        # DONE here one seed is generated but it should be one seed for every conducted trial
    if seed is None:
        seed=[int(10000*(i+1)*time.time()) for i in range(Ntrials_pop)]

    Tha_Drive_Vec = np.zeros(ncc*8)
    midd_cc = int(float(ncc)/2.+0.5)

    if thalamic_input == 'ON':
        Tha_Drive_Vec[(midd_cc-1)*8:midd_cc*8]=np.array(tha_drive_vec)

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
    mu_a = np.sum(J_theta,axis=1) * rates
    mu += mu_a

    #%%
    c = np.ones(K, dtype=float) * 10.
    Vreset = [neuron_params['v_reset'] - neuron_params['v_rest'] for k in range(K)]
    Vth = [neuron_params['v_thresh'] - neuron_params['v_rest'] for k in range(K)]
    delay = d_mean['E'] * 1e-3
    t_ref = neuron_params['tau_refrac'] *1e-3

    N = np.tile([ N_full['L23']['E'], N_full['L23']['I'],\
            N_full['L4']['E'], N_full['L4']['I'],\
            N_full['L5']['E'], N_full['L5']['I'],\
            N_full['L6']['E'], N_full['L6']['I'] ],ncc)

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



    #build file name 
    m=np.argmax(step)
    indx=np.unravel_index(m,step.shape)


    if workdir is None:
        workdir = cur_dir + '/' + 'Tha%s_Op%s'%(thalamic_input,optogenetic_inhibiton) +\
                '_B%g_%g_tstep%g_T%g'%(B,step[indx],tstep[indx], tend) +\
                '_dt%g_dtbin%g_'%(dtpop,int(dtbinpop/dtpop)*dtpop) +\
                'Ntrials%d'%(Ntrials_pop) +\
                '_seed%d'%(seed)

    assert(isinstance(workdir,str))

    if not os.path.exists(workdir):
        os.makedirs(workdir)

    m = int(Ntrials_pop / nproc) #trials per processor
    AH_part=[]
    AH_all=[]
    h_all=[]

    def trajec_pop_c(seed):
        """
        simulation function for an population.
        this is done using the parameters from parameters.py and uses
        the column modul as implemented in multipop.c 
        """
        pop = mp.MultiPop(dt=dtpop, rho_0=c, tau_m=taum, tau_sfa=tau_theta, J_syn=Js,
               taus1=taus1, delay=np.ones((K,K))*delay, t_ref= np.ones(K)*t_ref,
               V_reset=np.array(Vreset), J_a=J_theta, pconn=pconn, mu= np.array(mu),
               delta_u= DeltaV*np.ones(K), V_th= Vth*np.ones(K),sigma=np.zeros(K),
               mode=mode)
        pop.dt_rec=dtrec
        pop.build_network_tilo_populations()
        pop.simulate(tend,step=step,tstep=tstep, Nstep=Nstep, Ntstep=Ntstep, ncc=ncc, seed=seed)    #Nstep=Nstep, Ntstep=tstep,

        return [{'h':pop.sim_h_tot,'A' :pop.sim_A}]
    
    # choosen if calculation should be conducted on more than one procssor
    if nproc>1:
        pool=multproc.Pool(processes=nproc)
        AH_part=pool.map(trajec_pop_c,seed)
        pool.close()
        pool.join()
        AH_all += AH_part

    if verbose>0:
        bar = progressbar.ProgressBar(max_value=m)

    for n in range(m):
       if verbose>0:
           bar.update(n)
       AH_part = trajec_pop_c(seed[n])

       AH_all += AH_part

    if verbose>0:
        bar.finish()

    #
    # evaluate results
    # 
    pop_A=np.zeros((Ntrials_pop,int(tend/dtrec),ncc*npp))
    pop_h=np.zeros((Ntrials_pop,int(tend/dtrec),ncc*npp))

    for i in range(Ntrials_pop):
        pop_Ap= np.array(AH_all[i]['A'])
        pop_hp = np.array(AH_all[i]['h'])
        pop_A[i] = pop_Ap
        pop_h[i] = pop_hp

    L=pop_A.shape[1]
    pop_t = dtrec * np.arange(L)

    if save_params:
        fname2 = workdir + '/'+ 'params' + '.npz'

        np.savez(fname2, pconn=pconn, thalamic_input=thalamic_input,
                optogenetic_inhibiton=optogenetic_inhibiton, B=B, step=step[indx],
                tstep=tstep[indx], tend=tend, dt=dtpop,
                dtbin=int(dtbinpop/dtpop)*dtpop, Ntrials=Ntrials_pop, seed=seed)

        print("results have been writen to: ",fname2)

    end = time.time()
    # print("time elapsed: %d seconds" % (end - start_time))

    return (pop_A, pop_h)


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Simulate column model.')
    parser.add_argument('--seed', type=int, action='store', dest='seed', default=None, help='random seed')
    parser.add_argument('--verbose', action='store_true', dest='verbose', default=False, help='verbose')
    parser.add_argument('--workdir', type=str, action='store', dest='workdir', default=None, help='working directory')
    parser.add_argument('--input', type=str, action='store', dest='infile', default=None, help='input file')
    parser.add_argument('--output', type=str, action='store', dest='outfile', default=None, help='output file')
    parser.add_argument('--plot', action='store_true', dest='plot', default=None, help='generate plot')
    parser.add_argument('--mean', action='store_true', dest='mean', default=False, help='compute and save mean population activity')

    args = parser.parse_args()

    (pop_A, pop_h) = run_column_model(seed=args.seed, verbose=args.verbose, workdir=args.workdir, infile=args.infile, save_params=True)

    '''
    if --mean is given the mean over all n_pop_trials populations is 
    calculated and saved. --output has also to be set for this to work
    '''
    if args.mean:
        assert(args.outfile is not None)

    if args.outfile is not None:

        cids = np.arange(0,40)

        time = np.arange(0,tend-t0,dt)
        idx_start = int(t0/dt)
        idx_end = int(tend/dt)
        vmin=-10
        vmax=130
        dvp=2.0
        dtp=0.01

        nv = int((vmax-vmin)/dvp)-1
        nt = int((tend-t0)/dtp)

        # if mean is given save the mean over all populations, else save params and netresponse
        if args.mean:
            mean_pop_A = np.sum(pop_A, axis=0)/pop_A.shape[0]
            np.save(args.outfile, mean_pop_A)
        else: 
            net_response = np.zeros((ncc*npp,nt,nv))
            #  for j in range(cids.shape[0]):
                #  net_response[j,:,:] = get_data_distribution(time, pop_A[:,idx_start:idx_end,cids[j]].T.clip(0,120), dt=0.01, dv=2.0, vmin=-10, vmax=130)
            # np.savez(args.outfile, pop_A=pop_A, pop_h=pop_h, net_response=net_response)

            # only save pop_a for the moment to save memory
            np.save(args.outfile, pop_A)

        print('results have been written to: ', args.outfile)

    '''
    if the programm is run with --plot the mean over all n_pop_trials is 
    plotted and displayed over the full time.
    '''
    if args.plot:
        cids = np.arange(16,24,2)
        idx_start = int(t0/dt)
        idx_end = int(tend/dt)
        t_start = 0
        mean_pop_A = np.sum(pop_A[:,idx_start:idx_end], axis=0)
        fig = pl.figure(figsize=(15,15))
        gs = fig.add_gridspec(npp,ncc)
        gs.update(wspace=0,hspace=0)
        if ncc == 1:
            for i in range(8):
                ax = fig.add_subplot(gs[i,:])
                ax.plot(mean_pop_A[:,i])
        else:
            Titles = ['C1', 'C2', 'C3', 'C4','C5']
            Layers = ['L2/3', 'L4', 'L5', 'L6']
            Types = ['E','I']
            for i in range(ncc):
                for j in range(npp):
                    ax = fig.add_subplot(gs[j,i])
                    if j==0:
                        ax.set_title(Titles[i])
                    ax.set_ylim(0,200)
                    ax.plot(mean_pop_A[:,npp*i + j])
                    if i != 0:
                        ax.set_yticklabels([])
                    ax.annotate(Layers[j//2]+' '+Types[j%2],(-5.,150.))
        pl.show()



    #import pdb; pdb.set_trace()

