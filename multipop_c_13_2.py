# to build c libraries do
# make -f Makefile_fasthazard all
#change to multipop_c_12.py: based on Makefile_fasthazard, which uses lookup table for computing the hazard function

version=13

import numpy as np
import numpy.ctypeslib as npct
from ctypes import c_int,c_double
import multiprocessing as multproc
import pandas as pd
#np.set_printoptions(threshold='nan')



array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
array_1d_int = npct.ndpointer(dtype=np.intc, ndim=1, flags='CONTIGUOUS')
array_2d_int = npct.ndpointer(dtype=np.intc, ndim=2, flags='CONTIGUOUS')
array_2d_double = npct.ndpointer(dtype=np.double, ndim=2, flags='CONTIGUOUS')

print("running multi-population column model version: ",version)

# load library for simulation of population equations
lib = npct.load_library("glm_popdyn_%d"%(version,), ".")
lib.get_trajectory_with_2D_arrays.restype = None
lib.get_trajectory_with_2D_arrays.argtypes = [c_int, array_2d_double, array_2d_double, array_2d_double, \
                                                  c_int, array_1d_double, array_1d_double, \
                                                  array_2d_double, array_2d_double, \
                                                  array_2d_double, array_2d_double, \
                                                  array_2d_double, array_2d_double, \
                                                  array_1d_double,\
                                                  array_1d_double, array_1d_double, array_1d_double,\
                                                  array_1d_double, array_1d_double, \
                                                  array_2d_int, array_2d_double,\
                                                  array_2d_double, array_2d_double, array_1d_int,\
                                                  array_1d_double, array_1d_double, array_1d_double,\
                                                  array_1d_double, c_double,c_double,c_int,c_int] #dt,dtbin,mode

#lib.get_psd_with_2D_arrays.restype = None
#lib.get_psd_with_2D_arrays.argtypes = [c_int, array_2d_double, c_int, \
#                                           c_int, array_1d_double, array_1d_double, \
#                                           array_2d_double, array_2d_double, \
#                                           array_2d_double, array_2d_double, \
#                                           array_2d_double, array_2d_double, \
#                                           array_1d_double,\
#                                           array_1d_double, \
#                                           array_1d_double, array_1d_double,\
#                                           array_1d_double, array_1d_double,\
#                                           array_2d_int, array_2d_double,\
#                                           array_2d_double, array_1d_int,\
#                                           array_1d_double, array_1d_double, array_1d_double,\
#                                           array_1d_double, c_double,c_double,c_int] #dt,dtbin,mode



# load library for full spiking network simulation
#lib2 = npct.load_library(path+"glm_netw_sim", ".")
#lib2 = npct.load_library("../glm_netw_sim_%d"%(version,), ".")
#lib2 = npct.load_library("glm_netw_sim_%d"%(version,), ".")
#lib2.get_trajectory_srm_with_2D_arrays.restype = None
#lib2.get_trajectory_srm_with_2D_arrays.argtypes = [c_int, array_2d_double, \
#                                                  c_int, array_1d_double, array_1d_double, \
#                                                  array_2d_double, array_2d_double, \
#                                                   array_2d_double, array_2d_double, \
#                                                   array_2d_double, array_2d_double, \
#                                                  array_1d_double,\
#                                                  array_1d_double, array_1d_double, array_1d_double,\
#                                                  array_1d_double, array_1d_double, \
#                                                   array_2d_int, array_2d_double, \
#                                                  array_2d_double, array_2d_double, array_1d_int, \
#                                                  array_1d_double, array_1d_double, array_1d_double, \
#                                                   array_1d_double,  c_double,c_double,c_int,c_int,c_int]
#
#lib2.get_trajectory_voltage_srm_with_2D_arrays.restype = None
#lib2.get_trajectory_voltage_srm_with_2D_arrays.argtypes = [c_int, array_2d_double, \
#                                                           array_2d_double, \
#                                                           array_1d_int, c_double, c_int, \
#                                                           array_1d_double, array_1d_double, \
#                                                           array_2d_double, array_2d_double, \
#                                                           array_2d_double, array_2d_double, \
#                                                           array_2d_double, array_2d_double, \
#                                                           array_1d_double,\
#                                                           array_1d_double, array_1d_double, array_1d_double,\
#                                                           array_1d_double, array_1d_double, array_1d_int, array_2d_double,\
#                                                           array_2d_double, array_2d_double, array_1d_int,\
#                                                           array_1d_double, array_1d_double, array_1d_double,\
#                                                           array_1d_double, c_double,c_double, c_int,c_int,c_int]
#
#lib2.get_psd_srm_with_2D_arrays.restype = None
#lib2.get_psd_srm_with_2D_arrays.argtypes = [c_int, array_2d_double, c_int, \
#                                           c_int, array_1d_double, array_1d_double, \
#                                           array_2d_double, array_2d_double, \
#                                           array_2d_double, array_2d_double, \
#                                           array_2d_double, array_2d_double, \
#                                            array_1d_double,\
#                                           array_1d_double, array_1d_double, array_1d_double,\
#                                           array_1d_double,array_1d_double, \
#                                            array_2d_int, array_2d_double, \
#                                           array_2d_double, array_1d_int,\
#                                           array_1d_double, array_1d_double, array_1d_double,\
#                                           array_1d_double,  c_double,c_double,c_int]
#
#
#lib2.get_isih_with_2D_arrays.restype = None
#lib2.get_isih_with_2D_arrays.argtypes = [c_int, array_2d_double, c_int, \
#                                           c_int, array_1d_double, array_1d_double, \
#                                           array_2d_double, array_2d_double, \
#                                           array_2d_double, array_2d_double, \
#                                           array_2d_double, array_2d_double, \
#                                            array_1d_double,\
#                                           array_1d_double, array_1d_double, array_1d_double,\
#                                           array_1d_double,array_1d_double, \
#                                            array_1d_int, array_2d_double, \
#                                           array_2d_double, array_1d_int,\
#                                           array_1d_double, array_1d_double, array_1d_double,\
#                                           array_1d_double,  c_double,c_double,c_int]
#


# ################################################################################
# # Master equation (occupation number)
# ##############################
# lib3 = npct.load_library("../glm_mastereq_%d"%(version,), ".")
# lib3.get_trajectory_with_2D_arrays.restype = None
# lib3.get_trajectory_with_2D_arrays.argtypes = [c_int, array_2d_double, array_2d_double, \
#                                                   c_int, array_1d_double, array_1d_double, \
#                                                   array_2d_double, array_2d_double, \
#                                                   array_2d_double, array_2d_double, \
#                                                   array_2d_double, array_2d_double, \
#                                                   array_1d_double,\
#                                                   array_1d_double, array_1d_double, array_1d_double,\
#                                                   array_1d_double, array_1d_double, \
#                                                   array_1d_int, array_2d_double,\
#                                                   array_2d_double, array_2d_double, array_1d_int,\
#                                                   array_1d_double, array_1d_double, array_1d_double,\
#                                                   array_1d_double, c_double,c_double,c_int,c_int] #dt,dtbin,mode

# lib3.get_psd_with_2D_arrays.restype = None
# lib3.get_psd_with_2D_arrays.argtypes = [c_int, array_2d_double, c_int, \
#                                            c_int, array_1d_double, array_1d_double, \
#                                            array_2d_double, array_2d_double, \
#                                            array_2d_double, array_2d_double, \
#                                            array_2d_double, array_2d_double, \
#                                            array_1d_double,\
#                                            array_1d_double, \
#                                            array_1d_double, array_1d_double,\
#                                            array_1d_double, array_1d_double,\
#                                            array_1d_int, array_2d_double,\
#                                            array_2d_double, array_1d_int,\
#                                            array_1d_double, array_1d_double, array_1d_double,\
#                                            array_1d_double, c_double,c_double,c_int] #dt,dtbin,mode

################################################################################


#wrapper functions for multiprocessing of psd's

#def get_psd_pop(params):
#    lib.get_psd_with_2D_arrays(*params)
#    return params[1]
#
## def get_psd_master(params):
##     lib3.get_psd_with_2D_arrays(*params)
##     return params[1]
#
#def get_psd_neuron(params):
#    lib2.get_psd_srm_with_2D_arrays(*params)
#    return params[1]




class Multipop(object):

    def __init__(self, dtbin,dt, \
                     tref=[0.002,0.002],taum=[0.01,0.01], \
                     taus1=[[0,0.],[0.,0.]], taus2=[[0,0.],[0.,0.]], \
                     taur1=[[0,0.],[0.,0.]], taur2=[[0,0.],[0.,0.]], \
                     a1=[[0,0.],[0.,0.]], a2=[[0,0.],[0.,0.]], \
                     mu=[0.,0.], c=[10.,5.], DeltaV=[4., 4.], \
                     delay=[0.002,0.002], vth=[0.,0.], vreset=[0.,0.], \
                     #N=[400,100],
                     J=[[20,-22.],[20.,-22.]], \
                     p_conn=[[1.,1.],[1.,1.]], \
                     Jref=[0.,0.], J_theta=[[1.],[]], tau_theta=[[3.],[]], sigma=[0.,0.], mode=10):

        self.dt=dt
        self.dtbin=dtbin
        self.tref=np.array(tref)
        self.taum=np.array(taum)
        self.sigma=np.array(sigma)
        self.taus1=np.ascontiguousarray(taus1)
        self.taus2=np.ascontiguousarray(taus2)
        self.taur1=np.ascontiguousarray(taur1)
        self.taur2=np.ascontiguousarray(taur2)
        self.a1=np.ascontiguousarray(a1)
        self.a2=np.ascontiguousarray(a2)
        self.mu=np.array(mu)
        self.c=np.array(c)
        self.D=np.array(DeltaV)
        self.delay=np.array(delay)
        self.vth=np.array(vth)
        self.vreset=np.array(vreset)
        #self.N=np.array(N,dtype=np.intc)
        self.J=np.ascontiguousarray(J)
        self.p_conn=np.ascontiguousarray(p_conn)
        self.Jref=np.array(Jref)
        self.mode=mode #0:GLM, 10:GLIF, 20:GLM master, 30:GLIF master

        
        #bereinige Nullen in J_theta        
        self.J_theta=[]
        self.tau_theta=[]
        for i in range(len(J_theta)):
            L=J_theta[i]
            TAU=tau_theta[i]
            if L==[]:
                self.J_theta.append([])
                self.tau_theta.append([])
            else:
                Jlist=[]
                taulist=[]
                for j in range(len(L)):
                    J=L[j]
                    if J!=0.:
                        Jlist.append(J)
                        taulist.append(TAU[j])
                self.J_theta.append(Jlist)
                self.tau_theta.append(taulist)

        N_theta=[len(i) for i in self.J_theta]
        self.N_theta=np.array(N_theta,dtype=np.intc)

        self.J_theta_1d=np.hstack(self.J_theta)
        self.tau_theta_1d=np.hstack(self.tau_theta)

        self.Npop=len(tref)
        assert len(taum)==self.Npop
        assert len(mu)==self.Npop
        assert len(c)==self.Npop
        assert len(DeltaV)==self.Npop
        assert len(delay)==self.Npop
        assert len(vth)==self.Npop
        #assert len(N[0])==self.Npop
        assert len(N_theta)==self.Npop
        assert len(Jref)==self.Npop





    def get_trajectory_pop(self, Tsim, step=[],tstep=[], Nstep=[], Ntstep=[], ncc=[], seed=365):

        self.Nbin=int(Tsim/self.dtbin)
        self.A=np.zeros((self.Npop,self.Nbin),dtype=float)
        self.h_tot=np.zeros((self.Npop,self.Nbin),dtype=float)
        self.a=np.zeros((self.Npop,self.Nbin),dtype=float)
        t=np.arange(self.Nbin) * self.dtbin
        self.signal=np.zeros((self.Npop,self.Nbin),dtype=float)
        self.N=np.zeros((self.Npop,self.Nbin),dtype=np.intc) 
        NInit = np.array([20683,  5834, 21915,  5479,  4850,  1065, 14395,  2948])
#        NInit = np.array([4504,1270,3558,889,3257,715,4027,825])
        NInit = np.tile(NInit,ncc)
        
                
        for i in range(self.Nbin):
            self.N[:,i]=NInit
              
        
        if step is not None:
           step = np.array(step)
           tstep = np.array(tstep)

           for k in range(self.Npop):
            
            #sort step times to ensure temporal order
               s=np.argsort(tstep[k])
               step_time=np.array(tstep[k])[s]
               amp=np.array(step[k])[s]
               for i in range(len(amp)):
                   for j in range(self.Nbin):
                       if (t[j]>=step_time[i]):
                           self.signal[k,j] = amp[i]
        #print self.signal                  
        if Nstep is not None:
           Nstep = np.array(Nstep)
           Ntstep = np.array(Ntstep)
           #print Nstep

           for k in range(self.Npop):
            
            #sort step times to ensure temporal order
               s=np.argsort(Ntstep[k])
               Nstep_time=np.array(Ntstep[k])[s]
               Namp=np.array(Nstep[k])[s]
               
               for i in range(len(Namp)):
                   for j in range(self.Nbin):
                       if (t[j]>=Nstep_time[i]):
                           self.N[k,j] = Namp[i]
          #                 print k,j, self.N[k,j] 
        #print self.N 
        if (self.mode<20):
            lib.get_trajectory_with_2D_arrays(self.Nbin, self.h_tot, self.A, self.a, self.Npop, self.tref, self.taum, self.taus1, self.taus2, self.taur1, self.taur2, self.a1, self.a2, self.mu, self.c, self.D, self.delay, self.vth, self.vreset, self.N, self.J, self.p_conn, self.signal, self.N_theta, self.Jref, self.J_theta_1d, self.tau_theta_1d, self.sigma, self.dt, self.dtbin,self.mode,seed)
        # else:
        #     lib3.get_trajectory_with_2D_arrays(self.Nbin, self.A, self.a, self.Npop, self.tref, self.taum, self.taus1, self.taus2, self.taur1, self.taur2, self.a1, self.a2, self.mu, self.c, self.D, self.delay, self.vth, self.vreset, self.N, self.J, self.p_conn, self.signal, self.N_theta, self.Jref, self.J_theta_1d, self.tau_theta_1d, self.sigma, self.dt, self.dtbin,self.mode,seed)

    
    
#    def get_trajectory_voltage_neuron(self, Tsim, Nrecord, Vspike=90.,step=0,tstep=0., seed=365, seed_quenched=1):
#
#        self.Nbin=int(Tsim/self.dtbin)
#        self.A=np.zeros((self.Npop,self.Nbin),dtype=float)
#        self.a=np.zeros((self.Npop,self.Nbin),dtype=float)
#        t=np.arange(self.Nbin) * self.dtbin
#        self.signal=np.zeros((self.Npop,self.Nbin),dtype=float)
#
#        if step is not None:
#           step = np.array(step)
#           tstep = np.array(tstep)
#
#           for k in range(self.Npop):
#               
#            #sort step times to ensure temporal order
#               s=np.argsort(tstep[k])
#               step_time=np.array(tstep[k])[s]
#               amp=np.array(step[k])[s]
#               for i in range(len(amp)):
#                   for j in range(self.Nbin):
#                       if (t[j]>=step_time[i]):
#                           self.signal[k,j] = amp[i]
#
#        Nrecord=np.array(Nrecord,dtype=np.intc)
#        assert len(Nrecord)==self.Npop
#        Nrecord_tot=Nrecord.sum()
#        voltage_matrix=np.zeros((2*Nrecord_tot,self.Nbin),dtype=float)
#        print voltage_matrix.shape
#
#        lib2.get_trajectory_voltage_srm_with_2D_arrays(self.Nbin, self.A, voltage_matrix, Nrecord, Vspike, self.Npop, self.tref, self.taum, self.taus1, self.taus2, self.taur1, self.taur2, self.a1, self.a2, self.mu, self.c, self.D, self.delay, self.vth, self.vreset, self.N, self.J, self.p_conn, self.signal, self.N_theta, self.Jref, self.J_theta_1d, self.tau_theta_1d, self.sigma, self.dt, self.dtbin,self.mode, seed, seed_quenched)
#        self.voltage=[]
#        self.threshold=[]
#        indx=0
#        for i in range(self.Npop):
#            self.voltage.append(voltage_matrix[indx:indx+Nrecord[i],:])
#            indx2=indx+Nrecord_tot
#            self.threshold.append(voltage_matrix[indx2:indx2+Nrecord[i],:]+self.vth[i])
#            indx+=Nrecord[i]


#
#    def get_psd_pop(self, df=0.1, dt_sample=0.001, Ntrials=10,nproc=1):
#        Nbin=int(1./(df*dt_sample)+0.5)
#        Nf=int(Nbin/2)
#        self.f = df * (np.arange(Nf) + 1)
#
#        if (nproc > 1):
#            Ntrials_part=int(Ntrials/nproc)
#            SA_list=[np.zeros((self.Npop, Nf),dtype=float) for i in range(nproc)]
#            pool=multproc.Pool(processes=nproc)
#            
#            params=[(Nf, SA_list[i], Ntrials_part, self.Npop, self.tref, self.taum, self.taus1, self.taus2, self.taur1, self.taur2, self.a1, self.a2, self.mu, self.c, self.D, self.delay, self.vth, self.vreset, self.N, self.J, self.p_conn, self.N_theta, self.Jref, self.J_theta_1d, self.tau_theta_1d, self.sigma, self.dt, dt_sample,self.mode) for i in range(nproc)]
#            if (self.mode<20):
#                SA_list=pool.map(get_psd_pop,  params)
#            else:
#                SA_list=pool.map(get_psd_master,  params)
#                
#            self.SA= np.array(SA_list).mean(axis=0)
#        else:
#            self.SA= np.zeros((self.Npop, Nf),dtype=float)
#
#            if (self.mode<20):
#                lib.get_psd_with_2D_arrays(Nf, self.SA, Ntrials, self.Npop, self.tref, self.taum, self.taus1, self.taus2, self.taur1, self.taur2, self.a1, self.a2, self.mu, self.c, self.D, self.delay, self.vth, self.vreset, self.N, self.J, self.p_conn, self.N_theta, self.Jref, self.J_theta_1d, self.tau_theta_1d, self.sigma, self.dt, dt_sample,self.mode)
#            # else:
#            #     lib3.get_psd_with_2D_arrays(Nf, self.SA, Ntrials, self.Npop, self.tref, self.taum, self.taus1, self.taus2, self.taur1, self.taur2, self.a1, self.a2, self.mu, self.c, self.D, self.delay, self.vth, self.vreset, self.N, self.J, self.p_conn, self.N_theta, self.Jref, self.J_theta_1d, self.tau_theta_1d, self.sigma, self.dt, dt_sample,self.mode)



    def get_trajectory_neuron(self, Tsim,step=0,tstep=0., seed=365, seed_quenched=1):

        self.Nbin=int(Tsim/self.dtbin)
        self.A=np.zeros((self.Npop,self.Nbin),dtype=float)
        t=np.arange(self.Nbin) * self.dtbin
        self.signal=np.zeros((self.Npop,self.Nbin),dtype=float)

        if step is not None:
           step = np.array(step)
           tstep = np.array(tstep)

           for k in range(self.Npop):
               
            #sort step times to ensure temporal order
               s=np.argsort(tstep[k])
               step_time=np.array(tstep[k])[s]
               amp=np.array(step[k])[s]
               for i in range(len(amp)):
                   for j in range(self.Nbin):
                       if (t[j]>=step_time[i]):
                           self.signal[k,j] = amp[i]
                           
        lib2.get_trajectory_srm_with_2D_arrays(self.Nbin, self.A, self.Npop, self.tref, self.taum, self.taus1, self.taus2, self.taur1, self.taur2, self.a1, self.a2, self.mu, self.c, self.D, self.delay, self.vth, self.vreset, self.N, self.J, self.p_conn, self.signal, self.N_theta, self.Jref, self.J_theta_1d, self.tau_theta_1d, self.sigma, self.dt, self.dtbin,self.mode, seed, seed_quenched)


#
#    def get_psd_neuron(self, df=0.1, dt_sample=0.001, Ntrials=10, nproc=1):
#        Nbin=int(1./(df*dt_sample)+0.5)
#        Nf=int(Nbin/2)
#        self.f = df * (np.arange(Nf) + 1)
#
#        if (nproc > 1):
#            Ntrials_part=int(Ntrials/nproc)
#            SA_list=[np.zeros((self.Npop, Nf),dtype=float) for i in range(nproc)]
#            pool=multproc.Pool(processes=nproc)
#            
#            params=[(Nf, SA_list[i], Ntrials_part, self.Npop, self.tref, self.taum, self.taus1, self.taus2, self.taur1, self.taur2, self.a1, self.a2, self.mu, self.c, self.D, self.delay, self.vth, self.vreset, self.N, self.J, self.p_conn, self.N_theta, self.Jref, self.J_theta_1d, self.tau_theta_1d, self.sigma, self.dt, dt_sample,self.mode) for i in range(nproc)]
#            SA_list=pool.map(get_psd_neuron,  params)
#
#            self.SA= np.array(SA_list).mean(axis=0)
#        else:
#            self.SA= np.zeros((self.Npop, Nf),dtype=float)
#            lib2.get_psd_srm_with_2D_arrays(Nf, self.SA, Ntrials, self.Npop, self.tref, self.taum, self.taus1, self.taus2, self.taur1, self.taur2, self.a1, self.a2, self.mu, self.c, self.D, self.delay, self.vth, self.vreset, self.N, self.J, self.p_conn, self.N_theta, self.Jref, self.J_theta_1d, self.tau_theta_1d, self.sigma, self.dt, dt_sample,self.mode)
#
#
#
#    def get_isih_neuron(self, Nisi,dt_isi, Nspikes):
#        self.isih=np.zeros((self.Npop, Nisi),dtype=float)
#        lib2.get_isih_with_2D_arrays(Nisi, self.isih, Nspikes, self.Npop, self.tref, self.taum, self.taus1, self.taus2, self.taur1, self.taur2, self.a1, self.a2, self.mu, self.c, self.D, self.delay, self.vth, self.vreset, self.N, self.J, self.p_conn, self.N_theta, self.Jref, self.J_theta_1d, self.tau_theta_1d, self.sigma, self.dt, dt_isi,self.mode)
