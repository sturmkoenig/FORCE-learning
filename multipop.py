# change to multipop15.py: based on multipop_c_13
#
#
# Moritz Deger, moritz.deger@epfl.ch, May 15, 2015
# Tilo Schwalger, tilo.schwalger@epfl.ch, May 15, 2015

import pdb

import numpy as np
try:
   import pandas
except:
   print('failed to import pandas')
import pylab
import os.path
import os

#import nest    #for use of NEST simulator
import multipop_c as mpc #custom-made C simulation of networks and populations
from numpy.fft import fft


class MultiPop(object):

    def __init__(self, dt=0.0002, rho_0=[10.,5.], \
                 tau_m=[0.01,0.01], tau_sfa=[[3.],[1.]], \
                 J_syn=[[0.05,-0.22],[0.05,-0.22]], delay=[[0.002,0.002],[0.002,0.002]], \
                 t_ref=[0.002,0.002], V_reset=[0.,0.], J_a=[[1.],[0.0]], \
                 pconn=np.ones((2,2))*1., \
                 mu=[0.,0.], delta_u=[4.,4.], V_th=[0.,0.], \
                 taus1=[[0,0.],[0.,0.]],\
                 taur1=[[0,0.],[0.,0.]],\
                 taus2=[[0,0.],[0.,0.]],\
                 taur2=[[0,0.],[0.,0.]],\
                 a1=[[1.,1.],[1.,1.]],\
                 a2=[[0,0.],[0.,0.]], Jref=[10.,10.],sigma=[0.,0.], mode='glif' \
            ):
        self.dt = dt
        #self.N = np.array(N)
        
        self.rho_0 = np.array(rho_0)
        self.K = len(self.rho_0)
        self.tau_m = np.array(tau_m)
        self.tau_sfa = tau_sfa
        self.delay = np.array(delay)
        self.t_ref = np.array(t_ref)
        self.V_reset = np.array(V_reset)
        self.V_th = np.array(V_th)
        self.J_a = J_a
        self.J_syn = np.array(J_syn)
        self.pconn = np.resize(pconn, (self.K,self.K))
        self.delta_u = delta_u
        self.mu = np.array(mu) 
        self.taus1=np.resize(taus1,(self.K,self.K))
        self.taur1=np.resize(taur1, (self.K,self.K))
        self.taus2=np.resize(taus2, (self.K,self.K))
        self.taur2=np.resize(taur2, (self.K,self.K))
        self.a1=np.resize(a1, (self.K,self.K))
        self.a2=np.resize(a2, (self.K,self.K))
        self.sigma=np.array(sigma)
        if mode=='glif':
            self.mode=10
            self.Jref=np.zeros(self.K)
        elif mode=='glif_master':
            self.mode=30
            self.Jref=np.zeros(self.K)
        elif mode=='glif4':
            self.mode=14
            self.Jref=np.zeros(self.K)
        elif mode=='glm':
            self.mode=0
            self.Jref=np.array(Jref)
        elif mode=='glif_naiv':
            self.mode=12
            self.Jref=np.zeros(self.K)
        elif mode=='glm_naiv':
            self.mode=2
            self.Jref=np.array(Jref)
        elif mode=='glm_master':
            self.mode=20
            self.Jref=np.array(Jref)
        else:
            self.mode=10
            self.Jref=np.zeros(self.K)
            
            
        assert(self.J_syn.shape==(self.K,self.K))
        
        # non-exposed but settable parameters
        self.len_kernel = -1    # -1 triggers automatic history size
        self.local_num_threads = 1 #2
        self.dt_rec = self.dt
        self.n_neurons_record_rate = 10
        self.origin=0. #time origin
        self.step=None
        self.tstep=None
        self.Nstep=None
        self.Ntstep=None
        
        # internal switches
        self.__sim_mode__ = None
        

    def __build_network_tilo_common__(self):

        self.mp = mpc.Multipop(self.dt_rec, self.dt, \
                       tref=self.t_ref, taum = self.tau_m, \
                       taus1=self.taus1, taur1=self.taur1, taus2=self.taus2, taur2=self.taur2, a1=self.a1, a2=self.a2, \
                       mu=self.mu, c=self.rho_0, DeltaV=self.delta_u, \
                       delay = self.delay[0], vth=self.V_th, vreset=self.V_reset, \
                       J = self.J_syn, \
                       p_conn=np.ones((self.K, self.K)) * self.pconn,\
                       Jref=self.Jref, J_theta= self.J_a, tau_theta= self.tau_sfa, sigma=self.sigma, mode=self.mode)


    def build_network_tilo_populations(self):

        # #build corresponding fully-connected network
        # self.J_syn *= self.pconn
        # self.pconn=np.ones((self.K, self.K))

        self.__build_network_tilo_common__()
        
        # save information that we are in population mode
        self.__sim_mode__ = 'pop_tilo'

    def build_network_tilo_neurons(self, Nrecord=[0,0],Vspike=30.):
        self.__build_network_tilo_common__()
        
        if (sum(Nrecord)==0):
            # save information that we are in neuron mode
            self.__sim_mode__ = 'netw_tilo'
        else:
            self.__sim_mode__ = 'netw_tilo_record_voltage'
            self.Nrecord=np.array(Nrecord)
            self.Nrecord.resize(self.K,refcheck=False)  #fills with zeros if less elemens than number of populations
            self.Vspike=Vspike

    def retrieve_sim_data(self):
        if self.__sim_mode__=='populations':
            self.retrieve_sim_data_populations()
        elif self.__sim_mode__=='neurons':
            self.retrieve_sim_data_neurons()
        else:
            print('No network has been built. Call build_network... first!')
        self.rate = self.get_firingrates()
    

    def __moving_average__(self,a, n=3) :
        """
        computes moving average with time window of length n along the second axis
        """
        print('compute moving average with window length', n)
        ret = np.cumsum(a, axis = 1, dtype=float)
        ret[:,n:] = ret[:,n:] - ret[:,:-n]
        return ret/ n 
    
    def sigmoid(self,x):
        return np.exp(x)/(1+np.exp(x))
  


    def simulate(self, T,step=None,tstep=None,Nstep=None,Ntstep=None, ncc=None, seed=365, seed_quenched=1, ForceSim=False):
        self.sim_T = T
        self.seed=seed
        if step is None:
           self.step=None
           self.tstep=None         
        else: 
           self.step=np.array(step)
           self.tstep=np.array(tstep)
        if Nstep is None:
           self.Nstep=None
           self.Ntstep=None         
        else: 
           self.Nstep=np.array(Nstep)
 #          print(Nstep)
           self.Ntstep=np.array(Ntstep)
#           print(Ntstep)
         
        fname=self.__trajec_name__()
        #print fname
        if os.path.exists(fname) and not ForceSim:
            print('load existing trajectories')
            print(fname)
            f=np.load(fname)
            self.sim_t=f['t']
            self.sim_a=f['a']
            self.sim_A=f['A']
            self.sim_h_tot=f['h_tot']
            if self.__sim_mode__=='netw_tilo_record_voltage':
                self.voltage=[np.vstack(f['V'][i]) for i in range(self.K)]
                self.threshold=[np.vstack(f['theta'][i]) for i in range(self.K)]
        else:
            if self.__sim_mode__==None:
                print('No network has been built. Call build_network... first!')

            elif self.__sim_mode__=='pop_tilo':
                self.mp.get_trajectory_pop(self.sim_T,step,tstep,Nstep,Ntstep,ncc,seed=seed)
                self.sim_sig = self.mp.signal.T #?????
                self.sim_A=self.mp.A.T
                self.sim_a=self.mp.a.T  
                self.sim_h_tot = self.mp.h_tot.T
                self.sim_t = self.dt_rec * np.arange(len(self.sim_A))

            elif self.__sim_mode__=='netw_tilo':
                self.mp.get_trajectory_neuron(self.sim_T,step,tstep,seed=seed, seed_quenched=seed_quenched)
                self.sim_A=self.mp.A.T
                self.sim_a=self.__moving_average__(self.mp.A,int(0.05/self.dt_rec)).T 
                self.sim_t = self.dt_rec * np.arange(len(self.sim_A))

            elif self.__sim_mode__=='netw_tilo_record_voltage':
                self.mp.get_trajectory_voltage_neuron(self.sim_T,self.Nrecord, self.Vspike,step,tstep,seed=seed, seed_quenched=seed_quenched)
                #transpose data such that the 1st axis refers to time, 2nd axis is population or neuron, respectively
                self.sim_A=self.mp.A.T
                self.sim_a=self.__moving_average__(self.mp.A,int(0.05/self.dt_rec)).T
                self.voltage=[v.T for v in self.mp.voltage]
                self.threshold=[theta.T for theta in self.mp.threshold]
                self.sim_t = self.dt_rec * np.arange(len(self.sim_A))

            else:
                # msd =self.local_num_threads * seed + 1 #master seed
                # nest.SetKernelStatus({'rng_seeds': range(msd, msd+self.local_num_threads)})
                # print nest.GetKernelStatus('rng_seeds')
                self.sim_t = np.arange(0., self.sim_T, self.dt_rec)
                self.sim_A = np.ones( (self.sim_t.size, self.K) ) * np.nan
                self.sim_a = np.ones_like( self.sim_A ) * np.nan
                self.sim_h_tot = np.ones( (self.sim_t.size, self.K) ) * np.nan
                
                
                if (step!=None):
                    #set initial value (at t0+dt) of step current generator to zero
                    t0=self.origin * 1e3
                    tstep = np.hstack((self.dt * np.ones((self.K,1)), self.tstep)) * 1e3
                    step =  np.hstack((np.zeros((self.K,1)), self.step))
                    # create the step current devices if they do not exist already
                    if not self.__dict__.has_key('nest_stepcurrent'):
                        self.nest_stepcurrent = nest.Create('step_current_generator', self.K )
                    # set the parameters for the step currents
                    for i in range(self.K):
                        nest.SetStatus( [self.nest_stepcurrent[i]], {
                           'amplitude_times': tstep[i] + t0,
                           'amplitude_values': step[i] / (self.tau_m[i] * 1e3 / self.C_m[i]), 'origin': t0, 'stop': self.sim_T * 1e3#, 'stop': self.sim_T * 1e3 + t0
                           })
                        pop_ = self.nest_pops[i]
                        if type(self.nest_pops[i])==int:
                            pop_ = [pop_]
                        nest.Connect( [self.nest_stepcurrent[i]], pop_, syn_spec={'weight':1.} )

                # simulate 1 step longer to make sure all self.sim_t are simulated
                nest.Simulate( (self.sim_T+self.dt) * 1e3 )
                self.retrieve_sim_data()

    def __rebin_log__(self,f,y,nbin):
       x=np.log10(f)
       df=f[1]-f[0]
       n=nbin+1
       dx=(x[-1]-x[0])/nbin
        
       left=x[0]-0.5*dx + dx*np.arange(nbin)
       right=left+dx
       xc=left+0.5*dx
       count=np.zeros(nbin)
       y_av=np.zeros(nbin)
       for i in range(len(x)):
          indx=int((x[i]-left[0])/dx)
          if indx>=nbin: break
          count[indx]+=1
          y_av[indx]+=y[i]
       for i in range(nbin):
          if count[i]>0:
             y_av[i]=y_av[i]/count[i]
          else:
             y_av[i]=np.nan
       fout=10**(xc[np.where(np.isnan(y_av)==False)])
       yout=y_av[np.where(np.isnan(y_av)==False)]
       return (fout,yout)


 
#    def get_psd(self, df=0.1, dt_sample=0.001, Ntrials=10, nproc=1, dpoints=100):
#        print('')
#        if self.__sim_mode__==None:
#           print('get_psd(): No network has been built. Call build_network... first!')
#        elif self.__sim_mode__=='pop_tilo':
#           print('+++ GET PSD FROM MESOSCOPIC SIMULATION +++')
#        elif (self.__sim_mode__=='netw_tilo') or (self.__sim_mode__=='netw_tilo_record_voltage'):
#           print('+++ GET PSD FROM MICROSCOPIC SIMULATION +++')
#
#        self.Ntrials=Ntrials
#        self.df=df
#
#        fname=self.__psd_name__()
#        print fname
#        if os.path.exists(fname):
#            print 'LOAD EXISTING PSD DATA'
#            X=np.loadtxt(fname)
#            self.freq=X[:,0]
#            self.psd=X[:,1:]
#            self.freq_log=[]
#            self.psd_log=[]
#            for i in range(self.K):
#               x,y = self.__rebin_log__(self.freq,self.psd[:,i],dpoints)
#               self.freq_log.append(x)
#               self.psd_log.append(y)
#        else:
#            if self.__sim_mode__=='pop_tilo':
#                self.mp.get_psd_pop(df=df, dt_sample=dt_sample, Ntrials=Ntrials, nproc=nproc)
#                
#                self.freq_log=[]
#                self.psd_log=[]
#                for i in range(self.K):
#                   x,y = self.__rebin_log__(self.mp.f,self.mp.SA[i],dpoints)
#                   self.freq_log.append(x)
#                   self.psd_log.append(y)
#                self.freq=self.mp.f
#                self.psd=self.mp.SA.T #each column corresponds to the psd of one population
#
#            elif (self.__sim_mode__=='netw_tilo') or (self.__sim_mode__=='netw_tilo_record_voltage'):
#
#                self.mp.get_psd_neuron(df=df, dt_sample=dt_sample, Ntrials=Ntrials, nproc=nproc)
#                self.freq_log=[]
#                self.psd_log=[]
#                for i in range(self.K):
#                   x,y = self.__rebin_log__(self.mp.f,self.mp.SA[i],dpoints)
#                   self.freq_log.append(x)
#                   self.psd_log.append(y)
#                self.freq=self.mp.f
#                self.psd=self.mp.SA.T #each column corresponds to the psd of one population
#            elif (self.__sim_mode__=='neurons'):
#                self.build_network_neurons() #Reset Nest
#                NFFT=int(1./(dt_sample*df)+0.5)
#                df=1./(NFFT*dt_sample)
#                Ntot=NFFT*Ntrials
#                self.simulate(Ntot*dt_sample+0.005) #simulate 5ms more
#                print 'rate NEST: ',self.get_firingrates()
#                self.freq=[]
#                self.psd=[]
#                self.freq_log=[]
#                self.psd_log=[]
#
#                for i in range(self.K):
#                   x=self.sim_A[NFFT:,i]
#                   L=len(x)
#                   x=x[:(L/NFFT)*NFFT].reshape((-1,NFFT))
#                   ntrials=x.shape[0]
#                   xF=fft(x)
#                   S=np.sum(np.real(xF*xF.conjugate()),axis=0)*dt_sample/(NFFT-1)/ntrials
#                   psd=S[1:NFFT/2]
#                   freq=df*np.arange(NFFT/2-1)+df
#                   f_log,psd_log = self.__rebin_log__(freq,psd,dpoints)
#                   self.psd.append(psd)
#                   self.freq.append(freq)
#                   self.freq_log.append(f_log)
#                   self.psd_log.append(psd_log)
#                   
#                   self.psd=np.array(self.psd).T
#                   self.freq=np.array(self.freq[0])
#                   
#            elif (self.__sim_mode__=='populations'):
#                self.build_network_populations() #Reset Nest
#                NFFT=int(1./(dt_sample*df)+0.5)
#                df=1./(NFFT*dt_sample)
#                Ntot=NFFT*Ntrials
#                self.simulate(Ntot*dt_sample+0.005) #simulate 5ms more
#                print 'rate NEST: ',self.get_firingrates()
#                self.freq=[]
#                self.psd=[]
#                self.freq_log=[]
#                self.psd_log=[]
#
#                for i in range(self.K):
#                   x=self.sim_A[NFFT:,i]
#                   L=len(x)
#                   x=x[:(L/NFFT)*NFFT].reshape((-1,NFFT))
#                   ntrials=x.shape[0]
#                   xF=fft(x)
#                   S=np.sum(np.real(xF*xF.conjugate()),axis=0)*dt_sample/(NFFT-1)/ntrials
#                   psd=S[1:NFFT/2]
#                   freq=df*np.arange(NFFT/2-1)+df
#                   f_log,psd_log = self.__rebin_log__(freq,psd,dpoints)
#                   self.psd.append(psd)
#                   self.freq.append(freq)
#                   self.freq_log.append(f_log)
#                   self.psd_log.append(psd_log)
#
#                   self.psd=np.array(self.psd).T
#                   self.freq=np.array(self.freq[0])
#
#
#
#    def __get_rate_cv__(self, isih, dt):
#        n=len(isih[:,0])
#        npop=len(isih[0])
#        t=(np.arange(n)+0.5)*dt
#        m1=np.zeros(npop)
#        m2=np.zeros(npop)
#        v=np.zeros(npop)
#        for i in range(npop):
#            m1[i]=sum(t*isih[:,i])*dt
#            m2[i]=sum(t*t*isih[:,i])*dt
#            v[i]=m2[i]-m1[i]**2
#        return (1./m1,np.sqrt(v)/m1)
#
#    def get_isistat(self, tmax=2., Nbin=200, Nspikes=10000):
#        print ''
#        if self.__sim_mode__==None:
#           print 'get_isistat(): No network has been built. Call build_network... first!'
#        elif self.__sim_mode__=='pop_tilo':
#           print '+++ get_isistat(): sim_mode must be netw_tilo or netw_tilo_record_voltage +++'
#        elif (self.__sim_mode__=='netw_tilo') or (self.__sim_mode__=='netw_tilo_record_voltage'):
#           print '+++ GET ISIH +++'
#
#        self.Nspikes=Nspikes
#        self.dt_isi=tmax/Nbin
#
#        fname=self.__isi_name__()
#        print fname
#        if os.path.exists(fname):
#            print ''
#            print 'LOAD EXISTING ISI DATA'
#            X=np.loadtxt(fname)
#            self.T_isi=X[:,0]
#            self.isih=X[:,1:]
#            dt=self.T_isi[1]-self.T_isi[0]
#            self.rate,self.cv = self.__get_rate_cv__(self.isih, dt)
#        else:
#            if (self.__sim_mode__=='netw_tilo') or (self.__sim_mode__=='netw_tilo_record_voltage'):
#                self.mp.get_isih_neuron(Nbin, self.dt_isi, Nspikes)
#                self.T_isi=(np.arange(Nbin)+0.5) * self.dt_isi
#                self.isih=self.mp.isih.T #each column corresponds to the psd of one population
#
#                dt=self.T_isi[1]-self.T_isi[0]
#                self.rate,self.cv = self.__get_rate_cv__(self.isih, dt)



    def get_firingrates(self, Tinit=0.1):
        nstart=int(Tinit/self.dt_rec)
        if (self.__sim_mode__=='populations' or self.__sim_mode__=='pop_tilo'):
           return np.mean(self.sim_a[nstart:],axis=0)
        else:
           return np.mean(self.sim_A[nstart:],axis=0)
    
    
    def plot_sim(self, title='',legend=None,t0=0):
        dt=self.sim_t[1]-self.sim_t[0]
        i0=int(t0/dt)
        pylab.figure(30)
        if (self.__sim_mode__=='netw_tilo') or (self.__sim_mode__=='netw_tilo_record_voltage'):
            pylab.plot( self.sim_t[i0:], self.sim_A[i0:])
        else:
            pylab.plot( self.sim_t[i0:], self.sim_a[i0:])
        pylab.show()



    def xm_sim(self, param='sim.par',t0=0):
        dt=self.sim_t[1]-self.sim_t[0]
        i0=int(t0/dt)
        try: 
            self.xmtrajec.multi(1,1,hgap=0.3,vgap=0.3,offset=0.15)
        except: 
            self.xmtrajec=gracePlot(figsize=(720,540))
            self.xmtrajec.multi(1,1,hgap=0.3,vgap=0.3,offset=0.15)

        self.xmtrajec.focus(0,0)
        self.xmtrajec.plot( self.sim_t[i0:], self.sim_a[i0:])
        print("simulation time: ",self.sim_t[i0:])
        self.xmtrajec.grace('getp "%s"'%(param,))
        self.xmtrajec.grace('redraw')

#        
#    def plot_psd(self, title='',axis_scaling='loglog'):
#        pylab.figure(10)
#        if (axis_scaling=='loglog'):
#            pylab.loglog( self.freq, self.psd)
#        elif (axis_scaling=='semilogx'):
#            pylab.semilogx( self.freq, self.psd)
#        elif (axis_scaling=='semilogy'):
#            pylab.semilogy( self.freq, self.psd)
#        else:
#            pylab.plot( self.freq, self.psd)
#        pylab.xlabel('frequency [Hz]')
#        pylab.ylabel('psd [Hz]')
#        pylab.title(title)
#        pylab.show()
#
#    def xm_psd(self, param='psd.par'):
#        try: 
#            self.xmpsd.multi(1,1,hgap=0.3,vgap=0.3,offset=0.15)
#        except: 
#            self.xmpsd=gracePlot(figsize=(720,540))
#            self.xmpsd.multi(1,1,hgap=0.3,vgap=0.3,offset=0.15)
#
#        self.xmpsd.focus(0,0)
#        self.xmpsd.plot( self.freq, self.psd)
#        self.xmpsd.grace('getp "%s"'%(param,))
#        self.xmpsd.grace('redraw')


#    def plot_voltage(self,k=0,offset=0):
#        """
#        plot voltage traces for population k 
#        (1st population has index k=0)
#        """
#        if (self.Nrecord[k]>0):
#            pylab.figure(20+k)
#            Nbin=len(self.voltage[k])
#            t=self.dt_rec*np.arange(Nbin)
#            offset_matrix=np.outer(np.ones(Nbin),np.arange(self.Nrecord[k])) * offset
#            pylab.plot(t,self.voltage[k]+offset_matrix)
#            pylab.show()
#        else:
#            print 'Nrecord must be at least 1 to plot voltage!'
#
#    def xm_voltage(self,k=0,offset=0, param='voltage.par'):
#        """
#        plot voltage traces for population k 
#        (1st population has index k=0)
#        """
#        try: 
#            self.xmvolt.multi(1,1,hgap=0.3,vgap=0.3,offset=0.15)
#        except: 
#            self.xmvolt=gracePlot(figsize=(720,540))
#            self.xmvolt.multi(1,1,hgap=0.3,vgap=0.3,offset=0.15)
#
#        self.xmvolt.focus(0,0)
#
#        if (self.Nrecord[k]>0):
#            Nbin=len(self.voltage[k])
#            t=self.dt_rec*np.arange(Nbin)
#            offset_matrix=np.outer(np.ones(Nbin),np.arange(self.Nrecord[k])) * offset
#            self.xmvolt.plot(t,self.voltage[k]+offset_matrix)
#            self.xmvolt.grace('getp "%s"'%(param,))
#            self.xmvolt.grace('redraw')            
#        else:
#            print 'Nrecord must be at least 1 to plot voltage!'



    def __get_parameter_string__(self):


           
        if self.K>1:
           # if population mode take parameters of equivalent fully-connected network in file name
           if (self.__sim_mode__ == 'pop_tilo'):
              J1=self.J_syn[0][0] * self.pconn[0][0]
              J2=self.J_syn[0][1] * self.pconn[0][1]
              p1=1.
              p2=1.
           else:
              J1=self.J_syn[0][0]
              J2=self.J_syn[0][1]
              p1=self.pconn[0][0]
              p2=self.pconn[0][1]

           s='_mode%d_Npop%d_mu%g_du%g_vth%g_vr%g_c%g_J1_%g_J2_%g_p1_%g_p2_%g_taus1_%g_taus2_%g_taum%g_delay%g_tref%g_Na%d_Ja%g_taua%g_sigma%g'\
               %(self.mode,self.K,self.mu[0],self.delta_u[0],\
                    self.V_th[0], self.V_reset[0], self.rho_0[0], \
                    J1, J2, p1, p2, \
                    self.taus1[0][0], self.taus1[0][1], \
                    self.tau_m[0], \
                    #self.N[0], self.N[1],
                    self.delay[0][0],self.t_ref[0],len(self.J_a[0]),self.J_a[0][0],self.tau_sfa[0][0],self.sigma[0])
        else:
           # if population mode take parameters of equivalent fully-connected network in file name
           if (self.__sim_mode__ == 'pop_tilo'):
              J1=self.J_syn[0][0] * self.pconn[0][0]
              p1=1.
           else:
              J1=self.J_syn[0][0]
              p1=self.pconn[0][0]

           N_theta=len(self.J_a[0])
           if (N_theta>1):
              s='_mode%d_Npop%d_mu%g_du%g_vth%g_vr%g_c%g_J%g_p%g_taus1_%g_taum%g_N2_%d_delay%g_tref%g_Na%d_Ja1_%g_Ja2_%g_taua1_%g_taua2_%g_sigma%g'\
                 %(self.mode,self.K,self.mu[0],self.delta_u[0],\
                    self.V_th[0], self.V_reset[0], self.rho_0[0], \
                      J1, p1, \
                   self.taus1[0][0], \
                   self.tau_m[0],self.N[0], \
                   self.delay[0][0],self.t_ref[0],N_theta,self.J_a[0][0],self.J_a[0][1],self.tau_sfa[0][0],self.tau_sfa[0][1],self.sigma[0])

           else:
              s='_mode%d_Npop%d_mu%g_du%g_vth%g_vr%g_c%g_J%g_p%g_taus1_%g_taum%g_N2_%d_delay%g_tref%g_Na%d_Ja%g_taua%g_sigma%g'\
                 %(self.mode,self.K,self.mu[0],self.delta_u[0],\
                    self.V_th[0], self.V_reset[0], self.rho_0[0], \
                      J1, p1, \
                   self.taus1[0][0], \
                   self.tau_m[0],self.N[0], \
                   self.delay[0][0],self.t_ref[0],N_theta,self.J_a[0][0],self.tau_sfa[0][0],self.sigma[0])
        return s


#    def __psd_name__(self):
#        psd_str='_Ntrials%d'%(self.Ntrials,)
#        str2='_dt%g_dtbin%g_df%g.dat'%(self.dt,int(self.dt_rec/self.dt)*self.dt,self.df)
#        if self.__sim_mode__==None:
#            print 'No network has been built. Call build_network... first!'
#        elif self.__sim_mode__=='pop_tilo':
#            return 'data/psd_pop' + self.__get_parameter_string__() + psd_str + str2
#        elif (self.__sim_mode__=='netw_tilo') or (self.__sim_mode__=='netw_tilo_record_voltage'):
#            return 'data/psd_netw'+ self.__get_parameter_string__() + psd_str + str2
#        elif (self.__sim_mode__=='neurons'):
#            return 'data/psd_nestneur'+ self.__get_parameter_string__() + psd_str + str2
#        elif (self.__sim_mode__=='populations'):
#            return 'data/psd_nestpop'+ self.__get_parameter_string__() + psd_str + str2
#
#
#    def __isi_name__(self):
#        isi_str='_Nspikes%d'%(self.Nspikes,)
#        str2='_dt%g_dtbin%g.dat'%(self.dt,int(self.dt_isi/self.dt)*self.dt)
#        if self.__sim_mode__==None:
#            print 'No network has been built. Call build_network... first!'
#        elif (self.__sim_mode__=='netw_tilo') or (self.__sim_mode__=='netw_tilo_record_voltage'):
#            return 'data/isih_netw'+ self.__get_parameter_string__() + isi_str + str2


    def __trajec_name__(self):
       
        if self.step is not None:
           #use maximal step size in file name
           m=np.argmax(self.step)
           indx=np.unravel_index(m,self.step.shape)
           trajec_str='_step%g_tstep%g_T%g'%(self.step[indx],self.tstep[indx], self.sim_T)
        else:
           trajec_str='_T%g'%(self.sim_T,)
        str2='_dt%g_dtbin%g.npz'%(self.dt,int(self.dt_rec/self.dt)*self.dt)
        if self.__sim_mode__==None:
            print('No network has been built. Call build_network... first!')
        elif (self.__sim_mode__=='netw_tilo') or (self.__sim_mode__=='netw_tilo_record_voltage'):
            return 'data/trajec_netw'+ self.__get_parameter_string__() + trajec_str + '_seed%d'%(self.seed,) + str2
        elif self.__sim_mode__=='pop_tilo':
            return 'data/trajec_pop'+ self.__get_parameter_string__() + trajec_str + '_seed%d'%(self.seed,) + str2
        elif self.__sim_mode__=='neurons':
            return 'data/trj_nrn'+ self.__get_parameter_string__() + trajec_str + str2
        elif self.__sim_mode__=='populations':
            return 'data/trj_pop'+ self.__get_parameter_string__() + trajec_str + str2
        else:
            assert(False)

#
#    def save_psd(self):
#        fname=self.__psd_name__()
#        if os.path.exists(fname):
#            print 'file already exists. PSD not saved again.'
#        else:
#            if not os.path.exists('data'):
#                os.makedirs('data')
#            np.savetxt(fname,np.c_[self.freq,self.psd],fmt='%g')
#            print 'saved file ',fname
#
#    def clean_psd(self):
#        fname=self.__psd_name__()
#        if os.path.exists(fname):
#            os.remove(fname)  
#
#    def clean_isih(self):
#        fname=self.__psd_name__()
#        if os.path.exists(fname):
#            os.remove(fname)  

    def save_trajec(self):
        fname=self.__trajec_name__()
        if os.path.exists(fname):
            print('file already exists. Trajectories not saved again.')
        else:
            if not os.path.exists('data_trajectorv'):
                os.makedirs('data_trajectorv')
            if self.__sim_mode__=='netw_tilo_record_voltage':
                np.savez(fname,t=self.sim_t,A=self.sim_A, a=self.sim_a,V=self.voltage,theta=self.threshold)
            else:
                np.savez(fname,t=self.sim_t,A=self.sim_A, a=self.sim_a)

    def clean_trajec(self, fname=None):
        if fname==None:
           fname=self.__trajec_name__()
        if os.path.exists(fname):
            os.remove(fname)    

    def clean_all(self):
        self.clean_psd()
        self.clean_trajec()
        self.clean_isih()



#    def save_isih(self):
#        fname=self.__isi_name__()
#        if os.path.exists(fname):
#            print('file already exists. ISIH not saved again.')
#        else:
#            if not os.path.exists('data'):
#                os.makedirs('data')
#
#            np.savetxt(fname,np.c_[self.T_isi,self.isih],fmt='%g')
#            print 'saved file ',fname
#
#    def clean_isih(self):
#        fname=self.__isi_name__()
#        if os.path.exists(fname):
#            os.remove(fname)  
#
#            


    def get_singleneuron_rate_theory(self,i):
        """
        yields firing rate for a neuron in population i given constant input potential h
        no adaptation yet
        """
        tmax=0.
        dt=0.0001
        S_end=1
        while (S_end>0.001 and tmax<10.):
            tmax=tmax+1  #max ISI, in sec
            t=np.arange(0,tmax,dt)
            K=len(t)
            S=np.ones(K)
            if self.mode<10:
                eta=self.V_reset[i]*np.exp(-t/self.tau_m[i])
                rho=self.rho_0[i]*np.exp((self.mu[i]-eta-self.V_th[i])/self.delta_u[i])*(t>self.t_ref[i])
            else:
                v=self.mu[i]+(self.V_reset[i]-self.mu[i]) * np.exp(-(t-self.t_ref[i])/self.tau_m[i])
                rho=self.rho_0[i]*np.exp((v-self.V_th[i])/self.delta_u[i])*(t>self.t_ref[i])
            S=np.exp(-np.cumsum(rho)*dt)
            S_end=S[-1]
        return 1./(np.sum(S)*dt)

    def get_spike_afterpotential(self,k=0,tmax=0.5):
        t=linspace(0,tmax,200)
        v=self.mu+(self.V_reset-self.mu) * exp(-(t-self.t_ref)/self.tau_m)*(t>=self.t_ref)+ (t<self.t_ref)*self.V_reset


    def get_threshold_kernel(self,k=0,tmax=0.5,dt=0.001):
#       t=linspace(0,tmax,201)
       t=np.arange(0,tmax+0.5*dt,dt)
       n=len(t)
       theta=np.zeros(n)
       for i in range(len(self.J_a[k])):
          Ja=self.J_a[k][i]
          taua=self.tau_sfa[k][i]
          theta += Ja / taua * np.exp(-t / taua)
       return (t,theta)

            
def get_dataframe(nest_id):
    assert(type(nest_id)==int)
    return pandas.DataFrame( nest.GetStatus( [nest_id], keys=['events'] )[0][0])



