#
# simulation parameters.
#

thalamic_input = 'ON' #'ON' if 'ON' only principal/central column (PC) is stimulated with step simulation
optogenetic_inhibiton = 'ON' #'OFF' if on the number of inhibitory neurons is reduced by 40% in L23 and L5
barrel_cortex = 'ON'

ncc = 5 #number of cortical columns always odd for symmetry
npp = 8 #number of populations per column

# TODO what is B??
B = 19.

#simulation time
t0 = 1.   #time for relaxation of initial transient
tend = t0 + 0.300
Ntrials_pop = 1
nproc=1 #4

#set parameters of step stimulation
tstart = t0 + 0.100
duration=0.06

dt=0.0005
dtpop=0.0005
dtbin=0.0005
dtbinpop=dtbin

dtrec= 0.0005
mode='glif'

