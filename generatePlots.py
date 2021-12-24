import matplotlib.pyplot as plt
import numpy as np
import pdb
#from column import run_column_model


class plotInterface:
    def __init__(self,FORCE, dynplot):

        self.FORCE = FORCE
        # Plot of population activity etc.
        fig = plt.figure(constrained_layout=True)
        gs = fig.add_gridspec(2*len(FORCE.popV)+3,3)
        self.lastRep = np.zeros((len(self.FORCE.popV), self.FORCE.lenRep))
        self.pl = []
        for i in range(len(self.FORCE.popV)):
            ax0 = fig.add_subplot(gs[2*i,:])
            ax0.plot(self.FORCE.simtime,self.FORCE.ft[i])
            self.plotPrequsite(ax0)
            ax0.set_xlim(0,self.FORCE.simtime[-1])


            plot, = ax0.plot(self.FORCE.simtime,self.FORCE.zt[i])
            self.pl.append(plot)
            # shows last iteration
            ax1 = fig.add_subplot(gs[2*i+1,:-1])
            ax1.plot(self.FORCE.simtime[0:self.FORCE.lenRep], self.FORCE.ft[i,0:self.FORCE.lenRep])
            ax1.set_yticks([])
            ax1.spines['left'].set_visible(False)
            if i in self.FORCE.stimulatedPops:
                ax1.axvspan(self.FORCE.stimOnTime,self.FORCE.stimOffTime, color='black',alpha=0.5)
            self.plotPrequsite(ax1)
            plot2, = ax1.plot(self.FORCE.simtime[0:self.FORCE.lenRep],self.lastRep[0])
            self.pl.append(plot2)

        # plot of connection matrix
        fig2,ax = plt.subplots()
        ax.set_xticks([])
        ax.set_yticks([])
        self.plot_M = ax.imshow(self.FORCE.M.T, cmap='seismic')
        fig.colorbar(self.plot_M, ax=ax)

        axw = fig.add_subplot(gs[-1,-1])
        self.plotPrequsite(ax)
        axw.set_ylim(-0.1,0.1)
        self.w_grad_plot, = axw.plot(self.FORCE.wt, color='black')
        self.dynplot = dynplot
        if dynplot:
            plt.ion()

    def plotPrequsite(self, ax):
        ax.set_xticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def plotGrad(self):
        '''
        Plots changes in wo, usefull to see if some strangly big changes occur during simultion time.
        '''
        self.w_grad_plot.set_ydata(np.gradient(self.FORCE.wt))

    def plot(self, zt):
        '''
        if dynplot is True here a plot is generated.
        '''
        for i in range(len(self.FORCE.popV)):
            self.pl[2*i].set_ydata(zt[i])
        self.plotPause()
        return 0
    def plotPause(self):
        if self.dynplot:
            plt.pause(0.01)

    def plotLast(self,zt):
        '''
        Normally training functions given to self.FORCE a periodic
        this function plot the last Repetition of the training function 
        together with the generated network output.
        '''
        iteration = self.FORCE.ti // self.FORCE.lenRep
        self.plot_M.set_data(self.FORCE.M.T)
        for i in range(len(self.FORCE.popV)):
            self.lastRep = zt[:,(iteration-1)*self.FORCE.lenRep:iteration*self.FORCE.lenRep]
            self.pl[2*i+1].set_ydata(self.lastRep[i])
        performance = self.performanceInFORCE(zt)
        self.plotPause()
        return performance
    def plotLastColumn(self):
        pop_count = self.FORCE.M.shape[0]//8
        M_redux = np.zeros((8,8))
        self.plot_M.set_data(self.FORCE.M.T)
        for i in range(8):
            for j in range(8):
                M_redux[i,j] = np.mean(self.FORCE.M[i*pop_count:(i+1)*pop_count,j*pop_count:(j+1)*pop_count])
        popA, _ = run_column_model(pconn_hyper=M_redux) 
        for i in range(len(self.FORCE.popV)):
            self.pl[2*i+1].set_ydata(popA[0,2000:2600,i]/250)
        self.performanceInPotjans(popA[0,2000:2600,:]/250)

    def performanceInFORCE(self,zt):
        iteration = self.FORCE.ti // self.FORCE.lenRep
        performance = np.sum(np.abs(zt[:,(iteration-1)*self.FORCE.lenRep:iteration*self.FORCE.lenRep] -self.FORCE.ft[:,:self.FORCE.ft.shape[1]//self.FORCE.reps] ))
        print("performance in FORCE: {}".format(-performance))
        return performance
    def performanceInPotjans(self,popA):
        performance = np.sum(np.abs(popA.reshape(8,600)-self.FORCE.ft[:,:self.FORCE.ft.shape[1]//self.FORCE.reps]))
        print("performance in potjans: {}".format(-performance))
        return

    def printWeigth(self):
        '''
        prints the norm of the weight vector |wo|. 
        This is a very helpfull metric. Big values (in dependency of the system) 
        indicate big positve and negative constituents that cancel out, which 
        is very likely unstable. In my experience values of order O(1) are excaptable
        '''
        print("|wo| = %f"% np.sqrt(self.FORCE.wo.T*self.FORCE.wo))
    
