from IPython.display import clear_output
import collections
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

data_dict = collections.defaultdict(list)
        
def signif(t_test,threshold=2):
    sig = np.zeros(len(t_test))*np.nan
    sig[abs(t_test) > threshold]=1
    return sig

class Compute_Correlation():
    """Class to compute correlation between two datasets"""
    
    def __init__(self,y1, y2,nc=None,method="Trim"):
        # De-mean both timeseries 
        self.y1t = y1 - np.mean(y1) 
        self.y2t = y2 - np.mean(y2)   
        self.n = len(y1)
        self.nc=None
        self.method=method
        self.get_datapoints_2_correlate()
        
    def cor_series(self):
        corr = np.zeros(self.nc)
        t = np.zeros(self.nc)
        Neff = np.zeros(self.nc)
        for j in np.arange(0, self.nc):
            corr[j],t[j],Neff[j] = self.corr_one_roll(j)
        pval = ss.t.sf(np.abs(t), Neff)*2
        sig = np.zeros(len(t))*np.nan
        sig[abs(pval) < 0.05]=1
        
        #sig = signif(t)
        return corr,sig,t,Neff
    
    def get_datapoints_2_correlate(self):
        ## Looking for lagged correlations between two datasets
        ## Assumes y1 is leading
        ## Avoid end effects by using only first 2/3rds of timeseries
        if not self.nc and self.method!="Cyclic":
            self.nc = int(self.n/3.5)
        elif self.method == "Cyclic":
            self.nc = self.n
    
    def corr_one_roll(self,j):
        x1t=np.roll(self.y1t, j)
        x2t = self.y2t
        if self.method!="Cyclic":
            x1t = x1t[j:]
            x2t = x2t[j:]
        r1, tmp = ss.pearsonr(x1t[1:], np.roll(x1t, 1)[1:])
        r2, tmp = ss.pearsonr(x2t[1:], np.roll(x2t, 1)[1:])
        Neff = self.n*(1-r1*r2)/(1+r1*r2)
        corr = np.nanmean(x1t*x2t)/np.sqrt(np.nanmean(x1t**2)*np.nanmean(x2t**2))
        t = corr*Neff**(0.5)*(1-corr**2)**(-0.5)
        return corr,t,Neff
    
    def plot(self,data=data_dict, title='',ax=None,**kargs):
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
        if not data:
            data['x']= range(self.n)
            data['y1']= self.y1t
            data['y2']= self.y2t
            
        if 'y1' in data.keys() and 'y2' in data.keys():
            ax.plot(data['x'],data['y1'],'-k',**kargs)
            ax.plot(data['x'],data['y2'],'-', color='steelblue',**kargs)
        else:
            ax.plot(data['x'],data['y'],**kargs)
        ax.set_title(title)
        ax.grid()
        return ax
    
    def animate(self):
        data_series = collections.defaultdict(list)
        data = collections.defaultdict(list)
        corr = np.zeros(self.nc)*np.nan
        t = np.zeros(self.nc)*np.nan
        min_n = int(self.nc*0.1)
        for j in np.arange(0, self.nc):
            # Clear ouput
            clear_output(wait=True)
            fig, axs  = plt.subplots(2)
            fig.subplots_adjust(hspace=0.4)
            corr[j],t[j] = self.corr_one_roll(j)
            sig = signif(t)
            data_series['x'] = np.arange(len(self.y1t))
            data_series['y1'] = np.roll(self.y1t, j)
            data_series['y2'] = self.y2t
            ax = self.plot(data=data_series,title='De-mean signals'+' Lag:{0}'.format(j),ax=axs[0])
            data['x'] = np.arange(self.nc)
            data['y'] = corr
            ax = self.plot(data=data,title='Correlation',ax=axs[1])
            if j < min_n:
                ax.set_xlim(0,min_n)
            ax.set_ylim(-1.1,1.1)
            plt.show()