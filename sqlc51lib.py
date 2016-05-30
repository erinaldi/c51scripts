#!/usr/bin/env python

#Scalar current
import numpy as np
import matplotlib.pyplot as plt
import gvar as gv
import h5py
import yaml
import lsqfit
import collections
from tabulate import tabulate
from pandas import DataFrame
import os

### Define user information ###
def user_list():
    username = os.environ['USER']
    if username in ['cchang5', 'chang49']:
        name_flag = 'jason'
    return name_flag

### Fit function selection
def select_fitid(fit,params):
    if fit=='mres':
        return 1
    elif fit=='meson':
        n = params['decay_ward_fit']['nstates']
        if n==1:
            return 8
        elif n==2:
            return 9
        elif n==3:
            return 10

### create result submission
def make_result(fit,t):
    result = []
    for k in fit['pmean'][t].keys():
        result.append('"%s":%s' %(str(k),str(fit['pmean'][t][k])))
        result.append('"%s%s":%s' %('d',str(k),str(fit['psdev'][t][k])))
    result.append('"chi2":%s' %str(fit['chi2'][t]))
    result.append('"dof":%s' %str(fit['dof'][t]))
    result.append('"logGBF":%s' %str(fit['logGBF'][t]))
    result = "{%s}" %(','.join(result))
    return result

### read initial guess
def read_init(fit,t):
    init = dict()
    for k in fit['pmean'][t].keys():
        init[k] = fit['pmean'][t][k]
    return init

### filter prior selection
def baryon_priors(priors,basak,nstates):
    p = dict()
    for n in range(nstates):
        for k in priors[n+1].keys():
            k0 = k.split('_')[0]
            basak.append('E%s' %str(n))
            if k0 in basak:
                p[k] = priors[n+1][k]
    return p

### READ DATASET ###
def parity_avg(pos, neg, phase=1):
    neg = phase*np.roll(np.array(neg[:,::-1]), 1, axis=1)
    neg[:,0] = phase*neg[:,0]
    avg = 0.5*(pos + neg)
    return avg

def fold(meson, phase=1.0):
    meson_p = phase*np.roll(meson[:,::-1], 1, axis=1)
    meson_avg = 0.5*(meson + meson_p)
    return meson_avg

def make_gvars(data):
	data_gv = np.array(gv.dataset.avg_data(data))
	return data_gv

def ispin_avg(UU_up, UU_dn, DD_up=0, DD_dn=0, subset='twopt'):
    # this is for baryons of various current insertions
    # does isospin and spin averaging with correct phases
    if subset=='twopt':
        avg = 0.5*(UU_up + UU_dn)
    elif subset=='A3':
        # spin average
		savg_UU = 0.5*(UU_up - UU_dn)
		savg_DD = 0.5*(DD_up - DD_dn)
        # isospin average
		avg = savg_UU - savg_DD
    elif subset=='V4':
        # spin average
		savg_UU = 0.5*(UU_up + UU_dn)
		savg_DD = 0.5*(DD_up + DD_dn)
        # isospin average
		avg = savg_UU - savg_DD
    else: print "Need to define isospin + spin avg for current"
        
###    `. ---)..(
###      ||||(,o)
###      "`'" \__/

### ANALYSIS FUNCTIONS ###
class effective_plots:
    def __init__(self, T):
        self.T = T
    # derivative effective mass, for FH propagators
    def deriv_effective_mass(self, threept, twopt, tau=1):
    	dmeff = []
    	for t in range(len(twopt)-tau):
    		dmeff.append( (threept[(t+tau)%self.T]/twopt[(t+tau)%self.T] - threept[t]/twopt[t])/tau )
    	dmeff = np.array(dmeff)
    	return dmeff
    # effective mass
    def effective_mass(self, twopt, tau=2, style='log'):
        meff = []
        for t in range(len(twopt)):
            if style=='cosh':
        	    meff.append(np.arccosh((twopt[(t+tau)%len(twopt)]+twopt[t-tau])/(2*twopt[t])))
            elif style=='log':
                meff.append(np.log(twopt[t]/twopt[(t+tau)%len(twopt)])/tau)
            else: pass
        meff = np.array(meff)
        return meff
    # scaled two point
    def scaled_correlator(self, twopt, E0, phase=1.0):
        scaled2pt = []
        for t in range(len(twopt)):
            scaled2pt.append(twopt[t]/(np.exp(-E0*t)+phase*np.exp(-E0*(self.T-t))))
        scaled2pt = np.array(scaled2pt)
        return scaled2pt

def dict_of_tuple_to_gvar(dictionary):
    prior = dict()
    for name in dictionary.keys(): prior[name] = gv.gvar(dictionary[name][0], dictionary[name][1])
    return prior

def read_trange():
    fitparam = read_yaml()
    trange = fitparam['trange']
    return trange

def x_indep(tmin, tmax):
    x = np.arange(tmin, tmax+1)
    return x

def y_dep(x, y, sets=1):
    xh = x
    for s in range(1, sets):
        xh = np.append(xh,x+s*len(y)/sets)
    y = y[xh]
    #print y
    return y

# sets calculated
def fitscript_v2(trange,T,data,priors,fcn,init=None,basak=None):
    sets = len(data)/T
    print "sets:", sets
    pmean = []
    psdev = []
    post = []
    p0 = []
    prior = []
    tmintbl = []
    tmaxtbl = []
    chi2 = []
    dof = []
    lgbftbl = []
    rawoutput = []
    for tmin in range(trange['tmin'][0], trange['tmin'][1]+1):
        for tmax in range(trange['tmax'][0], trange['tmax'][1]+1):
            x = x_indep(tmin, tmax)
            y = y_dep(x, data, sets)
            if basak is not None:
                x = {'indep': x, 'basak': basak}
            else: pass
            fit = lsqfit.nonlinear_fit(data=(x,y),prior=priors,fcn=fcn,p0=init)
            pmean.append(fit.pmean)
            psdev.append(fit.psdev)
            post.append(fit.p)
            p0.append(fit.p0)
            prior.append(fit.prior)
            tmintbl.append(tmin)
            tmaxtbl.append(tmax)
            chi2.append(fit.chi2)
            dof.append(fit.dof)
            lgbftbl.append(fit.logGBF)
            rawoutput.append(fit)
    fittbl = dict()
    fittbl['tmin'] = tmintbl
    fittbl['tmax'] = tmaxtbl
    fittbl['pmean'] = pmean
    fittbl['psdev'] = psdev
    fittbl['post'] = post
    fittbl['p0'] = p0
    fittbl['prior'] = prior
    fittbl['chi2'] = chi2
    fittbl['dof'] = dof
    fittbl['logGBF'] = lgbftbl
    fittbl['rawoutput'] = rawoutput
    return fittbl

def tabulate_result(fit_proc, parameters):
    tbl = collections.OrderedDict()
    try:
        tbl['nstates'] = fit_proc.nstates
    except: pass
    tbl['tmin'] = fit_proc.tmin
    tbl['tmax'] = fit_proc.tmax
    for p in parameters:
        tbl[p] = fit_proc.read_boot0(p) #gv.gvar(fit_proc.read_boot0(p), fit_proc.read_boot0_sdev(p))
        tbl[p+' err'] = fit_proc.read_boot0_sdev(p)
    tbl['chi2/dof'] = fit_proc.chi2dof
    #tbl['logGBF'] = fit_proc.logGBF
    #tbl['normBF'] = fit_proc.normbayesfactor
    #tbl['logpost'] = fit_proc.logposterior
    #tbl['normpost'] = fit_proc.normposterior 
    return tabulate(tbl, headers='keys')

#FIT FUNCTIONS
class fit_function():
    def __init__(self, T, nstates=1):
        self.T = T
        self.nstates = nstates
    # derivative of effective mass for FH propagators
    def dmeff_fitfcn(self, t, p):
        fitfcn = p['ME'] + (p['C']*t+p['D'])*np.exp(-p['E']*t)
        return fitfcn
    # two point smear smear source sink
    def twopt_fitfcn_ss(self, t, p):
        En = p['E0']
        fitfcn = p['Z0_s']**2 * (np.exp(-1*En*t) + np.exp(-1*En*(self.T-t))) 
        for n in range(1, self.nstates):
            En += np.exp(p['E'+str(n)])
            fitfcn += p['Z'+str(n)+'_s']**2 * (np.exp(-1*En*t) + np.exp(-1*En*(self.T-t)))
        return fitfcn
    # two point point smear source sink
    def twopt_fitfcn_ps(self, t, p):
        En = p['E0']
        En_p = 0
        fitfcn = p['Z0_p']*p['Z0_s'] * (np.exp(-1*En*t) + np.exp(-1*En*(self.T-t)) )
        for n in range(1, self.nstates):
            En += np.exp(p['E'+str(n)])
            fitfcn += p['Z'+str(n)+'_p']*p['Z'+str(n)+'_s'] * (np.exp(-1*En*t) + np.exp(-1*En*(self.T-t)) )
        return fitfcn
    def twopt_fitfcn_pp(self, t, p):
        En = p['E0']
        fitfcn = p['Z0_p']**2 * (np.exp(-1*En*t) + np.exp(-1*En*(self.T-t)))
        for n in range(1, self.nstates):
            En += np.exp(p['E'+str(n)])
            fitfcn += p['Z'+str(n)+'_p']**2 * (np.exp(-1*En*t) + np.exp(-1*En*(self.T-t)))
        return fitfcn
    # baryon two points
    def twopt_baryon_ss(self,t,p,b):
        En = p['E0']
        fitfcn = p['%s_Z0s' %b]**2 * np.exp(-1*En*t)
        for n in range(1, self.nstates):
            En += np.exp(p['E'+str(n)])
            fitfcn += p['%s_Z%ss' %(b,str(n))]**2 * np.exp(-1*En*t)
        return fitfcn
    def twopt_baryon_ps(self,t,p,b):
        En = p['E0']
        fitfcn = p['%s_Z0s' %b]*p['%s_Z0p' %b] * np.exp(-1*En*t)
        for n in range(1, self.nstates):
            En += np.exp(p['E'+str(n)])
            fitfcn += p['%s_Z%ss' %(b,str(n))]*p['%s_Z%sp' %(b,str(n))] * np.exp(-1*En*t)
        return fitfcn
    def twopt_fitfcn_phiqq(self, t, p):
        En = p['E0']
        fitfcn = p['A0'] * (np.exp(-1*En*t) + np.exp(-1*En*(self.T-t)) + np.exp(-1*En*(self.T+t)))
        return fitfcn
    # Combined fitters
    # two point ss and ps simultaneous fit 
    def twopt_fitfcn_ss_ps(self, t, p):
        fitfcn_ss = self.twopt_fitfcn_ss(t, p)
        fitfcn_ps = self.twopt_fitfcn_ps(t, p)
        fitfcn = np.concatenate((fitfcn_ss, fitfcn_ps))
        return fitfcn
    def twopt_fitfcn_ss_pp(self,t,p):
        fitfcn_ss = self.twopt_fitfcn_ss(t,p)
        fitfcn_pp = self.twopt_fitfcn_pp(t,p)
        fitfcn = np.concatenate((fitfcn_ss,fitfcn_pp))
        return fitfcn
    # baryon two point simultaneous fit
    def twopt_baryon_ss_ps(self,t,p):
        ssl = []
        psl = []
        for b in t['basak']:
            ssl.append(self.twopt_baryon_ss(t['indep'],p,b))
            psl.append(self.twopt_baryon_ps(t['indep'],p,b))
        ss = ssl[0]
        ps = psl[0]
        for i in range(len(ssl)-1):
            ss = np.concatenate((ss,ssl[i+1]))
            ps = np.concatenate((ps,psl[i+1]))
        fitfcn = np.concatenate((ss,ps))
        return fitfcn
    # axial_ll and two point ss simultaneous fit
    def axial_twoptss_fitfcn(self, t, p):
        fitfcn_ss = self.twopt_fitfcn_ss(t, p)
        fitfcn_axial = self.decay_axial(t, p)
        fitfcn = np.concatenate((fitfcn_axial, fitfcn_ss))
        return fitfcn
    #Decay constant fit functions
    # decay constant from two point correlator
    def decay_constant(self, t, p):
        fitfcn = p['Ass_0']*np.exp(-p['E0']*t) + p['Aps_0']*np.exp(-p['E0']*t)
        return fitfcn
    # decay constant from axial_ll
    def decay_axial(self, t, p):
        En = p['E0']
        fitfcn = p['Z0_s']*p['F0']*(np.exp(-1*En*t) - np.exp(-1*En*(self.T-t)))
        for n in range(1, self.nstates):
            En += np.exp(p['E'+str(n)])
            fitfcn += p['F'+str(n)]*(np.exp(-1*En*t) - np.exp(-1*En*(self.T-t)))
        return fitfcn
    # mres
    def mres_fitfcn(self, x, p):
        fitfcn = x*p['mres']/x
        return fitfcn
###  ....._____.......
###      /     \/|
###      \o__  /\|
###          \|

### PLOT FUNCTIONS ###
def scatter_plot(x, data_gv, title='default title', xlabel='x', ylabel='y', xlim=[None,None], ylim=[None,None], grid_flg=True):
    y = np.array([dat.mean for dat in data_gv])
    e = np.array([dat.sdev for dat in data_gv])
    plt.figure()
    plt.errorbar(x, y, yerr=e)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    plt.grid(grid_flg)
    plt.draw()
    return 0

def stability_plot(fittbl, key, title=''):
    # tmin stability plot
    if fittbl['tmin'][-1]-fittbl['tmin'][0] > 0:
        if fittbl['tmax'][-1]-fittbl['tmax'][0] > 0:
            output = []
            for t in range(len(fittbl['tmin'])):
                output.append((fittbl['tmin'][t], fittbl['tmax'][t], fittbl['post'][t][key]))
            dtype = [('tmin', int), ('tmax', int), ('post', gv._gvarcore.GVar)]
            output = np.array(output, dtype=dtype)
            output = np.sort(output, order='tmax')
            setwidth = fittbl['tmin'][-1]-fittbl['tmin'][0]+1
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            for subset in range(len(output)/setwidth):
                pltdata = output[setwidth*subset:setwidth*(subset+1)]
                x = [pltdata[i][0] for i in range(len(pltdata))]
                y = [pltdata[i][2] for i in range(len(pltdata))]
                y_plt = np.array([dat.mean for dat in y])
                e_plt = np.array([dat.sdev for dat in y])
                ax1.errorbar(x, y_plt, e_plt, label='tmax: '+str(pltdata[0][1]))
            plt.title(title+' tmin stability plot')
            plt.xlabel('tmin')
            plt.ylabel(key)
            plt.xlim(x[0]-0.5, x[-1]+0.5)
            plt.legend()
        else:
            x = fittbl['tmin']
            y = np.array([data[key] for data in fittbl['post']])
            scatter_plot(x, y, title+' tmin stability plot', 'tmin (tmax='+str(fittbl['tmax'][0])+')', key, xlim=[x[0]-0.5,x[-1]+0.5])
    # tmax stability plot
    if fittbl['tmax'][-1]-fittbl['tmax'][0] > 0:
        if fittbl['tmin'][-1]-fittbl['tmin'][0] > 0:
            output = []
            for t in range(len(fittbl['tmin'])):
                output.append((fittbl['tmin'][t], fittbl['tmax'][t], fittbl['post'][t][key]))
            dtype = [('tmin', int), ('tmax', int), ('post', gv._gvarcore.GVar)]
            output = np.array(output, dtype=dtype)
            output = np.sort(output, order='tmin')
            setwidth = fittbl['tmax'][-1]-fittbl['tmax'][0]+1
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            for subset in range(len(output)/setwidth):
                pltdata = output[setwidth*subset:setwidth*(subset+1)]
                x = [pltdata[i][1] for i in range(len(pltdata))]
                y = [pltdata[i][2] for i in range(len(pltdata))]
                y_plt = np.array([dat.mean for dat in y])
                e_plt = np.array([dat.sdev for dat in y])
                ax1.errorbar(x, y_plt, e_plt, label='tmin: '+str(pltdata[0][0]))
            plt.title(title+' tmax stability plot')
            plt.xlabel('tmax')
            plt.ylabel(key)
            plt.xlim(x[0]-0.5, x[-1]+0.5)
            plt.legend()
        else:
            x = fittbl['tmax']
            y = np.array([data[key] for data in fittbl['post']])
            scatter_plot(x, y, title+' tmax stability plot', 'tmax (tmin='+str(fittbl['tmin'][0])+')', key, xlim=[x[0]-0.5,x[-1]+0.5])
    else: pass #print key,':',fittbl['post'][0][key]
    return 0

def nstate_stability_plot(fittbl, key, title=''):
    x = fittbl['nstates']
    y = np.array([data[key] for data in fittbl['post']])
    scatter_plot(x, y, title+' nstate stability plot', 'nstates ([tmin, tmax]=['+str(fittbl['tmin'][0])+', '+str(fittbl['tmax'][0])+'])', key, xlim=[x[0]-0.5,x[-1]+0.5])
    return 0

def histogram_plot(fittbl, key=None, xlabel=''):
    plt.figure()
    if key == None:
        bs_mean = fittbl
    else:
        bs_mean = np.array([fittbl[i]['post'][j][key].mean for i in range(len(fittbl)) for j in range(len(fittbl[i]['post']))])
    n, bins, patches = plt.hist(bs_mean, 50, facecolor='green')
    x = np.delete(bins, -1)
    plt.plot(x, n)
    plt.xlabel(xlabel)
    plt.ylabel('counts')
    plt.draw()
    return 0

def find_yrange(data, pltxmin, pltxmax):
	pltymin = np.min(np.array([dat.mean for dat in data[pltxmin:pltxmax]]))
	pltymax = np.max(np.array([dat.mean for dat in data[pltxmin:pltxmax]]))
	pltyerr = np.max(np.array([dat.sdev for dat in data[pltxmin:pltxmax]]))
	pltymin = np.min([pltymin-pltyerr, pltymin+pltyerr])
	pltymax = np.max([pltymax-pltyerr, pltymax+pltyerr])
	pltyrng = [pltymin, pltymax]
	return pltyrng

###      ()()       ^ ^ 
###      (_ _)    =(o o)=
###      (u u)o    (m m)~~

### BEGIN MAIN ###
if __name__=='__main__':
    print "c51 library"
