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
import tables as h5

### Define user information ###
def user_list():
    username = os.environ['USER']
    if username in ['cchang5', 'chang49']:
        name_flag = 'jason'
    return name_flag

### Fit function selection
def select_fitid(fit,params=None,nstates=None,basak=None,tau=None):
    if fit=='mres': return 1
    elif fit=='meson':
        n = params['decay_ward_fit']['nstates']
        if n==1: return 8
        elif n==2: return 9
        elif n==3: return 10
    elif fit=='baryon':
        if nstates==1:
            if len(basak)==1: return 19
            else: return 23
        if nstates==2:
            if len(basak)==1: return 20
            else: return 24
        if nstates==3:
            if len(basak)==1: return 21
            else: return 25
        if nstates==4:
            if len(basak)==1: return 22
            else: return 26
        if nstates==5:
            if len(basak)==1: return 29
            else: return 27
        if nstates==6:
            if len(basak)==1: return 30
            else: return 28
    elif fit=='fhbaryon':
        if nstates==1:
            if tau==1: return 31
            elif tau==2: return 37
            else: print "Define fit function"
        if nstates==2:
            if tau==1: return 32
            elif tau==2: return 38
            else: print "Define fit function"
        if nstates==3:
            if tau==1: return 33
            elif tau==2: return 38
            else: print "Define fit function"
        if nstates==4:
            if tau==1: return 34
            elif tau==2: return 39
            else: print "Define fit function"
        if nstates==5:
            if tau==1: return 35
            elif tau==2: return 40
            else: print "Define fit function"
        if nstates==6:
            if tau==1: return 36
            elif tau==2: return 41
            else: print "Define fit function"

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
    bsplit = []
    for b in basak:
        bsplit.append(b[:2])
        bsplit.append(b[2:])
    for n in range(nstates):
        for k in priors[n+1].keys():
            k0 = k.split('_')[0]
            bsplit.append('E%s' %str(n))
            if k0 in bsplit:
                p[k] = priors[n+1][k]
    return p

def meson_priors(priors,nstates):
    p = dict()
    for n in range(nstates):
        for k in priors[n+1].keys():
            p[k] = priors[n+1][k]
    return p

def baryon_initpriors(priors,basak,nstates):
    p = dict()
    bsplit = []
    for b in basak:
        bsplit.append(b[:2])
        bsplit.append(b[2:])
    for n in range(nstates):
        for k in priors[n+1].keys():
            k0 = k.split('_')[0]
            bsplit.append('E%s' %str(n))
            if k0 in bsplit:
                p[k] = priors[n+1][k][0]
    return p

def fhbaryon_priors(priors,fhpriors,basak,nstates,fhstates):
    if fhstates < nstates:
        p = baryon_priors(priors,basak,nstates)
    else:
        p = baryon_priors(priors,basak,fhstates)
    for n in range(fhstates):
        for k in fhpriors[n+1].keys():
            p[k] = fhpriors[n+1][k]
    # contact priors
    for b in basak:
        bsrc = b[:2]
        bsnk = b[2:]
        for n in range(fhstates):
            for k in fhpriors['c%s' %n].keys():
                prior_bsrc = k.split('_')[1]
                prior_bsnk = k.split('_')[0]
                if bsrc == prior_bsrc and bsnk == prior_bsnk:
                    p[k] = fhpriors['c%s' %n][k]
                else: pass
    return p

def fhbaryon_priors_v2(priors,fhpriors,gVpriors,basak,nstates,fhstates,gVstates):
    maxstates = max(nstates,fhstates,gVstates)
    p = baryon_priors(priors,basak,maxstates)
    for n in range(fhstates):
        for k in fhpriors[n+1].keys():
            p[k] = fhpriors[n+1][k]
    for n in range(gVstates):
        for k in gVpriors[n+1].keys():
            p[k] = gVpriors[n+1][k]
    # contact priors
    for b in basak:
        bsrc = b[:2]
        bsnk = b[2:]
        for n in range(fhstates):
            for k in fhpriors['c%s' %n].keys():
                prior_bsrc = k.split('_')[1]
                prior_bsnk = k.split('_')[0]
                if bsrc == prior_bsrc and bsnk == prior_bsnk:
                    p[k] = fhpriors['c%s' %n][k]
                else: pass
        for n in range(gVstates):
            for k in gVpriors['c%s' %n].keys():
                prior_bsrc = k.split('_')[1]
                prior_bsnk = k.split('_')[0]
                if bsrc == prior_bsrc and bsnk == prior_bsnk:
                    p[k] = gVpriors['c%s' %n][k]
                else: pass
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
    return avg
        
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
            scaled2pt.append(twopt[t]*2.*E0/(np.exp(-E0*t)))
        scaled2pt = np.array(scaled2pt)
        return scaled2pt
    def scaled_correlator_v2(self, twopt, E0, phase=1.0):
        scaled2pt = []
        for t in range(len(twopt)):
            scaled2pt.append(twopt[t]/(np.exp(-E0*t)))
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
    return xh, y

def y_dep_v2(x, y, sets):
    print "parsing for two + three point fit"
    x2 = x[0]
    fhx = x[1]
    subsets = sets/2
    x = x2
    for s in range(1, subsets):
        x = np.append(x, x2+s*len(y)/sets)
    for s in range(subsets):
        x = np.append(x, fhx+(subsets+s)*len(y)/sets)
    y = y[x]
    return x, y

def y_dep_v3(x, y, sets):
    #print "parsing for two + three + gV point fit"
    x2 = x[0]
    fhx = x[1]
    gVx = x[2]
    subsets = sets/3
    x = x2
    for s in range(1, subsets):
        x = np.append(x, x2+s*len(y)/sets)
    for s in range(subsets):
        x = np.append(x, fhx+(subsets+s)*len(y)/sets)
    for s in range(subsets):
        x = np.append(x, gVx+(2*subsets+s)*len(y)/sets)
    y = y[x]
    return x, y

def y_dep_axial(x, y, sets):
    print "parsing for two + axial fit"
    x2 = x[0]
    ax = x[1]
    twosub = sets*2/3
    axsub = sets/3
    x = x2
    for s in range(1, twosub):
        x = np.append(x, x2+s*len(y)/sets)
    for s in range(axsub):
        x = np.append(x, ax+(twosub+s)*len(y)/sets)
    y = y[x]
    return x, y

def chi2freq(chi2,prior,post):
    chi2f = chi2
    try:
        for k in prior.keys():
            chi2f += -1*( (prior[k].mean-post[k].mean)/prior[k].sdev )**2
    except:
        #print "doing unconstrained fit. chi^2 is kosher"
        pass
    return chi2f

# sets calculated
def fitscript_v2(trange,T,data,priors,fcn,init=None,basak=None,bayes=True):
    sets = len(data)/T
    #print "sets:", sets
    pmean = []
    psdev = []
    post = []
    p0 = []
    prior = []
    tmintbl = []
    tmaxtbl = []
    chi2 = []
    dof = []
    Q = []
    lgbftbl = []
    rawoutput = []
    for tmin in range(trange['tmin'][0], trange['tmin'][1]+1):
        for tmax in range(trange['tmax'][0], trange['tmax'][1]+1):
            x = x_indep(tmin, tmax)
            xlist, y = y_dep(x, data, sets)
            if basak is not None:
                x = {'indep': x, 'basak': basak}
            else: pass
            if bayes:
                fit = lsqfit.nonlinear_fit(data=(x,y),prior=priors,fcn=fcn,maxit=1000000) #,svdcut=1E-3)
            else:
                fit = lsqfit.nonlinear_fit(data=(x,y),fcn=fcn,p0=init,maxit=1000000) #,svdcut=1E-3)
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
            Q.append(fit.Q)
            rawoutput.append(fit)
    #fcnname = str(fcn.__name__)
    #fitline = fcn(x,fit.p)
    #print "%s_%s_t, %s_%s_y, +-, %s_%s_fit, +-" %(fcnname, basak[0], fcnname, basak[0], fcnname, basak[0])
    #for i in range(len(xlist)):
    #    print xlist[i], ',', y[i].mean, ',', y[i].sdev, ',', fitline[i].mean, ',', fitline[i].sdev
    #print '======'
    #print fcn.__name__, basak
    #print fit
    #print '======'
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
    fittbl['Q'] = Q
    fittbl['rawoutput'] = rawoutput
    return fittbl

def fitscript_v3(trange,fhtrange,T,data,priors,fcn,init=None,basak=None,axial=False,bayes=True):
    sets = len(data)/T
    #print "sets:", sets
    pmean = []
    psdev = []
    post = []
    p0 = []
    prior = []
    tmintbl = []
    tmaxtbl = []
    fhtmintbl = []
    fhtmaxtbl = []
    chi2 = []
    chi2f = []
    dof = []
    lgbftbl = []
    Q = []
    rawoutput = []
    for tmin in range(trange['tmin'][0], trange['tmin'][1]+1):
        for tmax in range(trange['tmax'][0], trange['tmax'][1]+1):
            for fhtmin in range(fhtrange['tmin'][0], fhtrange['tmin'][1]+1):
                for fhtmax in range(fhtrange['tmax'][0], fhtrange['tmax'][1]+1):
                    x2 = x_indep(tmin, tmax)
                    fhx = x_indep(fhtmin, fhtmax)
                    x = [x2,fhx]
                    if axial:
                        xlist, y = y_dep_axial(x, data, sets)
                    else:
                        xlist, y = y_dep_v2(x, data, sets)
                    if basak is not None:
                        x = {'indep': x, 'basak': basak}
                        #print gv.evalcov(y)
                        #print [(i.sdev)**2 for i in y]
                    else: pass
                    # nonlinear fit
                    if bayes:
                        fit = lsqfit.nonlinear_fit(data=(x,y),prior=priors,p0=init,fcn=fcn,maxit=1000000) #,svdcut=1E-2)
                    else:
                        fit = lsqfit.nonlinear_fit(data=(x,y),fcn=fcn,p0=init,maxit=1000000) #,svdcut=8.95E-4)
                    # empirical bayes
                    #def fitargs(z, data=(x,y), prior=priors,fcn=fcn,p0=init):
                    #    z = np.exp(z)
                    #    #prior['G1_Z0s'] = gv.gvar(1.2E-5, 6E-6 * z[0])
                    #    #prior['G1_Z0p'] = gv.gvar(2E-3, 1E-3 * z[1])
                    #    #prior['E0'] = gv.gvar(0.45, 0.04 * z[2])
                    #    for i, k in enumerate(prior.keys()):
                    #        prior[k] = gv.gvar(prior[k].mean, prior[k].sdev * z[i])
                    #    return dict(data=(x,y),prior=prior,fcn=fcn,p0=init)
                    #z0 = np.zeros(len(priors))
                    #fit, z = lsqfit.empbayes_fit(z0, fitargs, maxit=1000000) #,svdcut=1E-5)
                    #print fhtmin, fhtmax, fit.p['gA00'].mean, fit.p['gA00'].sdev, fit.p['E0'].mean, fit.p['E0'].sdev, fit.chi2/fit.dof, fit.Q, fit.logGBF
                    print fit
                    pmean.append(fit.pmean)
                    psdev.append(fit.psdev)
                    post.append(fit.p)
                    p0.append(fit.p0)
                    prior.append(fit.prior)
                    tmintbl.append(tmin)
                    tmaxtbl.append(tmax)
                    fhtmintbl.append(fhtmin)
                    fhtmaxtbl.append(fhtmax)
                    chi2.append(fit.chi2)
                    dof.append(fit.dof)
                    lgbftbl.append(fit.logGBF)
                    Q.append(fit.Q)
                    rawoutput.append(fit)
                    chi2f.append(chi2freq(fit.chi2,fit.prior,fit.p))
    ## print correlation matrix of data
    #corr = gv.evalcorr(fit.y)
    #string = ''
    #for i in range(len(corr)):
    #    for j in range(len(corr)):
    #        string +='%s,%s,%s\n' %(str(i),str(j),str(corr[i,j]))
    #f = open('./temp.csv','w')
    #f.write(string)
    #f.flush()
    #f.close()
    ## end corr
    #fitline = fcn(x,fit.p)
    #print "t, y, +-, fit, +-"
    #for i in range(len(xlist)):
    #    print xlist[i], ',', y[i].mean, ',', y[i].sdev, ',', fitline[i].mean, ',', fitline[i].sdev
    #print fit
    #print '======'
    #print fcn.__name__, basak
    #print fit
    #print '======'
    fittbl = dict()
    fittbl['tmin'] = tmintbl
    fittbl['tmax'] = tmaxtbl
    fittbl['fhtmin'] = fhtmintbl
    fittbl['fhtmax'] = fhtmaxtbl
    fittbl['pmean'] = pmean
    fittbl['psdev'] = psdev
    fittbl['post'] = post
    fittbl['p0'] = p0
    fittbl['prior'] = prior
    fittbl['chi2'] = chi2
    fittbl['chi2f'] = chi2f
    fittbl['dof'] = dof
    fittbl['logGBF'] = lgbftbl
    fittbl['Q'] = Q
    fittbl['rawoutput'] = rawoutput
    return fittbl

def fitscript_v4(trange,fhtrange,gVtrange,T,data,priors,fcn,init=None,basak=None,axial=False,bayes=True):
    sets = len(data)/T
    #print "sets:", sets
    pmean = []
    psdev = []
    post = []
    p0 = []
    prior = []
    tmintbl = []
    tmaxtbl = []
    fhtmintbl = []
    fhtmaxtbl = []
    gVtmintbl = []
    gVtmaxtbl = []
    chi2 = []
    chi2f = []
    dof = []
    lgbftbl = []
    Q = []
    rawoutput = []
    for tmin in range(trange['tmin'][0], trange['tmin'][1]+1):
        for tmax in range(trange['tmax'][0], trange['tmax'][1]+1):
            for fhtmin in range(fhtrange['tmin'][0], fhtrange['tmin'][1]+1):
                for fhtmax in range(fhtrange['tmax'][0], fhtrange['tmax'][1]+1):
                    for gVtmin in range(gVtrange['tmin'][0], gVtrange['tmin'][1]+1):
                        for gVtmax in range(gVtrange['tmax'][0], gVtrange['tmax'][1]+1):
                            x2 = x_indep(tmin, tmax)
                            fhx = x_indep(fhtmin, fhtmax)
                            gVx = x_indep(gVtmin, gVtmax)
                            x = [x2,fhx,gVx]
                            xlist, y = y_dep_v3(x, data, sets)
                            if basak is not None:
                                x = {'indep': x, 'basak': basak}
                                #print gv.evalcov(y)
                                #print [(i.sdev)**2 for i in y]
                            else: pass
                            # nonlinear fit
                            if bayes:
                                fit = lsqfit.nonlinear_fit(data=(x,y),prior=priors,p0=init,fcn=fcn,maxit=1000000) #,svdcut=1E-5)
                            else:
                                fit = lsqfit.nonlinear_fit(data=(x,y),fcn=fcn,p0=init,maxit=1000000) #,svdcut=8.95E-4)
                            #print fit.p
                            pmean.append(fit.pmean)
                            psdev.append(fit.psdev)
                            post.append(fit.p)
                            p0.append(fit.p0)
                            prior.append(fit.prior)
                            tmintbl.append(tmin)
                            tmaxtbl.append(tmax)
                            fhtmintbl.append(fhtmin)
                            fhtmaxtbl.append(fhtmax)
                            gVtmintbl.append(gVtmin)
                            gVtmaxtbl.append(gVtmax)
                            chi2.append(fit.chi2)
                            dof.append(fit.dof)
                            lgbftbl.append(fit.logGBF)
                            Q.append(fit.Q)
                            rawoutput.append(fit)
                            chi2f.append(chi2freq(fit.chi2,fit.prior,fit.p))
    fittbl = dict()
    fittbl['tmin'] = tmintbl
    fittbl['tmax'] = tmaxtbl
    fittbl['fhtmin'] = fhtmintbl
    fittbl['fhtmax'] = fhtmaxtbl
    fittbl['gVtmin'] = gVtmintbl
    fittbl['gVtmax'] = gVtmaxtbl
    fittbl['pmean'] = pmean
    fittbl['psdev'] = psdev
    fittbl['post'] = post
    fittbl['p0'] = p0
    fittbl['prior'] = prior
    fittbl['chi2'] = chi2
    fittbl['chi2f'] = chi2f
    fittbl['dof'] = dof
    fittbl['logGBF'] = lgbftbl
    fittbl['Q'] = Q
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
    def __init__(self, T, nstates=1, fhstates=1, gVstates=1, tau=1):
        self.T = T
        self.nstates = nstates
        self.fhstates = fhstates
        self.gVstates = gVstates
        self.tau = tau
    # rederived FH fit function and 2pt ########################################################
    # Z = Z/sqrt(2E0)   G = G/2E0
    def dwhisq_dm(self,t,p,b,snk,src,gV=False):
        twopt = self.dwhisq_twopt(t,p,b,snk,src)
        fh = self.dwhisq_fh(t,p,b,snk,src,gV)
        twopt1 = self.dwhisq_twopt(t+1,p,b,snk,src)
        fh1 = self.dwhisq_fh(t+1,p,b,snk,src,gV)
        r = fh/twopt
        r1 = fh1/twopt1
        dm = r1-r
        return dm
    def dwhisq_twopt(self,t,p,b,snk,src):
        bsrc = b[:2]
        bsnk = b[2:]
        C = 0
        for n in range(self.nstates):
            En = self.E(p,n)
            Zsrc = p['%s_Z%s%s' %(bsrc,n,src)]
            Zsnk = p['%s_Z%s%s' %(bsnk,n,snk)]
            C += Zsnk*Zsrc*np.exp(-En*t)
        return C
    def dwhisq_twopt_osc(self,t,p,snk,src):
        C = 0
        for n in range(self.nstates):
            En = self.E_osc(p,n)
            Zsrc = p['Z%s_%s' %(n,src)]
            Zsnk = p['Z%s_%s' %(n,snk)]
            C += Zsnk*Zsrc*(-1)**(n*t)*(np.exp(-1*En*t) + np.exp(-1*En*(self.T-t)))
        return C
    def mixed_twopt_osc(self,t,p):
        C = 0
        for n in range(self.nstates):
            En = self.E_osc(p,n)
            An = p['A%s' %n]
            C += An*(-1)**(n*t)*(np.exp(-1*En*t) + np.exp(-1*En*(self.T-t)))
        return C
    def mixed_twopt_noosc(self,t,p):
        C = 0
        for n in range(self.nstates):
            En = self.E(p,n)
            An = p['A%s' %n]
            C += An*(np.exp(-1*En*t) + np.exp(-1*En*(self.T-t)))
        return C
    def E_osc(self,p,n):
        En = p['E0']
        if n%2 == 0:
            for m in range(0,n,2):
                En += np.exp(p['E%s' %(m+2)])
            return En
        if n%2 == 1:
            En += np.exp(p['E1'])
            for m in range(1,n,2):
                En += np.exp(p['E%s' %(m+2)])
            return En
    def dwhisq_fh(self,t,p,b,snk,src,gV):
        bsrc = b[:2]
        bsnk = b[2:]
        M = 0
        if gV:
            states = self.gVstates
        else:
            states = self.fhstates
        for n in range(states):
            for m in range(states):
                En = self.E(p,n)
                Em = self.E(p,m)
                Gmn = self.G(p,b,m,n,gV)
                Zn = p['%s_Z%s%s' %(bsrc,n,src)]
                Zm = p['%s_Z%s%s' %(bsnk,m,snk)]
                if n == m:
                    M += Zm*Gmn*Zn*(t-1)*np.exp(-En*t)
                else:
                    Dnm = En-Em
                    M += Zm*Gmn*Zn*(np.exp(-0.5*Dnm)*np.exp(-Em*t)-np.exp(0.5*Dnm)*np.exp(-En*t))/(np.exp(0.5*Dnm)-np.exp(-0.5*Dnm))
        # Dn parameterizes contact terms + current outside of src snk
        for n in range(states):
            Dn = self.Dn(p,bsnk,bsrc,n,snk,src,gV)
            En = self.E(p,n)
            M += Dn*np.exp(-En*t)
        return M
    def dwhisq_twopt_ss_ps(self,t,p):
        ssl = []
        psl = []
        for b in t['basak']:
            ssl.append(self.dwhisq_twopt(t['indep'],p,b,'s','s'))
            psl.append(self.dwhisq_twopt(t['indep'],p,b,'p','s'))
        ss = ssl[0]
        ps = psl[0]
        for i in range(len(ssl)-1):
            ss = np.concatenate((ss,ssl[i+1]))
            ps = np.concatenate((ps,psl[i+1]))
        fitfcn = np.concatenate((ss,ps))
        return fitfcn
    def dwhisq_fh_ss_ps(self,t,p):
        x2 = t['indep'][0]
        fhx = t['indep'][1]
        ssl = []
        psl = []
        for b in t['basak']:
            ssl.append(self.dwhisq_twopt(x2,p,b,'s','s'))
            psl.append(self.dwhisq_twopt(x2,p,b,'p','s'))
        ss = ssl[0]
        ps = psl[0]
        for i in range(len(ssl)-1):
            ss = np.concatenate((ss,ssl[i+1]))
            ps = np.concatenate((ps,psl[i+1]))
        fhssl = []
        fhpsl = []
        for b in t['basak']:
            fhssl.append(self.dwhisq_fh(fhx,p,b,'s','s'))
            fhpsl.append(self.dwhisq_fh(fhx,p,b,'p','s'))
        fhss = fhssl[0]
        fhps = fhpsl[0]
        for i in range(1,len(fhssl)):
            fhss = np.concatenate((fhss,fhssl[i]))
            fhps = np.concatenate((fhps,fhpsl[i]))
        fitfcn = np.concatenate((ss,ps,fhss,fhps))
        return fitfcn
    def dwhisq_dm_ss_ps(self,t,p):
        x2 = t['indep'][0]
        fhx = t['indep'][1]
        ssl = []
        psl = []
        for b in t['basak']:
            ssl.append(self.dwhisq_twopt(x2,p,b,'s','s'))
            psl.append(self.dwhisq_twopt(x2,p,b,'p','s'))
        ss = ssl[0]
        ps = psl[0]
        for i in range(len(ssl)-1):
            ss = np.concatenate((ss,ssl[i+1]))
            ps = np.concatenate((ps,psl[i+1]))
        dmssl = []
        dmpsl = []
        for b in t['basak']:
            dmssl.append(self.dwhisq_dm(fhx,p,b,'s','s'))
            dmpsl.append(self.dwhisq_dm(fhx,p,b,'p','s'))
        dmss = dmssl[0]
        dmps = dmpsl[0]
        for i in range(1,len(dmssl)):
            dmss = np.concatenate((dmss,dmssl[i]))
            dmps = np.concatenate((dmps,dmpsl[i]))
        fitfcn = np.concatenate((ss,ps,dmss,dmps))
        return fitfcn
    def dwhisq_dm_gVdm_ss_ps(self,t,p):
        x2 = t['indep'][0]
        fhx = t['indep'][1]
        gVx = t['indep'][2]
        # two point
        ssl = []
        psl = []
        for b in t['basak']:
            ssl.append(self.dwhisq_twopt(x2,p,b,'s','s'))
            psl.append(self.dwhisq_twopt(x2,p,b,'p','s'))
        ss = ssl[0]
        ps = psl[0]
        for i in range(len(ssl)-1):
            ss = np.concatenate((ss,ssl[i+1]))
            ps = np.concatenate((ps,psl[i+1]))
        # gA
        dmssl = []
        dmpsl = []
        for b in t['basak']:
            dmssl.append(self.dwhisq_dm(fhx,p,b,'s','s'))
            dmpsl.append(self.dwhisq_dm(fhx,p,b,'p','s'))
        dmss = dmssl[0]
        dmps = dmpsl[0]
        for i in range(1,len(dmssl)):
            dmss = np.concatenate((dmss,dmssl[i]))
            dmps = np.concatenate((dmps,dmpsl[i]))
        # gV
        gVdmssl = []
        gVdmpsl = []
        for b in t['basak']:
            gVdmssl.append(self.dwhisq_dm(gVx,p,b,'s','s',True))
            gVdmpsl.append(self.dwhisq_dm(gVx,p,b,'p','s',True))
        gVdmss = gVdmssl[0]
        gVdmps = gVdmpsl[0]
        for i in range(1,len(dmssl)):
            gVdmss = np.concatenate((gVdmss,gVdmssl[i]))
            gVdmps = np.concatenate((gVdmps,gVdmpsl[i]))
        # concatenate
        fitfcn = np.concatenate((ss,ps,dmss,dmps,gVdmss,gVdmps))
        return fitfcn
    def dwhisq_axial(self,t,p):
        C = 0
        for n in range(self.nstates):
            En = self.E(p,n)
            Zs = p['Z%s_s' %(n)]
            Fn = p['F%s' %(n)]
            C += Fn*Zs*(np.exp(-1*En*t) - np.exp(-1*En*(self.T-t)))
        return C
    def dwhisq_axial_osc(self,t,p):
        C = 0
        for n in range(self.nstates):
            En = self.E(p,n)
            Zs = p['Z%s_s' %(n)]
            Fn = p['F%s' %(n)]
            C += Fn*Zs*(-1)**(n*t)*(np.exp(-1*En*t) - np.exp(-1*En*(self.T-t)))
        return C
    def dwhisq_twopt_axial(self,t,p):
        x2 = t[0]
        ax = t[1]
        twopt = self.twopt_fitfcn_ss_ps(x2, p)
        axial = self.dwhisq_axial(ax, p)
        fitfcn = np.concatenate((twopt,axial))
        return fitfcn
    def dwhisq_twopt_osc_axial(self,t,p):
        x2 = t[0]
        ax = t[1]
        twopt = self.dwhisq_twopt_osc_ss_ps(x2, p)
        axial = self.dwhisq_axial_osc(ax, p)
        fitfcn = np.concatenate((twopt,axial))
        return fitfcn
    ##########################################################################################
    def B(self,p,b,snk,src,n,m):
        bsrc = b[:2]
        bsnk = b[2:]
        E0 = p['E0']
        En = self.E(p,n)
        Em = self.E(p,m)
        Z0sc = p['%s_Z0%s' %(bsrc,src)]
        Z0sk = p['%s_Z0%s' %(bsnk,snk)]
        Znsc = p['%s_Z%s%s' %(bsrc,n,src)]
        Zmsk = p['%s_Z%s%s' %(bsnk,m,snk)]
        Bnm = E0*Znsc*Zmsk / (np.sqrt(En*Em)*Z0sc*Z0sk)
        return Bnm
    def A(self,t,p,b,snk,src):
        E0 = p['E0']
        denom = 1.
        for n in range(1,self.nstates):
            En = self.E(p,n)
            D = En-E0
            denom += self.B(p,b,snk,src,n,n)*np.exp(-D*t)
        return 1./denom
    def asqtad_A(self,t,p,b,snk,src):
        E0 = p['E0']
        denom = 1.
        for n in range(1,self.nstates):
            En = self.asqtad_En(p,n)
            D = En-E0
            denom += (-1.)**(n*(t+1))*self.B(p,b,snk,src,n,n)*np.exp(-D*t)
        return 1./denom
    def G(self,p,b,n,m,gV):
        if n > m:
            if gV:
                g = p['gV%s%s' %(str(n),str(m))]
            else:
                g = p['gA%s%s' %(str(n),str(m))]
        else:
            if gV:
                g = p['gV%s%s' %(str(m),str(n))]
            else:
                g = p['gA%s%s' %(str(m),str(n))]
        return g
    def Dn(self,p,bsnk,bsrc,n,snk,src,gV):
        if gV:
            return p['%s_%s_C%s%s%s' %(bsnk,bsrc,n,snk,src)]
        else:
            return p['%s_%s_D%s%s%s' %(bsnk,bsrc,n,snk,src)]
    def E(self,p,n):
        En = p['E0']
        for i in range(n):
            En += np.exp(p['E%s' %str(i+1)])
        return En
    def Ej(self,p,j): # A3 meson mass
        E = p['Em0']
        for i in range(j):
            E += np.exp(p['Em%s' %str(i+1)])
        return E
    def asqtad_En(self,p,n):
        if n%2 == 0:
            En = p['E0']
            for i in range(0,n,2):
                En += np.exp(p['E%s' %str(i+2)])
        if n%2 == 1:
            En = p['E1']
            for i in range(1,n,2):
                En += np.exp(p['E%s' %str(i+2)])
        return En
    def C(self,p,b,snk,src,n,j):
        bsrc = b[:2]
        bsnk = b[2:]
        E0 = p['E0']
        En = self.E(p,n)
        Ej = self.Ej(p,j)
        Z0sc = p['%s_Z0%s' %(bsrc,src)]
        Z0sk = p['%s_Z0%s' %(bsnk,snk)]
        Znsc = p['%s_Z%s%s' %(bsrc,n,src)]
        Znsk = p['%s_Z%s%s' %(bsrc,n,snk)]
        Zj = p['Zm%sp' %(j)]
        Cnj = E0*(Znsc*Zj+Zj*Znsk) / (np.sqrt(En*Ej)*Z0sc*Z0sk) # the src snk flip comes from region I vs III
        return Cnj
    def H(self,p,b,n,j):
        if n > j:
            g = p['gH%s%s' %(str(n),str(j))]
        else:
            g = p['gH%s%s' %(str(j),str(n))]
        return g
    # derivative of effective mass for FH propagators
    def dmeff_fitfcn(self, t, p):
        fitfcn = p['ME'] + (p['C']*t+p['D'])*np.exp(-p['E']*t)
        return fitfcn
    def fhbaryon(self,t,p,b,snk,src):
        tau = float(self.tau)
        E0 = p['E0']
        A = self.A(t,p,b,snk,src)
        R1 = p['gA00']
        R2 = 0
        for n in range(1,self.nstates):
            B = self.B(p,b,snk,src,n,n)
            G = self.G(p,b,n,n)
            D = self.E(p,n)-E0
            R2 += B*G*np.exp(-D*t) * (t*(np.exp(-D*tau)-1)+np.exp(-D*tau)*(tau+1)-1) / tau
        R3 = 0
        for n in range(1,self.nstates):
            B0n = self.B(p,b,snk,src,0,n)
            Bn0 = self.B(p,b,snk,src,n,0)
            G0n = self.G(p,b,0,n)
            Gn0 = self.G(p,b,n,0)
            D = self.E(p,n)-E0
            R3 += (B0n*G0n+Bn0*Gn0) * np.exp(-D*(t+1)) * (1-np.exp(-D*tau))/(1-np.exp(-D))
        R4 = 0
        for n in range(1,self.nstates):
            for m in range(1,self.nstates):
                Bnm = self.B(p,b,snk,src,n,m)
                Gnm = self.G(p,b,n,m)
                Dnm = self.E(p,n)-self.E(p,m)
                R4 += Bnm*Gnm*np.exp(-Dnm*(t+1)) * (1-np.exp(-D*tau))/(1-np.exp(-D))
        return A*(R1+R2+R3+R4)
    def Rt(self,t,p,b,snk,src):
        E0 = p['E0']
        A = self.A(t,p,b,snk,src)
        R1 = (t+1)*p['gA00']
        R2 = 0
        for n in range(1,self.nstates):
            B = self.B(p,b,snk,src,n,n)
            G = self.G(p,b,n,n)
            D = self.E(p,n)-E0
            R2 += (t+1)*B*G*np.exp(-D*t)
        R3 = 0
        for n in range(self.nstates):
            for m in range(n):
                Bnm = self.B(p,b,snk,src,n,m)
                Gnm = self.G(p,b,n,m)
                Dn0 = self.E(p,n)-E0
                Dm0 = self.E(p,m)-E0
                Dmn = self.E(p,m)-self.E(p,n)
                R3 += Bnm*Gnm*(np.exp(-Dn0*t+Dmn*0.5)-np.exp(-Dm0*t-Dmn*0.5))/(np.exp(Dmn*0.5)-np.exp(-Dmn*0.5))
        R4 = 0
        for m in range(self.nstates):
            for n in range(m):
                Bnm = self.B(p,b,snk,src,n,m)
                Gnm = self.G(p,b,n,m)
                Dn0 = self.E(p,n)-E0
                Dm0 = self.E(p,m)-E0
                Dmn = self.E(p,m)-self.E(p,n)
                R4 += Bnm*Gnm*(np.exp(-Dn0*t+Dmn*0.5)-np.exp(-Dm0*t-Dmn*0.5))/(np.exp(Dmn*0.5)-np.exp(-Dmn*0.5))
        S = 0
        for n in range(self.nstates):
            for j in range(self.nstates):
                Cnj = self.C(p,b,snk,src,n,j)
                Hnj = self.H(p,b,n,j)
                Dn0 = self.E(p,n)-E0
                #Ej = self.Ej(p,j)
                S += Cnj*Hnj*np.exp(-1.*Dn0*t) #*(1.-np.exp(-1.*Ej*(self.T-t)/2.))/(np.exp(Ej)-1.)
        return A*(R1+R2+R3+R4+S)
    def asqtad_Rt(self,t,p,b,snk,src):
        E0 = p['E0']
        A = self.asqtad_A(t,p,b,snk,src)
        R1 = (t+1)*p['gA00']
        R2 = 0
        for n in range(1,self.nstates):
            B = self.B(p,b,snk,src,n,n)
            G = self.G(p,b,n,n)
            D = self.asqtad_En(p,n)-E0
            R2 += (-1.)**(n*(t+1))*(t+1)*B*G*np.exp(-D*t)
        R3 = 0
        for n in range(self.nstates):
            for m in range(n):
                Bnm = self.B(p,b,snk,src,n,m)
                Gnm = self.G(p,b,n,m)
                Dn0 = self.asqtad_En(p,n)-E0
                Dm0 = self.asqtad_En(p,m)-E0
                Dmn = self.asqtad_En(p,m)-self.asqtad_En(p,n)
                R3 += Bnm*Gnm*((-1.)**(n*(t+1))*np.exp(-Dn0*t+Dmn*0.5)-(-1.)**(m*(t+1))*np.exp(-Dm0*t-Dmn*0.5))/(np.exp(Dmn*0.5)-np.exp(-Dmn*0.5))
        R4 = 0
        for m in range(self.nstates):
            for n in range(m):
                Bnm = self.B(p,b,snk,src,n,m)
                Gnm = self.G(p,b,n,m)
                Dn0 = self.asqtad_En(p,n)-E0
                Dm0 = self.asqtad_En(p,m)-E0
                Dmn = self.asqtad_En(p,m)-self.asqtad_En(p,n)
                R4 += Bnm*Gnm*((-1.)**(n*(t+1))*np.exp(-Dn0*t+Dmn*0.5)-(-1.)**(m*(t+1))*np.exp(-Dm0*t-Dmn*0.5))/(np.exp(Dmn*0.5)-np.exp(-Dmn*0.5))
        S = 0
        for n in range(self.nstates):
            for j in range(self.nstates):
                Cnj = self.C(p,b,snk,src,n,j)
                Hnj = self.H(p,b,n,j)
                Dn0 = self.asqtad_En(p,n)-E0
                #Ej = self.Ej(p,j)
                S += (-1.)**(n*(t+1))*Cnj*Hnj*np.exp(-1.*Dn0*t) #*(1.-np.exp(-1.*Ej*(self.T-t)/2.))/(np.exp(Ej)-1.)
        return A*(R1+R2+R3+R4+S)
    def dM(self,t,p,b,snk,src):
        R1 = self.Rt(t+1,p,b,snk,src)
        R0 = self.Rt(t,p,b,snk,src)
        return R1-R0
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
    def twopt_baryon(self,t,p,b,snk,src):
        bsrc = b[:2]
        bsnk = b[2:]
        En = p['E0']
        fitfcn = p['%s_Z0%s' %(str(bsnk),str(snk))]*p['%s_Z0%s' %(str(bsrc),str(src))]/(2.*En) * np.exp(-1*En*t)
        for n in range(1, self.nstates):
            En += np.exp(p['E'+str(n)])
            fitfcn += p['%s_Z%s%s' %(str(bsnk),str(n),str(snk))]*p['%s_Z%s%s' %(str(bsrc),str(n),str(src))]/(2.*En) * np.exp(-1*En*t)
        return fitfcn
    def asqtad_twopt_baryon(self,t,p,b,snk,src):
        bsrc = b[:2]
        bsnk = b[2:]
        En = p['E0']
        fitfcn = p['%s_Z0%s' %(str(bsnk),str(snk))]*p['%s_Z0%s' %(str(bsrc),str(src))] * np.exp(-1*En*t)
        for n in range(1, self.nstates):
            En = self.asqtad_En(p,n)
            fitfcn += p['%s_Z%s%s' %(str(bsnk),str(n),str(snk))]*p['%s_Z%s%s' %(str(bsrc),str(n),str(src))] * (-1.)**(n*(t+1)) * np.exp(-1*En*t)
        return fitfcn
    def twopt_fitfcn_phiqq(self, t, p):
        En = p['E0']
        fitfcn = p['A0'] * (np.exp(-1*En*t) + np.exp(-1*En*(self.T-t)) + np.exp(-1*En*(self.T+t)))
        return fitfcn
    # Combined fitters
    # fh combined fitter
    def baryon_fhbaryon_ss_ps(self,t,p):
        x2 = t['indep'][0]
        fhx = t['indep'][1]
        ssl = []
        psl = []
        for b in t['basak']:
            ssl.append(self.twopt_baryon(x2,p,b,'s','s'))
            psl.append(self.twopt_baryon(x2,p,b,'p','s'))
        ss = ssl[0]
        ps = psl[0]
        for i in range(len(ssl)-1):
            ss = np.concatenate((ss,ssl[i+1]))
            ps = np.concatenate((ps,psl[i+1])) 
        fhssl = []
        fhpsl = []
        for b in t['basak']:
            fhssl.append(self.fhbaryon(fhx,p,b,'s','s'))
            fhpsl.append(self.fhbaryon(fhx,p,b,'p','s'))
        fhss = fhssl[0]
        fhps = fhpsl[0]
        for i in range(1,len(fhssl)):
            fhss = np.concatenate((fhss,fhssl[i]))
            fhps = np.concatenate((fhps,fhpsl[i]))
        fitfcn = np.concatenate((ss,ps,fhss,fhps))
        return fitfcn
    def baryon_rt_ss_ps(self,t,p):
        x2 = t['indep'][0]
        fhx = t['indep'][1]
        ssl = []
        psl = []
        for b in t['basak']:
            ssl.append(self.twopt_baryon(x2,p,b,'s','s'))
            psl.append(self.twopt_baryon(x2,p,b,'p','s'))
        ss = ssl[0]
        ps = psl[0]
        for i in range(len(ssl)-1):
            ss = np.concatenate((ss,ssl[i+1]))
            ps = np.concatenate((ps,psl[i+1]))
        fhssl = []
        fhpsl = []
        for b in t['basak']:
            fhssl.append(self.Rt(fhx,p,b,'s','s'))
            fhpsl.append(self.Rt(fhx,p,b,'p','s'))
        fhss = fhssl[0]
        fhps = fhpsl[0]
        for i in range(1,len(fhssl)):
            fhss = np.concatenate((fhss,fhssl[i]))
            fhps = np.concatenate((fhps,fhpsl[i]))
        fitfcn = np.concatenate((ss,ps,fhss,fhps))
        return fitfcn
    def asqtad_baryon_rt_ss_ps(self,t,p):
        x2 = t['indep'][0]
        fhx = t['indep'][1]
        ssl = []
        psl = []
        for b in t['basak']:
            ssl.append(self.asqtad_twopt_baryon(x2,p,b,'s','s'))
            psl.append(self.asqtad_twopt_baryon(x2,p,b,'p','s'))
        ss = ssl[0]
        ps = psl[0]
        for i in range(len(ssl)-1):
            ss = np.concatenate((ss,ssl[i+1]))
            ps = np.concatenate((ps,psl[i+1]))
        fhssl = []
        fhpsl = []
        for b in t['basak']:
            fhssl.append(self.asqtad_Rt(fhx,p,b,'s','s'))
            fhpsl.append(self.asqtad_Rt(fhx,p,b,'p','s'))
        fhss = fhssl[0]
        fhps = fhpsl[0]
        for i in range(1,len(fhssl)):
            fhss = np.concatenate((fhss,fhssl[i]))
            fhps = np.concatenate((fhps,fhpsl[i]))
        fitfcn = np.concatenate((ss,ps,fhss,fhps))
        return fitfcn
    def baryon_dm_ss_ps(self,t,p):
        x2 = t['indep'][0]
        fhx = t['indep'][1]
        ssl = []
        psl = []
        for b in t['basak']:
            ssl.append(self.twopt_baryon(x2,p,b,'s','s'))
            psl.append(self.twopt_baryon(x2,p,b,'p','s'))
        ss = ssl[0]
        ps = psl[0]
        for i in range(len(ssl)-1):
            ss = np.concatenate((ss,ssl[i+1]))
            ps = np.concatenate((ps,psl[i+1]))
        fhssl = []
        fhpsl = []
        for b in t['basak']:
            fhssl.append(self.dM(fhx,p,b,'s','s'))
            fhpsl.append(self.dM(fhx,p,b,'p','s'))
        fhss = fhssl[0]
        fhps = fhpsl[0]
        for i in range(1,len(fhssl)):
            fhss = np.concatenate((fhss,fhssl[i]))
            fhps = np.concatenate((fhps,fhpsl[i]))
        fitfcn = np.concatenate((ss,ps,fhss,fhps))
        return fitfcn
    def baryon_rt_fitline(self,t,p):
        b = 'G1G1'
        ss = self.Rt(t+1,p,b,'s','s') - self.Rt(t,p,b,'s','s')
        ps = self.Rt(t+1,p,b,'p','s') - self.Rt(t,p,b,'p','s')
        return ss, ps
    def fhbaryon_ss_ps(self,t,p):
        fhssl = []
        fhpsl = []
        for b in t['basak']:
            fhssl.append(self.fhbaryon(t['indep'],p,b,'s','s'))
            fhpsl.append(self.fhbaryon(t['indep'],p,b,'p','s'))
        fhss = fhssl[0]
        fhps = fhpsl[0]
        for i in range(1,len(fhssl)):
            fhss = np.concatenate((fhss,fhssl[i]))
            fhps = np.concatenate((fhps,fhpsl[i]))
        fitfcn = np.concatenate((fhss,fhps))
        return fitfcn
    # two point ss and ps simultaneous fit 
    def dwhisq_twopt_osc_ss_ps(self, t, p):
        fitfcn_ss = self.dwhisq_twopt_osc(t,p,'s','s')
        fitfcn_ps = self.dwhisq_twopt_osc(t,p,'p','s')
        fitfcn = np.concatenate((fitfcn_ss, fitfcn_ps))
        return fitfcn
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
            ssl.append(self.twopt_baryon(t['indep'],p,b,'s','s'))
            psl.append(self.twopt_baryon(t['indep'],p,b,'p','s'))
        ss = ssl[0]
        ps = psl[0]
        for i in range(len(ssl)-1):
            ss = np.concatenate((ss,ssl[i+1]))
            ps = np.concatenate((ps,psl[i+1]))
        fitfcn = np.concatenate((ss,ps))
        return fitfcn
    def asqtad_twopt_baryon_ss_ps(self,t,p):
        ssl = []
        psl = []
        for b in t['basak']:
            ssl.append(self.asqtad_twopt_baryon(t['indep'],p,b,'s','s'))
            psl.append(self.asqtad_twopt_baryon(t['indep'],p,b,'p','s'))
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
    # tmin stability plot
    try:
        if fittbl['fhtmin'][-1]-fittbl['fhtmin'][0] > 0:
            if fittbl['fhtmax'][-1]-fittbl['fhtmax'][0] > 0:
                output = []
                for t in range(len(fittbl['fhtmin'])):
                    output.append((fittbl['fhtmin'][t], fittbl['fhtmax'][t], fittbl['post'][t][key]))
                dtype = [('fhtmin', int), ('fhtmax', int), ('post', gv._gvarcore.GVar)]
                output = np.array(output, dtype=dtype)
                output = np.sort(output, order='fhtmax')
                setwidth = fittbl['fhtmin'][-1]-fittbl['fhtmin'][0]+1
                fig = plt.figure()
                ax1 = fig.add_subplot(111)
                for subset in range(len(output)/setwidth):
                    pltdata = output[setwidth*subset:setwidth*(subset+1)]
                    x = [pltdata[i][0] for i in range(len(pltdata))]
                    y = [pltdata[i][2] for i in range(len(pltdata))]
                    y_plt = np.array([dat.mean for dat in y])
                    e_plt = np.array([dat.sdev for dat in y])
                    ax1.errorbar(x, y_plt, e_plt, label='fhtmax: '+str(pltdata[0][1]))
                plt.title(title+' fhtmin stability plot')
                plt.xlabel('fhtmin')
                plt.ylabel(key)
                plt.xlim(x[0]-0.5, x[-1]+0.5)
                plt.legend()
            else:
                x = fittbl['fhtmin']
                y = np.array([data[key] for data in fittbl['post']])
                scatter_plot(x, y, title+' fhtmin stability plot', 'fhtmin (fhtmax='+str(fittbl['fhtmax'][0])+')', key, xlim=[x[0]-0.5,x[-1]+0.5])
    except: pass
    try: 
        # tmax stability plot
        if fittbl['fhtmax'][-1]-fittbl['fhtmax'][0] > 0:
            if fittbl['fhtmin'][-1]-fittbl['fhtmin'][0] > 0:
                output = []
                for t in range(len(fittbl['fhtmin'])):
                    output.append((fittbl['fhtmin'][t], fittbl['fhtmax'][t], fittbl['post'][t][key]))
                dtype = [('fhtmin', int), ('fhtmax', int), ('post', gv._gvarcore.GVar)]
                output = np.array(output, dtype=dtype)
                output = np.sort(output, order='fhtmin')
                setwidth = fittbl['fhtmax'][-1]-fittbl['fhtmax'][0]+1
                fig = plt.figure()
                ax1 = fig.add_subplot(111)
                for subset in range(len(output)/setwidth):
                    pltdata = output[setwidth*subset:setwidth*(subset+1)]
                    x = [pltdata[i][1] for i in range(len(pltdata))]
                    y = [pltdata[i][2] for i in range(len(pltdata))]
                    y_plt = np.array([dat.mean for dat in y])
                    e_plt = np.array([dat.sdev for dat in y])
                    ax1.errorbar(x, y_plt, e_plt, label='fhtmin: '+str(pltdata[0][0]))
                plt.title(title+' fhtmax stability plot')
                plt.xlabel('fhtmax')
                plt.ylabel(key)
                plt.xlim(x[0]-0.5, x[-1]+0.5)
                plt.legend()
            else:
                x = fittbl['fhtmax']
                y = np.array([data[key] for data in fittbl['post']])
                scatter_plot(x, y, title+' fhtmax stability plot', 'fhtmax (fhtmin='+str(fittbl['fhtmin'][0])+')', key, xlim=[x[0]-0.5,x[-1]+0.5])
        else: pass #print key,':',fittbl['post'][0][key]
    except: pass
    # tmin stability plot
    try:
        if fittbl['gVtmin'][-1]-fittbl['gVtmin'][0] > 0:
            if fittbl['gVtmax'][-1]-fittbl['gVtmax'][0] > 0:
                output = []
                for t in range(len(fittbl['gVtmin'])):
                    output.append((fittbl['gVtmin'][t], fittbl['gVtmax'][t], fittbl['post'][t][key]))
                dtype = [('gVtmin', int), ('gVtmax', int), ('post', gv._gvarcore.GVar)]
                output = np.array(output, dtype=dtype)
                output = np.sort(output, order='gVtmax')
                setwidth = fittbl['gVtmin'][-1]-fittbl['gVtmin'][0]+1
                fig = plt.figure()
                ax1 = fig.add_subplot(111)
                for subset in range(len(output)/setwidth):
                    pltdata = output[setwidth*subset:setwidth*(subset+1)]
                    x = [pltdata[i][0] for i in range(len(pltdata))]
                    y = [pltdata[i][2] for i in range(len(pltdata))]
                    y_plt = np.array([dat.mean for dat in y])
                    e_plt = np.array([dat.sdev for dat in y])
                    ax1.errorbar(x, y_plt, e_plt, label='fhtmax: '+str(pltdata[0][1]))
                plt.title(title+' gVtmin stability plot')
                plt.xlabel('gVtmin')
                plt.ylabel(key)
                plt.xlim(x[0]-0.5, x[-1]+0.5)
                plt.legend()
            else:
                x = fittbl['gVtmin']
                y = np.array([data[key] for data in fittbl['post']])
                scatter_plot(x, y, title+' gVtmin stability plot', 'gVtmin (gVtmax='+str(fittbl['gVtmax'][0])+')', key, xlim=[x[0]-0.5,x[-1]+0.5])
    except: pass
    try:
        # tmax stability plot
        if fittbl['gVtmax'][-1]-fittbl['gVtmax'][0] > 0:
            if fittbl['gVtmin'][-1]-fittbl['gVtmin'][0] > 0:
                output = []
                for t in range(len(fittbl['gVtmin'])):
                    output.append((fittbl['gVtmin'][t], fittbl['gVtmax'][t], fittbl['post'][t][key]))
                dtype = [('gVtmin', int), ('gVtmax', int), ('post', gv._gvarcore.GVar)]
                output = np.array(output, dtype=dtype)
                output = np.sort(output, order='fhtmin')
                setwidth = fittbl['gVtmax'][-1]-fittbl['gVtmax'][0]+1
                fig = plt.figure()
                ax1 = fig.add_subplot(111)
                for subset in range(len(output)/setwidth):
                    pltdata = output[setwidth*subset:setwidth*(subset+1)]
                    x = [pltdata[i][1] for i in range(len(pltdata))]
                    y = [pltdata[i][2] for i in range(len(pltdata))]
                    y_plt = np.array([dat.mean for dat in y])
                    e_plt = np.array([dat.sdev for dat in y])
                    ax1.errorbar(x, y_plt, e_plt, label='gVtmin: '+str(pltdata[0][0]))
                plt.title(title+' gVtmax stability plot')
                plt.xlabel('gVtmax')
                plt.ylabel(key)
                plt.xlim(x[0]-0.5, x[-1]+0.5)
                plt.legend()
            else:
                x = fittbl['gVtmax']
                y = np.array([data[key] for data in fittbl['post']])
                scatter_plot(x, y, title+' gVtmax stability plot', 'gVtmax (gVtmin='+str(fittbl['gVtmin'][0])+')', key, xlim=[x[0]-0.5,x[-1]+0.5])
        else: pass #print key,':',fittbl['post'][0][key]
    except: pass
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
