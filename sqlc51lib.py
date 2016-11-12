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

def fhbaryon_priors(priors,fhpriors,basak,nstates):
    p = baryon_priors(priors,basak,nstates)
    for n in range(nstates):
        for k in fhpriors[n+1].keys():
            p[k] = fhpriors[n+1][k]
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
            scaled2pt.append(twopt[t]*2.*E0/(np.exp(-E0*t)+phase*np.exp(-E0*(self.T-t))))
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

def chi2freq(chi2,prior,post):
    chi2f = chi2
    for k in prior.keys():
        chi2f += -1*( (prior[k].mean-post[k].mean)/prior[k].sdev )**2
    return chi2f

# sets calculated
def fitscript_v2(trange,T,data,priors,fcn,init=None,basak=None):
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
    lgbftbl = []
    rawoutput = []
    for tmin in range(trange['tmin'][0], trange['tmin'][1]+1):
        for tmax in range(trange['tmax'][0], trange['tmax'][1]+1):
            x = x_indep(tmin, tmax)
            xlist, y = y_dep(x, data, sets)
            if basak is not None:
                x = {'indep': x, 'basak': basak}
            else: pass
            fit = lsqfit.nonlinear_fit(data=(x,y),prior=priors,fcn=fcn,p0=init,maxit=1000000) #,svdcut=1E-3)
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
    fittbl['rawoutput'] = rawoutput
    return fittbl

def fitscript_v3(trange,fhtrange,T,data,priors,fcn,init=None,basak=None):
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
    rawoutput = []
    for tmin in range(trange['tmin'][0], trange['tmin'][1]+1):
        for tmax in range(trange['tmax'][0], trange['tmax'][1]+1):
            for fhtmin in range(fhtrange['tmin'][0], fhtrange['tmin'][1]+1):
                for fhtmax in range(fhtrange['tmax'][0], fhtrange['tmax'][1]+1):
                    x2 = x_indep(tmin, tmax)
                    fhx = x_indep(fhtmin, fhtmax)
                    x = [x2,fhx]
                    xlist, y = y_dep_v2(x, data, sets)
                    if basak is not None:
                        x = {'indep': x, 'basak': basak}
                    else: pass
                    fit = lsqfit.nonlinear_fit(data=(x,y),prior=priors,fcn=fcn,p0=init,maxit=1000000) #,svdcut=1E-5)
                    print fhtmin, fhtmax, fit.p['gA00'], fit.chi2/fit.dof
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
    def __init__(self, T, nstates=1, tau=1):
        self.T = T
        self.nstates = nstates
        self.tau = tau
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
    def G(self,p,b,n,m):
        if n > m:
            g = p['gA%s%s' %(str(n),str(m))]
        else:
            g = p['gA%s%s' %(str(m),str(n))]
        return g
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
