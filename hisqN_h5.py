# Fit hisq mesons for tuning
import numpy as np
import h5py as h5
import lsqfit
import gvar as gv
import matplotlib.pyplot as plt
import yaml as yml


def meta():
    params = dict()
    params['update_boot0'] = False
    params['plot'] = True
    params['tmin'] = [8,20]
    params['tmax'] = [24,24]
    params['Ns'] = [3,3]
    params['corrs'] = ['S2p8_S1p4', 'S4p2_S1p4', 'S1p4_S1p4', 'P_S1p4'] #, 'S1p4_P', 'S4p2_P', 'S2p8_P', 'P_P']
    #params['corrs'] = ['S4p2_S1p4','S2p8_S1p4','P_S1p4']
    #params['corrs'] = ['P_S1p4']
    params['srckey'] = []
    return params

def set_priors(params,Ns):
    snkkey = [p.split('_')[0] for p in params['corrs']]
    srckey = params['srckey']
    p0 = dict()
    ex = dict()
    p0['E0'] = gv.gvar(0.725, 0.05)
    p0['A0_P_S1p4']    = gv.gvar(0.0040, 0.0020)
    p0['A0_S1p4_S1p4'] = gv.gvar(0.0016, 0.0008)
    p0['A0_S2p8_S1p4'] = gv.gvar(0.0010, 0.0005)
    p0['A0_S4p2_S1p4'] = gv.gvar(0.0008, 0.0004)
    p0['E1'] = gv.gvar(-1.1, 0.65)
    p0['A1_P_S1p4']    = gv.gvar(0.0, 0.0040)
    p0['A1_S1p4_S1p4'] = gv.gvar(0.0, 0.0008)
    p0['A1_S2p8_S1p4'] = gv.gvar(0.0, 0.0005)
    p0['A1_S4p2_S1p4'] = gv.gvar(0.0, 0.0004)
    p0['E2'] = gv.gvar(-1.1, 0.65)
    p0['A2_P_S1p4']    = gv.gvar(0.0, 0.0040)
    p0['A2_S1p4_S1p4'] = gv.gvar(0.0, 0.0008)
    p0['A2_S2p8_S1p4'] = gv.gvar(0.0, 0.0005)
    p0['A2_S4p2_S1p4'] = gv.gvar(0.0, 0.0004)
    p0['E3'] = gv.gvar(-1.1, 0.65)
    p0['A3_P_S1p4']    = gv.gvar(0.0, 0.0040)
    p0['A3_S1p4_S1p4'] = gv.gvar(0.0, 0.0008)
    p0['A3_S2p8_S1p4'] = gv.gvar(0.0, 0.0005)
    p0['A3_S4p2_S1p4'] = gv.gvar(0.0, 0.0004)
    #p0['E4'] = gv.gvar(-1.1, 0.65)
    ##p0['Z4_P'] = gv.gvar(0.0, 0.015)
    #p0['Z4_S1p4'] = gv.gvar(0.0, 0.012)
    ##p0['Z4_S2p8'] = gv.gvar(0.0, 0.008)
    #p0['Z4_S4p2'] = gv.gvar(0.0, 0.006)
    #p0['E5'] = gv.gvar(-1.1, 0.65)
    #p0['Z5_P'] = gv.gvar(0.0, 0.015)
    #p0['Z5_S1p4'] = gv.gvar(0.0, 0.012)
    #p0['Z5_S2p8'] = gv.gvar(0.0, 0.008)
    #p0['Z5_S4p2'] = gv.gvar(0.0, 0.006)
    #p0['E6'] = gv.gvar(-1.1, 0.65)
    #p0['Z6_P'] = gv.gvar(0.0, 0.015)
    #p0['Z6_S1p4'] = gv.gvar(0.0, 0.012)
    #p0['Z6_S2p8'] = gv.gvar(0.0, 0.008)
    #p0['Z6_S4p2'] = gv.gvar(0.0, 0.006)
    #p0['E7'] = gv.gvar(-1.1, 0.65)
    #p0['Z7_P'] = gv.gvar(0.0, 0.015)
    #p0['Z7_S1p4'] = gv.gvar(0.0, 0.012)
    #p0['Z7_S2p8'] = gv.gvar(0.0, 0.008)
    #p0['Z7_S4p2'] = gv.gvar(0.0, 0.006)
    #p0['E8'] = gv.gvar(-1.1, 0.65)
    #p0['Z8_P'] = gv.gvar(0.0, 0.015)
    #p0['Z8_S1p4'] = gv.gvar(0.0, 0.012)
    #p0['Z8_S2p8'] = gv.gvar(0.0, 0.008)
    #p0['Z8_S4p2'] = gv.gvar(0.0, 0.006)
    #p0['E9'] = gv.gvar(-1.1, 0.65)
    #p0['Z9_P'] = gv.gvar(0.0, 0.015)
    #p0['Z9_S1p4'] = gv.gvar(0.0, 0.012)
    #p0['Z9_S2p8'] = gv.gvar(0.0, 0.008)
    #p0['Z9_S4p2'] = gv.gvar(0.0, 0.006)
    return p0

def read_file(params,fname):
    pname = '/Users/cchang5/Physics/c51/data'
    f = h5.File('%s/%s' %(pname,fname))
    if fname == 'l2464f211b600m0130m0509m635a_avg.h5':
        corr = np.array(f['/l2464f211b600m0130m0509m635a/wf1p0_m51p2_l58_a51p5_smrw5p0_n75/hisq_spec/ml0p0130_ms0p0509/phi_jj_5/corr'].value).real
    f.close()
    return corr

def make_gvars(params,corr):
    data = gv.dataset.avg_data(corr)
    return data

def plot_data(params,data):
    T = len(data)
    fig = plt.figure(figsize=(7,4.326237))
    ax = plt.axes([0.15,0.15,0.8,0.8])
    meff = np.array([np.arccosh((data[(t+1)%T]+data[t-1])/(2*data[t])) for t in range(T)])
    ax.errorbar(x=range(T),y=[i.mean for i in meff],yerr=[i.sdev for i in meff],marker='s',fillstyle='none')
    #ax.set_ylim([0.2,0.22])
    #ax.set_xlim([0,60])
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    ax.set_xlabel('t', fontsize=20)
    ax.set_ylabel('meff', fontsize=20)
    plt.draw()
    plt.show()
    raise SystemExit
    fig = plt.figure(figsize=(7,4.326237))
    ax = plt.axes([0.15,0.15,0.8,0.8])
    #for src in params['srckey']:
    #    scor = np.array(cdic['%s_%s' %(src,src)]*np.exp(mdic['%s_%s' %(src,src)]*range(T))/(2.*mdic['%s_%s' %(src,src)]))
    #    sdic['%s_%s' %(src,src)] = scor
    #    ax.errorbar(x=range(T),y=[i.mean for i in scor],yerr=[i.sdev for i in scor],marker='s',fillstyle='none',label='%s_%s' %(src,src))
    #ax.legend()
    #ax.set_ylim([0,0.1])
    #ax.set_xlim([0,25])
    #ax.xaxis.set_tick_params(labelsize=16)
    #ax.yaxis.set_tick_params(labelsize=16)
    #ax.set_xlabel('t', fontsize=20)
    #ax.set_ylabel('Z', fontsize=20)
    #plt.draw()
    #plt.show()
    for c in params['corrs']:
        snk = c.split('_')[0]
        src = c.split('_')[1]
        #if snk == src:
        #    continue
        fig = plt.figure(figsize=(7,4.326237))
        ax = plt.axes([0.15,0.15,0.8,0.8])
        scor = np.array(cdic['%s_%s' %(snk,src)]*np.exp(mdic['%s_%s' %(snk,src)]*range(T))*2*mdic['%s_%s' %(snk,src)])
        sdic['%s_%s' %(snk,src)] = scor
        ax.errorbar(x=np.array(range(T))[2:T/2],y=np.array([i.mean for i in scor])[2:T/2],yerr=np.array([i.sdev for i in scor])[2:T/2],marker='s',fillstyle='none',label='%s_%s' %(snk,src))
        ax.legend()
        #ax.set_ylim([0,0.1])
        ax.set_xlim([0,25])
        ax.xaxis.set_tick_params(labelsize=16)
        ax.yaxis.set_tick_params(labelsize=16)
        ax.set_xlabel('t', fontsize=20)
        ax.set_ylabel('Z', fontsize=20)
        plt.draw()
        plt.show()
    return 0

class fit_function():
    def __init__(self, T, Ns):
        self.T = T
        self.Ns = Ns
    def En(self,p,n):
        E = p['E0']
        for i in range(1,n):
            E += np.exp(p['E%s' %n])
        return E
    def twopt_fit(self,x,p):    
        t = x['t']
        corrs = x['corrs']
        result = []
        for c in corrs:
            fitn = self.twopt_single(t,p,c)
            if len(result) == 0:
                result = fitn
            else:
                result = np.concatenate((result,fitn),axis=0)
        return result
    def twopt_single(self,t,p,c):
        f = 0
        for n in range(self.Ns):
            E = self.En(p,n)
            f += p['A%s_%s' %(n,c)]*np.exp(-E*t)/(2.*E)
        return f
        

def fit_data(params,data):
    result = dict()
    result['tmin'] = []
    result['tmax'] = []
    result['Ns'] = []
    result['fit'] = []
    T = len(data)/len(params['corrs'])
    Ncorr = len(params['corrs'])
    for tmin in range(params['tmin'][0],params['tmin'][1]+1):
        for tmax in range(params['tmax'][0],params['tmax'][1]+1):
            for n in range(params['Ns'][0],params['Ns'][1]+1):
                print "trange: [%s,%s] Ns: %s" %(tmin,tmax,n)
                x = dict()
                x['t'] = np.array(range(tmin, tmax+1))
                x['corrs'] = params['corrs']
                tmask = [i+j*T for j in range(Ncorr) for i in x['t']]
                y = data[tmask]
                try:
                    #f = open('./nucleon_p0/%s_%s_%s.yml' %(tmin,tmax,n),'r')
                    f = open('./nucleon_p0/boot0.yml','r')
                    p0 = yml.load(f)
                    f.close()
                except:
                    p0 = None
                fcn = fit_function(T,n)
                priors = set_priors(params,n)
                fit = lsqfit.nonlinear_fit(data=(x,y),prior=priors,p0=p0,fcn=fcn.twopt_fit,maxit=100000)
                result['tmin'].append(tmin)
                result['tmax'].append(tmax)
                result['Ns'].append(n)
                result['fit'].append(fit)
                if params['update_boot0']:
                    #f = open('./nucleon_p0/%s_%s_%s.yml' %(tmin,tmax,n),'w+')
                    f = open('./nucleon_p0/boot0.yml', 'w+')
                    boot0 = dict()
                    for k in fit.p.keys():
                        boot0[k] = fit.p[k].mean
                    yml.dump(boot0,f)
                    f.flush()
                    f.close()
    return result

def plot_result(result):
    if result['tmin'][-1]-result['tmin'][0] > 0:
        fig = plt.figure(figsize=(7,4.326237))
        ax = plt.axes([0.15,0.30,0.8,0.65])
        ax.errorbar(x=result['tmin'],y=[i.p['E0'].mean for i in result['fit']],yerr=[i.p['E0'].sdev for i in result['fit']],marker='s',fillstyle='none',ls='None')
        ax.legend()
        ax.xaxis.set_tick_params(labelsize=16)
        ax.yaxis.set_tick_params(labelsize=16)
        ax.set_xlabel('$t_{min}$', fontsize=20)
        ax.set_ylabel('$E_0$', fontsize=20)
        ax.set_xlim([result['tmin'][0]-1,result['tmin'][-1]+1])
        ax2 = plt.axes([0.15,0.15,0.8,0.15])
        ax2.errorbar(x=result['tmin'],y=[i.Q for i in result['fit']],marker='s',fillstyle='none')
        ax2.errorbar(x=[-10,100], y = [0.05, 0.05], ls='--', color='#ec5d57')
        ax2.xaxis.set_tick_params(labelsize=16)
        ax2.yaxis.set_tick_params(labelsize=16)
        ax2.set_xlabel('$t_{min}$', fontsize=20)
        ax2.set_ylabel('$Q$', fontsize=20)
        ax2.set_xlim([result['tmin'][0]-1,result['tmin'][-1]+1])
        ax2.set_ylim([0.01,1.1])
        ax2.set_yscale('log')
        ax2.set_xlim([result['tmin'][0]-1,result['tmin'][-1]+1])
        plt.draw()
        plt.show()
    if result['tmax'][-1]-result['tmax'][0] > 0:
        fig = plt.figure(figsize=(7,4.326237))
        ax = plt.axes([0.15,0.30,0.8,0.65])
        ax.errorbar(x=result['tmax'],y=[i.p['E0'].mean for i in result['fit']],yerr=[i.p['E0'].sdev for i in result['fit']],marker='s',fillstyle='none')
        ax.legend()
        ax.xaxis.set_tick_params(labelsize=16)
        ax.yaxis.set_tick_params(labelsize=16)
        ax.set_xlabel('$t_{max}$', fontsize=20)
        ax.set_ylabel('$E_0$', fontsize=20)
        ax.set_xlim([result['tmax'][0]-1,result['tmax'][-1]+1])
        ax2 = plt.axes([0.15,0.15,0.8,0.15])
        ax2.errorbar(x=result['tmax'],y=[i.Q for i in result['fit']],marker='s',fillstyle='none')
        ax2.xaxis.set_tick_params(labelsize=16)
        ax2.yaxis.set_tick_params(labelsize=16)
        ax2.errorbar(x=[-1,100], y = [0.05, 0.05], ls='--', color='#ec5d57')
        ax2.set_xlabel('$t_{max}$', fontsize=20)
        ax2.set_ylabel('$Q$', fontsize=20)
        ax2.set_xlim([result['tmax'][0]-1,result['tmax'][-1]+1])
        ax2.set_ylim([0.01,1.1])
        ax2.set_yscale('log')
        plt.draw()
        plt.show()
    if result['Ns'][-1]-result['Ns'][0] > 0:
        fig = plt.figure(figsize=(7,4.326237))
        ax = plt.axes([0.15,0.30,0.8,0.65])
        ax.errorbar(x=result['Ns'],y=[i.p['E0'].mean for i in result['fit']],yerr=[i.p['E0'].sdev for i in result['fit']],marker='s',fillstyle='none')
        ax.legend()
        ax.xaxis.set_tick_params(labelsize=16)
        ax.yaxis.set_tick_params(labelsize=16)
        ax.set_xlabel('$N_{s}$', fontsize=20)
        ax.set_ylabel('$E_0$', fontsize=20)
        ax2 = plt.axes([0.15,0.15,0.8,0.15])
        ax2.errorbar(x=result['Ns'],y=[i.Q for i in result['fit']],marker='s',fillstyle='none')
        ax2.errorbar(x=[-1,20], y = [0.05, 0.05], ls='--', color='#ec5d57')
        ax2.xaxis.set_tick_params(labelsize=16)
        ax2.yaxis.set_tick_params(labelsize=16)
        ax2.set_xlabel('$N_{s}$', fontsize=20)
        ax2.set_ylabel('$Q$', fontsize=20)
        ax2.set_ylim([0.01,1.1])
        ax2.set_yscale('log')
        plt.draw()
        plt.show()
    if result['tmax'][-1]-result['tmax'][0] == 0:
        if result['tmin'][-1]-result['tmin'][0] == 0:
            if result['Ns'][-1]-result['Ns'][0] == 0:
                print result['fit'][0]
                print result['fit'][0].p['E0']

if __name__=='__main__':
    params = meta()
    corr = read_file(params, 'l2464f211b600m0130m0509m635a_avg.h5')
    data = make_gvars(params,corr)
    if params['plot']:
        pltchk = plot_data(params,data)
    result = fit_data(params,data)
    reschk = plot_result(result)
