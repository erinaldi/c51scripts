#!/usr/bin/env python

#Scalar current
import numpy as np
import matplotlib.pyplot as plt
import gvar as gv
import h5py
import yaml
import lsqfit
import collections

### READ DATASET ###
def parameters():
    params = dict()
    #Possible datasets: S, A3, V4, T12
    params['filename'] = 'l2464f211b600m0102m0509m635_tune_avg.h5'
    params['dirname'] = '' #'l1648f211b580m013m065m838a/wf1p0_m51p3_l512_a52p0_smrw4p5_n60/spec_ml0p0158_ms0p0902/'
    params['baryon'] = '' #'proton'
    params['subset'] = '' #'V4'
    params['plt_flag'] = 'on'
    #Plotting parameters
    params['pltxmin'] = 1
    params['pltxmax'] = 15
    return params

def read_data(filename, datapath, src=0, snk=0):
    print "Deprecated to open_data"
    #src/snk smear or point
    #00 = smear smear
    #10 = point smear
    f = h5py.File('/Users/cchang5/c51/data/'+filename,'r')
    data = f[datapath][()]
    data = data[:,:,src,snk] #00 is particle particle smear smear
    f.close()
    return data

def open_data(filename, datapath):
    f = h5py.File(filename,'r')
    data = f[datapath][()]
    f.close()
    return data

def parity_avg(pos, neg, phase=1):
    neg = phase*np.roll(np.array(neg[:,::-1]), 1, axis=1)
    neg[:,0] = phase*neg[:,0]
    avg = 0.5*(pos + neg)
    return avg

def fold(meson, phase=1.0):
    meson_p = phase*np.roll(meson[:,::-1], 1, axis=1)
    meson_avg = 0.5*(meson + meson_p)
    T = len(meson[0])/2
    return meson_avg[:,:T]

def make_gvars(data):
	data_gv = np.array(gv.dataset.avg_data(data))
	return data_gv

def ispin_avg(UU_up, UU_dn, DD_up=0, DD_dn=0, subset='twopt'):
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

def bs_draws(params, nbs):
    ens = params['current_fit']['ens']
    ml = str(params['current_fit']['ml'])
    ms = str(params['current_fit']['ms'])
    loc = params[ens][ml+'_'+ms]['data_loc']
    cfgs = len(open_data(loc['file_loc'], loc['cfgs_srcs']))
    boot0 = np.array([np.arange(cfgs)])
    draws = np.random.randint(cfgs, size=(nbs, cfgs))
    draws = np.concatenate((boot0, draws), axis=0)
    return draws

class process_bootstrap():
    def __init__(self, fittbl):
        self.fittbl_boot0 = fittbl[0, 1]
        self.fittbl_bs = fittbl[1:, 1]
        self.tmin = self.fittbl_boot0['tmin']
        self.tmax = self.fittbl_boot0['tmax']
    def __call__(self):
        return self.fittbl_boot0, self.fittbl_bs
    def read_boot0(self, key):
        boot0 = np.array([self.fittbl_boot0['post'][i][key].mean for i in range(len(self.fittbl_boot0['post']))])
        return boot0
    def read_boot0_sdev(self, key):
        boot0_sdev = np.array([self.fittbl_boot0['post'][i][key].sdev for i in range(len(self.fittbl_boot0['post']))])
        return boot0_sdev
    def read_bs(self, key, reshape_flag='off'):
        bs = np.array([self.fittbl_bs[i]['post'][j][key].mean for i in range(len(self.fittbl_bs)) for j in range(len(self.fittbl_bs[i]['post']))])
        if reshape_flag == 'on':
            bs = bs.reshape((len(bs)/len(self.tmin),len(self.tmin)))
        return bs

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
        for t in range(len(twopt)-tau):
            if style=='cosh':
        	    meff.append(np.arccosh((twopt[(t+tau)%self.T]+twopt[t-tau])/(2*twopt[t])))
            elif style=='log':
                meff.append(np.log(twopt[t]/twopt[(t+tau)%self.T])/tau)
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

def read_yaml(ymlname):
    f = open('/Users/cchang5/c51/scripts/'+ymlname,'r')
    dictionary = yaml.load(f)
    f.close()
    return dictionary

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

def y_dep(x, y, sets):
    for s in range(1, sets): x = np.concatenate((x, x+len(y)/sets))
    #print x
    y = y[x]
    #print y
    return y

# define number of sets
def fitscript(trange, data, prior, fcn, sets=1, result_flag='off'):
    print "depricated to v2 I think?"
    posterior = []
    tmintbl = []
    tmaxtbl = []
    for tmin in range(trange['tmin'][0], trange['tmin'][1]+1):
        for tmax in range(trange['tmax'][0], trange['tmax'][1]+1):
            x = x_indep(tmin, tmax)
            y = y_dep(x, data, sets)
            fit = lsqfit.nonlinear_fit(data=(x,y), prior=prior, fcn=fcn)
            if result_flag=='on':
                print tmin, tmax,
                #for n in prior.keys():
                #    print fit.p[n].mean,'(',fit.p[n].sdev,')',
                #print '\n'
                print fit
            posterior.append(fit.p)
            tmintbl.append(tmin)
            tmaxtbl.append(tmax)
    fittbl = dict()
    fittbl['tmin'] = tmintbl
    fittbl['tmax'] = tmaxtbl
    fittbl['post'] = posterior
    return fittbl

# sets calculated
def fitscript_v2(trange, T, data, priors, fcn, result_flag='off'):
    sets = len(data)/(T/2)
    posterior = []
    tmintbl = []
    tmaxtbl = []
    for tmin in range(trange['tmin'][0], trange['tmin'][1]+1):
        for tmax in range(trange['tmax'][0], trange['tmax'][1]+1):
            x = x_indep(tmin, tmax)
            y = y_dep(x, data, sets)
            fit = lsqfit.nonlinear_fit(data=(x,y), prior=priors, fcn=fcn)
            if result_flag=='on':
                print tmin, tmax,
                print fit
            posterior.append(fit.p)
            tmintbl.append(tmin)
            tmaxtbl.append(tmax)
    fittbl = dict()
    fittbl['tmin'] = tmintbl
    fittbl['tmax'] = tmaxtbl
    fittbl['post'] = posterior
    return fittbl

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
        fitfcn = p['Z0_s']**2 * (np.exp(-1*En*t) + np.exp(-1*En*(self.T-t)) + np.exp(-1*En*(self.T+t)))
        #fitfcn = p['Z0_s']*np.exp(-1*En*t)
        for n in range(1, self.nstates):
            En += np.exp(p['E'+str(n)])
            fitfcn += p['Z'+str(n)+'_s']**2 * (np.exp(-1*En*t) + np.exp(-1*En*(self.T-t)) + np.exp(-1*En*(self.T+t)))
            #fitfcn += p['Z'+str(n)+'_s']*np.exp(-1*En*t)
        return fitfcn
    # two point point smear source sink
    def twopt_fitfcn_ps(self, t, p):
        En = p['E0']
        En_p = 0
        fitfcn = p['Z0_p']*p['Z0_s'] * (np.exp(-1*En*t) + np.exp(-1*En*(self.T-t)) + np.exp(-1*En*(self.T+t)))
        #fitfcn = p['Z0_p']*np.exp(-1*En*t)
        for n in range(1, self.nstates):
            En += np.exp(p['E'+str(n)])
            fitfcn += p['Z'+str(n)+'_p']*p['Z'+str(n)+'_s'] * (np.exp(-1*En*t) + np.exp(-1*En*(self.T-t)) + np.exp(-1*En*(self.T+t)))
            #En_p += np.exp(p['E'+str(n)+'_n'])
            #fitfcn += -1.0*p['Z'+str(n)+'_p_n']**2*np.exp(En_p*t)
            #fitfcn += p['Z'+str(n)+'_p']*np.exp(-1*En*t)
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
            plt.title(title+' tmax stability plot')
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
            plt.title(title+' tmin stability plot')
            plt.xlabel('tmax')
            plt.ylabel(key)
            plt.xlim(x[0]-0.5, x[-1]+0.5)
            plt.legend()
        else:
            x = fittbl['tmax']
            y = np.array([data[key] for data in fittbl['post']])
            scatter_plot(x, y, title+' tmax stability plot', 'tmax (tmin='+str(fittbl['tmin'][0])+')', key, xlim=[x[0]-0.5,x[-1]+0.5])
    else: print key,':',fittbl['post'][0][key]
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
    ##Read Data
    #subset = parameters()['subset']
    #three_UU_up = read_data(subset+'_UU/spin_up')
    #three_UU_dn = read_data(subset+'_UU/spin_dn')
    #three_DD_up = read_data(subset+'_DD/spin_up')
    #three_DD_dn = read_data(subset+'_DD/spin_dn')
    #twopt_up = read_data('spin_up')
    #twopt_dn = read_data('spin_dn')
    #
    ##Read negative parity data
    #three_UU_np_up = read_data_np(subset+'_UU/spin_up')
    #three_UU_np_dn = read_data_np(subset+'_UU/spin_dn')
    #three_DD_np_up = read_data_np(subset+'_DD/spin_up')
    #three_DD_np_dn = read_data_np(subset+'_DD/spin_dn')
    #twopt_up_np = read_data_np('spin_up')
    #twopt_dn_np = read_data_np('spin_dn')
    #
    ##Parity average
    ##three_UU_up_avg = parity_avg(three_UU_up, three_UU_np_up)
    ##three_UU_dn_avg = parity_avg(three_UU_dn, three_UU_np_dn)
    #
    #
    ##Make data into gvars
    #three_UU_up_gv = make_gvars(three_UU_up)
    #three_UU_dn_gv = make_gvars(three_UU_dn)
    #three_DD_up_gv = make_gvars(three_DD_up)
    #three_DD_dn_gv = make_gvars(three_DD_dn)
    #twopt_up_gv = make_gvars(twopt_up)
    #twopt_dn_gv = make_gvars(twopt_dn)
    #
    ##Spin and isospin average
    #three_gv = ispin_avg(three_UU_up_gv, three_UU_dn_gv, three_DD_up_gv, three_DD_dn_gv, subset)
    #twopt_gv = ispin_avg(twopt_up_gv, twopt_dn_gv)
    #
    ##Calculate effective mass
    #pltxmin = parameters()['pltxmin']
    #pltxmax = parameters()['pltxmax']
    #meff = effective_mass(twopt_gv)
    #x = np.arange(len(meff))
    #yrange = find_yrange(meff, pltxmin, pltxmax)
    #plot_scatter(x, meff, 'effective mass', 'time', 'meff', xlim=[pltxmin,pltxmax], ylim=yrange)
    ##Calculate derivative of effective mass
    #dmeff = deriv_effective_mass(three_gv, twopt_gv)
    #x = np.arange(len(dmeff))
    #yrange = find_yrange(dmeff, pltxmin, pltxmax)
    #plot_scatter(x, dmeff, 'dmeff', 'time', 'dmeff', xlim=[pltxmin,pltxmax], ylim=yrange)
    #
    ##Lsqfit procedure
    #trange = read_trange()
    #prior = read_prior()
    ##Fit meff
    ##twopt_fit = fitscript(trange, twopt_gv, prior, twopt_fitfcn)
    ##Fit dmeff
    #dmeff_fit = fitscript(trange, dmeff, prior, dmeff_fitfcn)
    #plot_stability(dmeff_fit, 'ME')	

    twopt = read_data('prot_w6p1_n94')
    twopt_gv = make_gvars(twopt)
    trange = read_trange()
    prior = read_prior()
    meff = effective_mass(twopt_gv)
    pltxmin = parameters()['pltxmin']
    pltxmax = parameters()['pltxmax']
    x = np.arange(len(meff))
    yrange = find_yrange(meff, pltxmin, pltxmax)
    plot_scatter(x, meff, 'meff', 'time', 'meff', xlim=[pltxmin, pltxmax], ylim=yrange)
    twopt_fit = fitscript(trange, twopt_gv, prior, twopt_fitfcn)
    plot_stability(twopt_fit, 'E0')
 
    #Keep plots open at the end of the code
    if parameters()['plt_flag']=='on': plt.show()
    else: pass

