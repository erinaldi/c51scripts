# fit for gA of the proton
import sys
sys.path.append('$HOME/c51/scripts/')
import sqlc51lib as c51
import password_file as pwd
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
import gvar as gv
from tabulate import tabulate
import yaml
import collections
import copy

def read_gA_bs(params):
    # read data
    ens = params['LHCP_params']['ens']
    print "reading for gA ens %s" %str(ens)
    # read two point
    f = h5.File('/Users/cchang5/Physics/c51/data/lhpc_test.h5','r')
    up = f['/prot/%s/spin_up_all' %str(ens)][()]
    dn = f['/prot/%s/spin_down_all' %str(ens)][()]
    twopt_pp = c51.ispin_avg(up, dn)
    up = f['/prot/%s/spin_up_np_all' %str(ens)][()]
    dn = f['/prot/%s/spin_down_np_all' %str(ens)][()]
    twopt_np = c51.ispin_avg(up, dn)
    twopt = c51.parity_avg(twopt_pp, twopt_np, phase=-1.0)
    ## read fh correlator
    Uup = f['/prot_A3_up/%s/spin_up_all' %str(ens)][()]
    Udn = f['/prot_A3_up/%s/spin_down_all' %str(ens)][()]
    Dup = f['/prot_A3_down/%s/spin_up_all' %str(ens)][()]
    Ddn = f['/prot_A3_down/%s/spin_down_all' %str(ens)][()]
    fh_pp = c51.ispin_avg(Uup, Udn, Dup, Ddn, subset='A3')
    Uup = f['/prot_A3_up/%s/spin_up_np_all' %str(ens)][()]
    Udn = f['/prot_A3_up/%s/spin_down_np_all' %str(ens)][()]
    Dup = f['/prot_A3_down/%s/spin_up_np_all' %str(ens)][()]
    Ddn = f['/prot_A3_down/%s/spin_down_np_all' %str(ens)][()]
    fh_np = c51.ispin_avg(Uup, Udn, Dup, Ddn, subset='A3')
    fh = c51.parity_avg(fh_pp, fh_np)
    ## concatenate
    SSl = twopt[:,:,:3,:]
    PSl = twopt[:,:,3:,:]
    #SS = SSl[:,:,0,0] 
    #PS = PSl[:,:,0,0] 
    SS = np.concatenate((SSl[:,:,0,0], SSl[:,:,1,0], SSl[:,:,2,0], SSl[:,:,0,1], SSl[:,:,1,1], SSl[:,:,2,1], SSl[:,:,0,2], SSl[:,:,1,2], SSl[:,:,2,2]), axis=1)
    PS = np.concatenate((PSl[:,:,0,0], PSl[:,:,1,0], PSl[:,:,2,0], PSl[:,:,0,1], PSl[:,:,1,1], PSl[:,:,2,1], PSl[:,:,0,2], PSl[:,:,1,2], PSl[:,:,2,2]), axis=1)
    SSl = fh[:,:,:3,:]
    PSl = fh[:,:,3:,:]
    #fhSS = SSl[:,:,0,0]
    #fhPS = PSl[:,:,0,0]
    fhSS = np.concatenate((SSl[:,:,0,0], SSl[:,:,1,0], SSl[:,:,2,0], SSl[:,:,0,1], SSl[:,:,1,1], SSl[:,:,2,1], SSl[:,:,0,2], SSl[:,:,1,2], SSl[:,:,2,2]), axis=1)
    fhPS = np.concatenate((PSl[:,:,0,0], PSl[:,:,1,0], PSl[:,:,2,0], PSl[:,:,0,1], PSl[:,:,1,1], PSl[:,:,2,1], PSl[:,:,0,2], PSl[:,:,1,2], PSl[:,:,2,2]), axis=1)
    boot0 = np.concatenate((SS, PS, fhSS, fhPS), axis=1)
    return boot0

def fit_proton(params,gvboot0):
    # read data
    ens = params['LHCP_params']['ens']
    nstates = params['LHCP_params']['nstates']
    barp = params[ens]['proton']
    basak = ['G1G1', 'G1G2', 'G1G3', 'G2G1', 'G2G2', 'G2G3', 'G3G1', 'G3G2', 'G3G3'] # src snk
    mq = 'unitary'
    # make gvars
    spec = gvboot0[:len(gvboot0)/2]
    fh = gvboot0[len(gvboot0)/2:]
    T = len(fh)/(2*len(basak))
    # R(t)
    Rl = fh/spec
    # dmeff [R(t+tau) - R(t)] / tau
    dM = Rl
    #dM = (np.roll(Rl,-tau)-Rl)/float(tau) #This is needed to plot gA correctly in the effective plots, but will give wrong data to fit with.
    # plot data
    if params['flags']['plot_data']:
        SSl = spec[:len(spec)/2].reshape((len(basak),T))
        PSl = spec[len(spec)/2:].reshape((len(basak),T))
        for b in range(len(basak)):
            SS = SSl[b]
            PS = PSl[b]
            # raw correlator
            c51.scatter_plot(np.arange(len(SS)), SS, '%s %s ss' %(basak[b],str(mq)))
            c51.scatter_plot(np.arange(len(PS)), PS, '%s %s ps' %(basak[b],str(mq)))
            plt.show()
            # effective mass
            eff = c51.effective_plots(T)
            meff_ss = eff.effective_mass(SS, 1, 'log')
            meff_ps = eff.effective_mass(PS, 1, 'log')
            xlim = [2, len(meff_ss)/3-2]
            ylim = c51.find_yrange(meff_ss, xlim[0], xlim[1])
            c51.scatter_plot(np.arange(len(meff_ss)), meff_ss, '%s %s ss effective mass' %(basak[b],str(mq)), xlim = xlim, ylim = ylim)
            ylim = c51.find_yrange(meff_ps, xlim[0], xlim[1])
            c51.scatter_plot(np.arange(len(meff_ps)), meff_ps, '%s %s ps effective mass' %(basak[b],str(mq)), xlim = xlim, ylim = ylim)
            plt.show()
            # scaled correlator
            E0 = barp['priors'][1]['E0'][0]
            scaled_ss = eff.scaled_correlator(SS, E0, phase=1.0)
            scaled_ps = eff.scaled_correlator(PS, E0, phase=1.0)
            ylim = c51.find_yrange(scaled_ss, xlim[0], xlim[1])
            c51.scatter_plot(np.arange(len(scaled_ss)), scaled_ss, '%s %s ss scaled correlator (take sqrt to get Z0_s)' %(basak[b],str(mq)), xlim = xlim, ylim = ylim)
            ylim = c51.find_yrange(scaled_ps, xlim[0], xlim[1])
            c51.scatter_plot(np.arange(len(scaled_ps)), scaled_ps, '%s %s ps scaled correlator (divide by Z0_s to get Z0_p)' %(basak[b],str(mq)), xlim = xlim, ylim = ylim)
            plt.show()
    if params['flags']['fit_twopt']:
        # data already concatenated previously
        # read priors
        prior = c51.baryon_priors(barp['priors'],basak,nstates)
        ## read trange
        trange = barp['trange']
        ## fit boot0
        boot0gv = spec
        boot0p = c51.dict_of_tuple_to_gvar(prior)
        fitfcn = c51.fit_function(T,nstates)
        boot0fit = c51.fitscript_v2(trange,T,boot0gv,boot0p,fitfcn.asqtad_twopt_baryon_ss_ps,basak=basak)
        print boot0fit['rawoutput'][0]
        if params['flags']['stability_plot']:
            c51.stability_plot(boot0fit,'E0','%s' %str(mq))
            c51.stability_plot(boot0fit,'E1','%s' %str(mq))
            plt.show()
        if params['flags']['tabulate']:
            tbl_print = collections.OrderedDict()
            tbl_print['tmin'] = boot0fit['tmin']
            tbl_print['tmax'] = boot0fit['tmax']
            tbl_print['E0'] = [boot0fit['pmean'][t]['E0'] for t in range(len(boot0fit['pmean']))]
            tbl_print['dE0'] = [boot0fit['psdev'][t]['E0'] for t in range(len(boot0fit['pmean']))]
            tbl_print['E1'] = [boot0fit['pmean'][t]['E1'] for t in range(len(boot0fit['pmean']))]
            tbl_print['dE1'] = [boot0fit['psdev'][t]['E1'] for t in range(len(boot0fit['pmean']))]
            blist = []
            for b in basak:
                blist.append(b[2:])
                blist.append(b[:2])
            blist = np.unique(blist)
            for b in blist:
                tbl_print['%s_Z0s' %b] = [boot0fit['pmean'][t]['%s_Z0s'%b] for t in range(len(boot0fit['pmean']))]
                tbl_print['%s_dZ0s' %b] = [boot0fit['psdev'][t]['%s_Z0s' %b] for t in range(len(boot0fit['pmean']))]
                tbl_print['%s_Z0p' %b] = [boot0fit['pmean'][t]['%s_Z0p' %b] for t in range(len(boot0fit['pmean']))]
                tbl_print['%s_dZ0p' %b] = [boot0fit['psdev'][t]['%s_Z0p' %b] for t in range(len(boot0fit['pmean']))]
            tbl_print['chi2/dof'] = np.array(boot0fit['chi2'])/np.array(boot0fit['dof'])
            tbl_print['logGBF'] = boot0fit['logGBF']
            print tabulate(tbl_print, headers='keys')
    return 0

def fit_gA(params,gvboot0):
    # read data
    ens = params['LHCP_params']['ens']
    nstates = params['LHCP_params']['nstates']
    barp = params[ens]['proton']
    fhbp = params[ens]['gA']
    basak = ['G1G1', 'G1G2', 'G1G3', 'G2G1', 'G2G2', 'G2G3', 'G3G1', 'G3G2', 'G3G3'] # src snk
    mq = 'unitary'
    tau = 1
    # make gvars
    spec = gvboot0[:len(gvboot0)/2]
    fh = gvboot0[len(gvboot0)/2:]
    T = len(fh)/(2*len(basak))
    # R(t)
    Rl = fh/spec
    # dmeff [R(t+tau) - R(t)] / tau
    dM = Rl
    #dM = (np.roll(Rl,-tau)-Rl)/float(tau) #This is needed to plot gA correctly in the effective plots, but will give wrong data to fit with.
    # plot fh correlator
    if params['flags']['plot_fhdata']:
        fhSSl = fh[:len(fh)/2].reshape((len(basak),T))
        fhPSl = fh[len(fh)/2:].reshape((len(basak),T))
        RSSl = Rl[:len(Rl)/2].reshape((len(basak),T))
        RPSl = Rl[len(Rl)/2:].reshape((len(basak),T))
        dM_plot = (np.roll(Rl,-tau)-Rl)/float(tau)
        dMSSl = dM_plot[:len(dM_plot)/2].reshape((len(basak),T))
        dMPSl = dM_plot[len(dM_plot)/2:].reshape((len(basak),T))
        for b in range(len(basak)):
            # raw correlator dC_lambda/dlambda
            fhSS = fhSSl[b]
            fhPS = fhPSl[b]
            c51.scatter_plot(np.arange(len(fhSS)), fhSS, '%s %s fh ss' %(basak[b],str(mq)))
            c51.scatter_plot(np.arange(len(fhPS)), fhPS, '%s %s fh ps' %(basak[b],str(mq)))
            plt.show()
            # R(t)
            RSS = RSSl[b]
            RPS = RPSl[b]
            c51.scatter_plot(np.arange(len(RSS)), RSS, '%s %s R(t) ss' %(basak[b],str(mq)))
            c51.scatter_plot(np.arange(len(RPS)), RPS, '%s %s R(t) ps' %(basak[b],str(mq)))
            plt.show()
            # dmeff R(t+tau) - R(t)
            dMSS = dMSSl[b]
            dMPS = dMPSl[b]
            xlim = [0, 20]
            ylim = [0.5, 2.0]
            c51.scatter_plot(np.arange(len(dMSS)), dMSS, '%s %s [R(t+%s)-R(t)]/%s ss' %(basak[b],str(mq),str(tau),str(tau)),xlim=xlim,ylim=ylim)
            c51.scatter_plot(np.arange(len(dMPS)), dMPS, '%s %s [R(t+%s)-R(t)]/%s ps' %(basak[b],str(mq),str(tau),str(tau)),xlim=xlim,ylim=ylim)
            if False:
                for i in range(len(dMSS)):
                    print i,',',dMSS[i].mean,',',dMSS[i].sdev,',',dMPS[i].mean,',',dMPS[i].sdev
            plt.show()
    # fit fh correlator
    if params['flags']['fit_gA']:
        # data concatenated previously
        # read priors
        prior = c51.fhbaryon_priors(barp['priors'],fhbp['priors'],basak,nstates)
        #print prior
        # read trange
        trange = barp['trange']
        fhtrange = fhbp['trange']
        # fit boot0
        boot0gv = np.concatenate((spec, dM))
        boot0p = c51.dict_of_tuple_to_gvar(prior)
        fitfcn = c51.fit_function(T,nstates,tau)
        boot0fit = c51.fitscript_v3(trange,fhtrange,T,boot0gv,boot0p,fitfcn.asqtad_baryon_rt_ss_ps,basak=basak)
        #print boot0fit['rawoutput'][0]
        #print {k: boot0fit['p0'][0][k] for k in [bk for n in range(nstates) for bk in barp['priors'][n+1].keys()]}
        if False: # plot R(t+1) - R(t)
            posterior = boot0fit['post'][0]
            x = np.arange(20)
            ssline, psline = fitfcn.baryon_rt_fitline(x,posterior)
            for i in range(len(ssline)):
                print x[i],',',ssline[i].mean,',',ssline[i].sdev,',',psline[i].mean,',',psline[i].sdev
        if params['flags']['stability_plot']:
            c51.stability_plot(boot0fit,'E0','%s' %str(mq))
            c51.stability_plot(boot0fit,'gA00','%s' %str(mq))
            plt.show()
        if params['flags']['tabulate']:
            tbl_print = collections.OrderedDict()
            tbl_print['tmin'] = boot0fit['tmin']
            tbl_print['tmax'] = boot0fit['tmax']
            tbl_print['fhtmin'] = boot0fit['fhtmin']
            tbl_print['fhtmax'] = boot0fit['fhtmax']
            #tbl_print['E0'] = [boot0fit['pmean'][t]['E0'] for t in range(len(boot0fit['pmean']))]
            #tbl_print['dE0'] = [boot0fit['psdev'][t]['E0'] for t in range(len(boot0fit['pmean']))]
            tbl_print['gA00'] = [boot0fit['pmean'][t]['gA00'] for t in range(len(boot0fit['pmean']))]
            tbl_print['dgA00'] = [boot0fit['psdev'][t]['gA00'] for t in range(len(boot0fit['pmean']))]
            #blist = []
            #for b in params['gA_fit']['basak']:
            #    blist.append(b[2:])
            #    blist.append(b[:2])
            #blist = np.unique(blist)
            #for b in blist:
            #    tbl_print['%s_Z0s' %b] = [boot0fit['pmean'][t]['%s_Z0s'%b] for t in range(len(boot0fit['pmean']))]
            #    tbl_print['%s_dZ0s' %b] = [boot0fit['psdev'][t]['%s_Z0s' %b] for t in range(len(boot0fit['pmean']))]
            #    tbl_print['%s_Z0p' %b] = [boot0fit['pmean'][t]['%s_Z0p' %b] for t in range(len(boot0fit['pmean']))]
            #    tbl_print['%s_dZ0p' %b] = [boot0fit['psdev'][t]['%s_Z0p' %b] for t in range(len(boot0fit['pmean']))]
            tbl_print['chi2/dof'] = np.array(boot0fit['chi2'])/np.array(boot0fit['dof'])
            tbl_print['chi2'] = boot0fit['chi2']
            tbl_print['chi2f'] = boot0fit['chi2f']
            tbl_print['logGBF'] = boot0fit['logGBF']
            print tabulate(tbl_print, headers='keys')
        if params['flags']['csvformat']:
            for t in range(len(boot0fit['fhtmin'])):
                print nstates,',',boot0fit['tmin'][t],',',boot0fit['tmax'][t],',',boot0fit['fhtmin'][t],',',boot0fit['fhtmax'][t],',',boot0fit['pmean'][t]['gA00'],',',boot0fit['psdev'][t]['gA00'],',',(np.array(boot0fit['chi2'])/np.array(boot0fit['dof']))[t],',',(np.array(boot0fit['chi2'])/np.array(boot0fit['chi2f']))[t],',',boot0fit['logGBF'][t]
        return {'gA_fit': boot0fit['rawoutput'][0]}
    return 0

if __name__=='__main__':
    # read master
    f = open('./LHCP_master.yml','r')
    params = yaml.load(f)
    # fit gA
    boot0 = read_gA_bs(params)
    gvboot0 = c51.make_gvars(boot0)
    fit_proton(params,gvboot0)
    res = fit_gA(params,gvboot0)
    print res['gA']
