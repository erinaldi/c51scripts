# fit for gA and gV of the proton
import sys
sys.path.append('$HOME/c51/scripts/')
import sqlc51lib as c51
import calsql as sql
import password_file as pwd
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
import gvar as gv
import multiprocessing as multi
from tabulate import tabulate
import yaml
import collections
import copy
import tqdm
import cPickle as pickle

def read_gA_bs(psql,params,twopt=False):
    # read data
    mq = params['gA_fit']['ml']
    basak = copy.deepcopy(params['gA_fit']['basak'])
    tag = params['gA_fit']['ens']['tag']
    stream = params['gA_fit']['ens']['stream']
    Nbs = params['gA_fit']['nbs']
    Mbs = params['gA_fit']['mbs']
    nstates = params['gA_fit']['nstates']
    tau = params['gA_fit']['tau']
    barp = params[tag]['proton'][mq]
    if twopt:
        pass
    else:
        fhbp = params[tag]['gA'][mq]
        gVbp = params[tag]['gV'][mq]
    print "reading for twopt, gA, gV mq %s, basak %s, ens %s%s, Nbs %s, Mbs %s" %(str(mq),str(basak),str(tag),str(stream),str(Nbs),str(Mbs))
    # read two point
    SSl = np.array([psql.data('dwhisq_corr_baryon',idx) for idx in [barp['meta_id']['SS'][i] for i in basak]])
    PSl = np.array([psql.data('dwhisq_corr_baryon',idx) for idx in [barp['meta_id']['PS'][i] for i in basak]])
    T = len(SSl[0,0])
    if twopt:
        pass
    else:
        # read fh correlator
        fhSSl = np.array([psql.data('dwhisq_fhcorr_baryon',idx) for idx in [fhbp['meta_id']['SS'][i] for i in basak]])
        fhPSl = np.array([psql.data('dwhisq_fhcorr_baryon',idx) for idx in [fhbp['meta_id']['PS'][i] for i in basak]])
        # read gV correlator
        gVSSl = np.array([psql.data('dwhisq_fhcorr_baryon',idx) for idx in [gVbp['meta_id']['SS'][i] for i in basak]])
        gVPSl = np.array([psql.data('dwhisq_fhcorr_baryon',idx) for idx in [gVbp['meta_id']['PS'][i] for i in basak]])
    # concatenate and make gvars to preserve correlations
    SS = SSl[0]
    PS = PSl[0]
    for i in range(len(SSl)-1): # loop over basak operators
        SS = np.concatenate((SS,SSl[i+1]),axis=1)
        PS = np.concatenate((PS,PSl[i+1]),axis=1)
    if twopt:
        pass
    else:
        fhSS = fhSSl[0]
        fhPS = fhPSl[0]
        for i in range(len(fhSSl)-1):
            fhSS = np.concatenate((fhSS,fhSSl[i+1]),axis=1)
            fhPS = np.concatenate((fhPS,fhPSl[i+1]),axis=1)
        gVSS = gVSSl[0]
        gVPS = gVPSl[0]
        for i in range(len(gVSSl)-1):
            gVSS = np.concatenate((gVSS,gVSSl[i+1]),axis=1)
            gVPS = np.concatenate((gVPS,gVPSl[i+1]),axis=1)
        boot0 = np.concatenate((SS, PS, fhSS, fhPS, gVSS, gVPS), axis=1)
    boot0twopt = np.concatenate((SS, PS), axis=1)
    if twopt:
        return boot0twopt
    else:
        return boot0

def fit_proton(psql,params,gvboot0,twopt=False):
    # read data
    mq = params['gA_fit']['ml']
    basak = copy.deepcopy(params['gA_fit']['basak'])
    tag = params['gA_fit']['ens']['tag']
    stream = params['gA_fit']['ens']['stream']
    Nbs = params['gA_fit']['nbs']
    Mbs = params['gA_fit']['mbs']
    nstates = params['gA_fit']['nstates']
    tau = params['gA_fit']['tau']
    barp = params[tag]['proton'][mq]
    # make gvars
    if twopt:
        spec = gvboot0
    else:
        spec = gvboot0[:len(gvboot0)/3]
    T = len(spec)/(2*len(basak))
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
            xlim = [2, len(meff_ss)/3]
            ylim = c51.find_yrange(meff_ss, xlim[0], xlim[1])
            c51.scatter_plot(np.arange(len(meff_ss)), meff_ss, '%s %s ss effective mass' %(basak[b],str(mq)), xlim = xlim, ylim = ylim)
            ylim = c51.find_yrange(meff_ps, xlim[0], xlim[1])
            c51.scatter_plot(np.arange(len(meff_ps)), meff_ps, '%s %s ps effective mass' %(basak[b],str(mq)), xlim = xlim, ylim = ylim)
            #print stuff
            #print 'meff_ss'
            #f_meff = open('/Users/cchang5/Documents/Papers/FH/c51_p1/paper/figures/meff_ss.csv', 'w+')
            #string = 't_meff, meff, +-\n'
            #for t in range(len(meff_ss)):
            #    string += '%s, %s, %s\n' %(str(t), str(meff_ss[t].mean), str(meff_ss[t].sdev))
            #f_meff.write(string)
            #f_meff.flush()
            #f_meff.close()
            #print 'meff_ps'
            #for t in range(len(meff_ps)):
            #    print t, meff_ps[t].mean, meff_ps[t].sdev
            plt.show()
            # scaled correlator
            E0 = barp['priors'][1]['E0'][0]
            scaled_ss = np.sqrt(eff.scaled_correlator_v2(SS, E0, phase=1.0))
            scaled_ps = eff.scaled_correlator_v2(PS, E0, phase=1.0)/scaled_ss
            ylim = c51.find_yrange(scaled_ss, xlim[0], xlim[1])
            c51.scatter_plot(np.arange(len(scaled_ss)), scaled_ss, '%s %s ss scaled correlator (Z0_s)' %(basak[b],str(mq)), xlim = xlim, ylim = ylim)
            ylim = c51.find_yrange(scaled_ps, xlim[0], xlim[1])
            c51.scatter_plot(np.arange(len(scaled_ps)), scaled_ps, '%s %s ps scaled correlator (Z0_p)' %(basak[b],str(mq)), xlim = xlim, ylim = ylim)
            #print stuff
            #f_scaled = open('/Users/cchang5/Documents/Papers/FH/c51_p1/paper/figures/scaled_corr.csv', 'w+')
            #string = 't_scaled, scaled_ss, +-, scaled_ps, +-\n'
            #for t in range(len(scaled_ss)):
            #    string += '%s, %s, %s, %s, %s\n' %(t, scaled_ss[t].mean, scaled_ss[t].sdev, scaled_ps[t].mean, scaled_ps[t].sdev)
            #f_scaled.write(string)
            #f_scaled.flush()
            #f_scaled.close()
            f_plot = open('./fh_fitline/%s_zeff_data.csv' %(tag), 'w+')
            string = 't, zss, +-, zps, +-\n'
            for i in range(len(scaled_ss)):
                string += '%s, %s, %s, %s, %s\n' %(i, scaled_ss[i].mean, scaled_ss[i].sdev, scaled_ps[i].mean, scaled_ps[i].sdev)
            f_plot.write(string)
            f_plot.flush()
            f_plot.close()
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
        boot0fit = c51.fitscript_v2(trange,T,boot0gv,boot0p,fitfcn.dwhisq_twopt_ss_ps,basak=params['gA_fit']['basak'])
        print boot0fit['rawoutput'][0]
        if params['flags']['fitline_plot']:
            p = boot0fit['rawoutput'][0].p
            t = np.linspace(0, 30, 1000)
            b = params['gA_fit']['basak'][0]
            # output fit curve
            ss = fitfcn.dwhisq_twopt(t,p,b,'s','s')
            ps = fitfcn.dwhisq_twopt(t,p,b,'p','s')
            scaled_ss = np.sqrt(ss*np.exp(p['E0']*t))
            scaled_ps = ps*np.exp(p['E0']*t)/scaled_ss
            meffss = np.log(ss/np.roll(ss,-1))/t[1]
            meffps = np.log(ps/np.roll(ps,-1))/t[1]
            f_plot = open('./fh_fitline/%s_twopt_2.csv' %(tag), 'w+')
            string = 't, css, +-, cps, +-\n'
            for i in range(len(t)):
                string += '%s, %s, %s, %s, %s\n' %(t[i], meffss[i].mean, meffss[i].sdev, meffps[i].mean, meffps[i].sdev)
            f_plot.write(string)
            f_plot.flush()
            f_plot.close()
            f_plot = open('./fh_fitline/%s_zeff_2.csv' %(tag), 'w+')
            string = 't, zss, +-, zps, +-\n'
            for i in range(len(t)):
                string += '%s, %s, %s, %s, %s\n' %(t[i], scaled_ss[i].mean, scaled_ss[i].sdev, scaled_ps[i].mean, scaled_ps[i].sdev)
            f_plot.write(string)
            f_plot.flush()
            f_plot.close()
            #f_plot = open('./fh_fitline/%s_scaled2pt.csv' %(tag), 'w+')
            # data scaled two point: propagate E0 error
            #SSl = spec[:len(spec)/2].reshape((len(basak),T))[0]
            #PSl = spec[len(spec)/2:].reshape((len(basak),T))[0]
            #t = np.linspace(0,T,T+1)
            #scaled_ss = np.sqrt(SSl*np.exp(p['E0']*t))
            #scaled_ps = PSl*np.exp(p['E0']*t)/scaled_ss
            #string = 't, zss, +-, zps, +-\n'
            #for i in range(len(t)):
            #    string += '%s, %s, %s, %s, %s\n' %(t[i], scaled_ss[i].mean, scaled_ss[i].sdev, scaled_ps[i].mean, scaled_ps[i].sdev)
            #f_plot = open('/Users/cchang5/Documents/Papers/FH/c51_p1/paper/figures/scaled_corr.csv', 'w+')
            #f_plot.write(string)
            #f_plot.flush()
            #f_plot.close()
        if params['flags']['stability_plot']:
            c51.stability_plot(boot0fit,'E0','%s' %str(mq))
            plt.show()
        if params['flags']['tabulate']:
            tbl_print = collections.OrderedDict()
            tbl_print['tmin'] = boot0fit['tmin']
            tbl_print['tmax'] = boot0fit['tmax']
            tbl_print['E0'] = [boot0fit['pmean'][t]['E0'] for t in range(len(boot0fit['pmean']))]
            tbl_print['dE0'] = [boot0fit['psdev'][t]['E0'] for t in range(len(boot0fit['pmean']))]
            blist = []
            for b in params['gA_fit']['basak']:
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
            tbl_print['logGBF'] = boot0fit['logGBF']
            tbl_print['Q'] = boot0fit['Q']
            print tabulate(tbl_print, headers='keys')
            print "tmin, E0%s, +-, chi2dof%s, Q%s, logGBF%s" %(nstates, nstates, nstates, nstates)
            for i in range(len(boot0fit['tmin'])):
                print '%s, %s, %s, %s, %s, %s' %(str(boot0fit['tmin'][i]), str(boot0fit['pmean'][i]['E0']), str(boot0fit['psdev'][i]['E0']), boot0fit['chi2'][i]/boot0fit['dof'][i], boot0fit['Q'][i], boot0fit['logGBF'][i])
        return {'nucleon_fit': boot0fit['rawoutput'][0]}
    return 0

def fit_gA(psql,params,gvboot0):
    # read data
    mq = params['gA_fit']['ml']
    basak = copy.deepcopy(params['gA_fit']['basak'])
    tag = params['gA_fit']['ens']['tag']
    stream = params['gA_fit']['ens']['stream']
    Nbs = params['gA_fit']['nbs']
    Mbs = params['gA_fit']['mbs']
    nstates = params['gA_fit']['nstates']
    fhstates = params['gA_fit']['fhstates']
    gVstates = params['gA_fit']['gVstates']
    tau = params['gA_fit']['tau']
    barp = params[tag]['proton'][mq]
    fhbp = params[tag]['gA'][mq]
    gVbp = params[tag]['gV'][mq]
    #print "nstates: %s" %nstates
    #print "fhstates: %s" %fhstates
    # make gvars
    spec = gvboot0[:len(gvboot0)/3]
    fh = gvboot0[len(gvboot0)/3:len(gvboot0)*2/3]
    gV = gvboot0[len(gvboot0)*2/3:]
    T = len(fh)/(2*len(basak))
    # R(t)
    Rl = fh/spec
    dM = (np.roll(Rl,-tau)-Rl)/float(tau) #This is needed to plot gA correctly in the effective plots, but will give wrong data to fit with.
    gVRl = gV/spec
    gVdM = (np.roll(gVRl,-tau)-gVRl)/float(tau)
    # plot fh correlator
    if params['flags']['plot_fhdata']:
        fhSSl = fh[:len(fh)/2].reshape((len(basak),T))
        fhPSl = fh[len(fh)/2:].reshape((len(basak),T))
        RSSl = Rl[:len(Rl)/2].reshape((len(basak),T))
        RPSl = Rl[len(Rl)/2:].reshape((len(basak),T))
        dMSSl = dM[:len(dM)/2].reshape((len(basak),T))
        dMPSl = dM[len(dM)/2:].reshape((len(basak),T))
        gVSSl = gV[:len(gV)/2].reshape((len(basak),T))
        gVPSl = gV[len(gV)/2:].reshape((len(basak),T))
        gVRSSl = gVRl[:len(gVRl)/2].reshape((len(basak),T))
        gVRPSl = gVRl[len(gVRl)/2:].reshape((len(basak),T))
        gVdMSSl = gVdM[:len(gVdM)/2].reshape((len(basak),T))
        gVdMPSl = gVdM[len(gVdM)/2:].reshape((len(basak),T))
        for b in range(len(basak)):
            # ground state nucleon mass prior central value
            E0 = barp['priors'][1]['E0'][0]
            # raw correlator dC_lambda/dlambda
            fhSS = fhSSl[b]
            fhPS = fhPSl[b]
            gVSS = gVSSl[b]
            gVPS = gVPSl[b]
            #c51.scatter_plot(np.arange(len(fhSS)), fhSS, '%s %s fh ss' %(basak[b],str(mq)))
            #c51.scatter_plot(np.arange(len(fhPS)), fhPS, '%s %s fh ps' %(basak[b],str(mq)))
            print "%s fhSS[1]*exp(E0):" %basak[b], fhSS[1]*np.exp(E0) #, "fhSS[1]:", fhSS[1]
            print "%s fhPS[1]*exp(E0):" %basak[b], fhPS[1]*np.exp(E0) #, "fhPS[1]:", fhPS[1]
            print "%s gVSS[1]*exp(E0):" %basak[b], gVSS[1]*np.exp(E0) #, "fhSS[1]:", fhSS[1]
            print "%s gVPS[1]*exp(E0):" %basak[b], gVPS[1]*np.exp(E0) #, "fhPS[1]:", fhPS[1]
            # dmeff R(t+tau) - R(t)
            dMSS = dMSSl[b]
            dMPS = dMPSl[b]
            gVdMSS = gVdMSSl[b]
            gVdMPS = gVdMPSl[b]
            xlim = [0, 20]
            ylim = [0.5, 2.0]
            c51.scatter_plot(np.arange(len(dMSS)), dMSS, '%s %s [R(t+%s)-R(t)]/%s ss' %(basak[b],str(mq),str(tau),str(tau)),xlim=xlim,ylim=ylim)
            c51.scatter_plot(np.arange(len(dMPS)), dMPS, '%s %s [R(t+%s)-R(t)]/%s ps' %(basak[b],str(mq),str(tau),str(tau)),xlim=xlim,ylim=ylim)
            c51.scatter_plot(np.arange(len(gVdMSS)), gVdMSS, '%s %s [R(t+%s)-R(t)]/%s gVss' %(basak[b],str(mq),str(tau),str(tau)),xlim=xlim,ylim=ylim)
            c51.scatter_plot(np.arange(len(gVdMPS)), gVdMPS, '%s %s [R(t+%s)-R(t)]/%s gVps' %(basak[b],str(mq),str(tau),str(tau)),xlim=xlim,ylim=ylim)
            if False:
                for i in range(len(dMSS)):
                    print i,',',dMSS[i].mean,',',dMSS[i].sdev,',',dMPS[i].mean,',',dMPS[i].sdev
            plt.show()
    # fit fh correlator
    if params['flags']['fit_gA']:
        # data concatenated previously
        # read priors
        prior = c51.fhbaryon_priors_v2(barp['priors'],fhbp['priors'],gVbp['priors'],basak,nstates,fhstates,gVstates)
        # read init
        try:
            #print "found init file"
            tstring = '%s_%s_%s_%s_%s_%s' %(barp['trange']['tmin'][0], barp['trange']['tmax'][0], fhbp['trange']['tmin'][0], fhbp['trange']['tmax'][0], gVbp['trange']['tmin'][0], gVbp['trange']['tmax'][0])
            f_init = open('./fh_posterior/gAgV_%s_%s.yml' %(tag,tstring), 'r')
            init = yaml.load(f_init)
            f_init.close()
        except:
            init = None
        #print prior
        # read trange
        trange = barp['trange']
        fhtrange = fhbp['trange']
        gVtrange = gVbp['trange']
        # fit boot0
        boot0gv = np.concatenate((spec,dM,gVdM)) # dM
        #boot0gv = gvboot0 # fh
        boot0p = c51.dict_of_tuple_to_gvar(prior)
        fitfcn = c51.fit_function(T,nstates,fhstates,gVstates,tau)
        boot0fit = c51.fitscript_v4(trange,fhtrange,gVtrange,T,boot0gv,boot0p,fitfcn.dwhisq_dm_gVdm_ss_ps,basak=params['gA_fit']['basak'],init=init,bayes=params['flags']['bayes']) # dM
        #boot0fit = c51.fitscript_v4(trange,fhtrange,gVtrange,T,boot0gv,boot0p,fitfcn.dwhisq_fh_gVfh_ss_ps,basak=params['gA_fit']['basak'],init=init,bayes=params['flags']['bayes'])
        if params['flags']['rawoutput']:
            print boot0fit['rawoutput'][0]
        if params['flags']['fitline_plot']:
            p = boot0fit['rawoutput'][0].p
            t = np.linspace(0, 30, 1000)
            b = params['gA_fit']['basak'][0]
            # output fit curve
            ss = fitfcn.dwhisq_twopt(t,p,b,'s','s')
            ps = fitfcn.dwhisq_twopt(t,p,b,'p','s')
            yss = fitfcn.dwhisq_dm(t,p,b,'s','s')
            yps = fitfcn.dwhisq_dm(t,p,b,'p','s')
            vss = fitfcn.dwhisq_dm(t,p,b,'s','s',True)
            vps = fitfcn.dwhisq_dm(t,p,b,'p','s',True)
            f_plot = open('./fh_fitline/%s.csv' %(tag), 'w+')
            string = 't, yss, +-, yps, +-, vss, +-, vps, +-\n'
            for i in range(len(t)):
                string += '%s, %s, %s, %s, %s, %s, %s, %s, %s\n' %(t[i], yss[i].mean, yss[i].sdev, yps[i].mean, yps[i].sdev, vss[i].mean, vss[i].sdev, vps[i].mean, vps[i].sdev)
            f_plot.write(string)
            f_plot.flush()
            f_plot.close()
            meffss = np.log(ss/np.roll(ss,-1))/t[1]
            meffps = np.log(ps/np.roll(ps,-1))/t[1]
            f_plot = open('./fh_fitline/%s_twopt.csv' %(tag), 'w+')
            string = 't, css, +-, cps, +-\n'
            for i in range(len(t)):
                string += '%s, %s, %s, %s, %s\n' %(t[i], meffss[i].mean, meffss[i].sdev, meffps[i].mean, meffps[i].sdev)
            f_plot.write(string)
            f_plot.flush()
            f_plot.close()
            scaled_ss = np.sqrt(ss*np.exp(p['E0']*t))
            scaled_ps = ps*np.exp(p['E0']*t)/scaled_ss
            #scaled_ss = np.sqrt(ss*np.exp(meffss))
            #scaled_ps = ps*np.exp(meffps)/scaled_ss
            f_plot = open('./fh_fitline/%s_zeff.csv' %(tag), 'w+')
            string = 't, zss, +-, zps, +-\n'
            for i in range(len(t)):
                string += '%s, %s, %s, %s, %s\n' %(t[i], scaled_ss[i].mean, scaled_ss[i].sdev, scaled_ps[i].mean, scaled_ps[i].sdev)
            f_plot.write(string)
            f_plot.flush()
            f_plot.close()
            # output data
            #dMSSl = fh[:len(fh)/2].reshape((len(basak),T))[0]
            #dMPSl = fh[len(fh)/2:].reshape((len(basak),T))[0]
            dMSSl = dM[:len(dM)/2].reshape((len(basak),T))[0]
            dMPSl = dM[len(dM)/2:].reshape((len(basak),T))[0]
            gVdMSSl = gVdM[:len(gVdM)/2].reshape((len(basak),T))[0]
            gVdMPSl = gVdM[len(gVdM)/2:].reshape((len(basak),T))[0]
            t = np.linspace(0,30,31)
            #f_data = open('./fh_fitline/%s_fh_dat.csv' %(tag), 'w+')
            f_data = open('./fh_fitline/%s_dat.csv' %(tag), 'w+')
            #f_data = open('../plots/gA/autocorrelation/%s_bin%s.csv' %(tag,params['gA_fit']['bin']), 'w+')
            string = 't_dat, yss_dat, +-, yps_dat, +-, vss_dat, +-, vps_dat, +-\n'
            for i in range(len(t)):
                string += '%s, %s, %s, %s, %s, %s, %s, %s, %s\n' %(t[i], dMSSl[i].mean, dMSSl[i].sdev, dMPSl[i].mean, dMPSl[i].sdev, gVdMSSl[i].mean, gVdMSSl[i].sdev, gVdMPSl[i].mean, gVdMPSl[i].sdev)
            f_data.write(string)
            f_data.flush()
            f_data.close()
            SSl = spec[:len(spec)/2].reshape((len(basak),T))[0]
            PSl = spec[len(spec)/2:].reshape((len(basak),T))[0]
            meffSS = np.log(SSl/np.roll(SSl,-1))
            meffPS = np.log(PSl/np.roll(PSl,-1))
            f_data = open('./fh_fitline/%s_twopt_dat.csv' %(tag), 'w+')
            string = 't_dat, ss_dat, +-, ps_dat, +-\n'
            for i in range(len(t)):
                string += '%s, %s, %s, %s, %s\n' %(t[i], meffSS[i].mean, meffSS[i].sdev, meffPS[i].mean, meffPS[i].sdev)
            f_data.write(string)
            f_data.flush()
            f_data.close()
        if params['flags']['boot0_update']:
            fh_post = boot0fit['rawoutput'][0].pmean
            fh_dump = dict()
            for k in fh_post.keys():
                fh_dump[k] = float(fh_post[k])
            tstring = '%s_%s_%s_%s_%s_%s' %(barp['trange']['tmin'][0], barp['trange']['tmax'][0], fhbp['trange']['tmin'][0], fhbp['trange']['tmax'][0], gVbp['trange']['tmin'][0], gVbp['trange']['tmax'][0])
            f_dump = open('./fh_posterior/gAgV_%s_%s.yml' %(tag,tstring), 'w+')
            yaml.dump(fh_dump, f_dump)
            f_dump.flush()
            f_dump.close()
        if params['flags']['stability_plot']:
            c51.stability_plot(boot0fit,'E0','%s' %str(mq))
            c51.stability_plot(boot0fit,'gA00','%s' %str(mq))
            c51.stability_plot(boot0fit,'gV00','%s' %str(mq))
            plt.show()
        if params['flags']['tabulate']:
            tbl_print = collections.OrderedDict()
            tbl_print['tmin'] = boot0fit['tmin']
            tbl_print['tmax'] = boot0fit['tmax']
            tbl_print['fhtmin'] = boot0fit['fhtmin']
            tbl_print['fhtmax'] = boot0fit['fhtmax']
            tbl_print['gVtmin'] = boot0fit['gVtmin']
            tbl_print['gVtmax'] = boot0fit['gVtmax']
            tbl_print['E0'] = [boot0fit['pmean'][t]['E0'] for t in range(len(boot0fit['pmean']))]
            tbl_print['dE0'] = [boot0fit['psdev'][t]['E0'] for t in range(len(boot0fit['pmean']))]
            tbl_print['gA00'] = [boot0fit['pmean'][t]['gA00'] for t in range(len(boot0fit['pmean']))]
            tbl_print['dgA00'] = [boot0fit['psdev'][t]['gA00'] for t in range(len(boot0fit['pmean']))]
            tbl_print['gV00'] = [boot0fit['pmean'][t]['gV00'] for t in range(len(boot0fit['pmean']))]
            tbl_print['dgV00'] = [boot0fit['psdev'][t]['gV00'] for t in range(len(boot0fit['pmean']))]
            tbl_print['gA/gV'] = [(boot0fit['rawoutput'][t].p['gA00']/boot0fit['rawoutput'][t].p['gV00']).mean for t in range(len(boot0fit['rawoutput']))]
            tbl_print['dgA/gV'] = [(boot0fit['rawoutput'][t].p['gA00']/boot0fit['rawoutput'][t].p['gV00']).sdev for t in range(len(boot0fit['rawoutput']))]
            tbl_print['chi2/dof'] = np.array(boot0fit['chi2'])/np.array(boot0fit['dof'])
            tbl_print['chi2'] = boot0fit['chi2']
            tbl_print['chi2f'] = boot0fit['chi2f']
            tbl_print['logGBF'] = boot0fit['logGBF']
            tbl_print['Q'] = boot0fit['Q']
            print tabulate(tbl_print, headers='keys')
            print "tmin, %s%sE0%s, +-, %s%sgA%s, +-, %s%sgV%s, +-, %s%sgAgV%s, +-, %s%schi2dof%s, %s%sQ%s, %s%slogGBF%s" %(gVstates, fhstates, nstates, gVstates, fhstates, nstates, gVstates, fhstates, nstates, gVstates, fhstates, nstates, gVstates, fhstates, nstates, gVstates, fhstates, nstates, gVstates, fhstates, nstates)
            for i in range(len(boot0fit['fhtmin'])):
                gAgV = boot0fit['rawoutput'][i].p['gA00']/boot0fit['rawoutput'][i].p['gV00']
                print '%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s' %(str(boot0fit['gVtmin'][i]), str(boot0fit['pmean'][i]['E0']), str(boot0fit['psdev'][i]['E0']), str(boot0fit['pmean'][i]['gA00']), str(boot0fit['psdev'][i]['gA00']), str(boot0fit['pmean'][i]['gV00']), str(boot0fit['psdev'][i]['gV00']), str(gAgV.mean), str(gAgV.sdev), boot0fit['chi2'][i]/boot0fit['dof'][i], boot0fit['Q'][i], boot0fit['logGBF'][i])
        if params['flags']['csvformat']:
            for t in range(len(boot0fit['fhtmin'])):
                print nstates,',',boot0fit['tmin'][t],',',boot0fit['tmax'][t],',',boot0fit['fhtmin'][t],',',boot0fit['fhtmax'][t],',',boot0fit['pmean'][t]['gA00'],',',boot0fit['psdev'][t]['gA00'],',',(np.array(boot0fit['chi2'])/np.array(boot0fit['dof']))[t],',',(np.array(boot0fit['chi2'])/np.array(boot0fit['chi2f']))[t],',',boot0fit['logGBF'][t]
        return {'gA_fit': boot0fit['rawoutput'][0]}
    return 0

if __name__=='__main__':
    # read master
    f = open('./fh.yml', 'r')
    params = yaml.load(f)
    f.close()
    # log in sql
    psqlpwd = pwd.passwd()
    psql = sql.pysql('cchang5','cchang5',psqlpwd)
    # fit gA
    if params['flags']['fit_gA']:
        boot0 = read_gA_bs(psql,params)
    else:
        boot0 = read_gA_bs(psql,params,twopt=True)
    # bin
    boot0 = np.array(gv.dataset.bin_data(boot0,binsize=params['gA_fit']['bin']))
    print "boot0 shape:", np.shape(boot0)
    gvboot0 = c51.make_gvars(boot0)
    if params['flags']['fit_twopt']:
        fit_proton(psql,params,gvboot0,twopt=True)
    if params['flags']['fit_gA']:
        res = fit_gA(psql,params,gvboot0)
    # bootstrap gA
    if params['flags']['bootstrap']:
        params['flags']['bayes'] = False
        params['flags']['csvformat'] = False
        params['flags']['tabulate'] = False
        params['flags']['rawoutput'] = False
        Nbs = params['gA_fit']['nbs']
        Nbsmin = params['gA_fit']['nbsmin']
        Mbs = params['gA_fit']['mbs']
        # update boot0
        params['flags']['boot0_update'] = True
        res = fit_gA(psql,params,gvboot0)
        params['flags']['boot0_update'] = False
        print "updated boot0"
        # read bootstrap
        bslist = psql.fetch_draws(params['gA_fit']['ens']['tag'], Nbs, Mbs)
        # read data
        b = params['gA_fit']['basak'][0]
        nstates = params['gA_fit']['nstates']
        fhstates = params['gA_fit']['fhstates']
        gVstates = params['gA_fit']['gVstates']
        tau = params['gA_fit']['tau']
        mq = params['gA_fit']['ml']
        basak = copy.deepcopy(params['gA_fit']['basak'])
        tag = params['gA_fit']['ens']['tag']
        stream = params['gA_fit']['ens']['stream']
        nstates = params['gA_fit']['nstates']
        fhstates = params['gA_fit']['fhstates']
        gVstates = params['gA_fit']['gVstates']
        tau = params['gA_fit']['tau']
        barp = params[tag]['proton'][mq]
        fhbp = params[tag]['gA'][mq]
        gVbp = params[tag]['gV'][mq]
        filetag = "%s_%s_%s_%s_%s_%s_%s_%s_%s" %(nstates,fhstates,gVstates,barp['trange']['tmin'][0], barp['trange']['tmax'][0], fhbp['trange']['tmin'][0], fhbp['trange']['tmax'][0], gVbp['trange']['tmin'][0], gVbp['trange']['tmax'][0])
        fh = gvboot0[len(gvboot0)/3:len(gvboot0)*2/3]
        gV = gvboot0[len(gvboot0)*2/3:]
        T = len(fh)/2
        fitfcn = c51.fit_function(T,nstates,fhstates,gVstates,tau)
        x = np.linspace(0, T/2, 100*T/2)
        fit2ptss = []
        fit2ptps = []
        fit3ptss = []
        fit3ptps = []
        fitgVss = []
        fitgVps = []
        fitgAgV = []
        fitgA = []
        fitgV = []
        fitE0 = []
        p = res['gA_fit'].pmean
        b02ptss = fitfcn.dwhisq_twopt(x,p,b,'s','s')
        b02ptps = fitfcn.dwhisq_twopt(x,p,b,'p','s')
        b0mss = np.log(b02ptss/np.roll(b02ptss,-100))/x[100]
        b0mps = np.log(b02ptps/np.roll(b02ptps,-100))/x[100]
        b03ptss = fitfcn.dwhisq_dm(x,p,b,'s','s')
        b03ptps = fitfcn.dwhisq_dm(x,p,b,'p','s')
        gV3ptss = fitfcn.dwhisq_dm(x,p,b,'s','s',True)
        gV3ptps = fitfcn.dwhisq_dm(x,p,b,'p','s',True)
        # write fit_meta
        psql.insert_gA_v1_meta(params)
        #for sfit in tqdm.tqdm(res['gA_fit'].bootstrapped_fit_iter(n=4000)):
        for g in tqdm.tqdm(range(Nbsmin,Nbs+1)): # THIS FITS boot0 ALSO
            draws = np.array(bslist[g][1])
            bootn = boot0[draws,:]
            gvboot0 = c51.make_gvars(bootn)
            sfit = fit_gA(psql,params,gvboot0)['gA_fit']
            p = sfit.pmean
            p['Q'] = sfit.Q
            psql.insert_gA_v1_result(params,g,len(draws),p)
            ss2pt = fitfcn.dwhisq_twopt(x,p,b,'s','s')
            ps2pt = fitfcn.dwhisq_twopt(x,p,b,'p','s')
            mss = np.log(ss2pt/np.roll(ss2pt,-100))/x[100]
            mps = np.log(ps2pt/np.roll(ps2pt,-100))/x[100]
            fit2ptss.append(mss)
            fit2ptps.append(mps)
            fit3ptss.append(fitfcn.dwhisq_dm(x,p,b,'s','s'))
            fit3ptps.append(fitfcn.dwhisq_dm(x,p,b,'p','s'))
            fitgVss.append(fitfcn.dwhisq_dm(x,p,b,'s','s',True))
            fitgVps.append(fitfcn.dwhisq_dm(x,p,b,'p','s',True))
            fitgAgV.append(p['gA00']/p['gV00'])
            fitgA.append(p['gA00'])
            fitgV.append(p['gV00'])
            fitE0.append(p['E0'])
        fit2ptss = np.array(fit2ptss)
        fit2ptps = np.array(fit2ptps)
        fit3ptss = np.array(fit3ptss)
        fit3ptps = np.array(fit3ptps)
        fitgVss = np.array(fitgVss)
        fitgVps = np.array(fitgVps)
        fitgAgV = np.array(fitgAgV)
        fitgA = np.array(fitgA)
        fitgV = np.array(fitgV)
        fitE0 = np.array(fitE0)
        std2ptss = np.std(fit2ptss,axis=0)
        std2ptps = np.std(fit2ptps,axis=0)
        std3ptss = np.std(fit3ptss,axis=0)
        std3ptps = np.std(fit3ptps,axis=0)
        stdgVss = np.std(fitgVss,axis=0)
        stdgVps = np.std(fitgVps,axis=0)
        string = 'x, 2ss, +-, 2ps, +-, 3ss, +-, 3ps, +-, gVss, +-, gVps, +-\n'
        for i in range(len(x)):
            string += '%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n' %(x[i],b0mss[i],std2ptss[i],b0mps[i],std2ptps[i],b03ptss[i],std3ptss[i],b03ptps[i],std3ptps[i],gV3ptss[i],stdgVss[i],gV3ptps[i],stdgVps[i])
        f_bs = open('./fh_bootstrap/%s_lsqfititer_%s.csv' %(params['gA_fit']['ens']['tag'],filetag), 'w+')
        f_bs.write(string)
        f_bs.flush()
        f_bs.close()
        sortfitgAgV = np.sort(fitgAgV)
        sortfitgA = np.sort(fitgA)
        sortfitgV = np.sort(fitgV)
        sortfitE0 = np.sort(fitE0)
        CI = [sortfitgAgV[int(len(fitgAgV)*0.158655254)], sortfitgAgV[int(len(fitgAgV)*0.841344746)], sortfitgAgV[int(len(fitgAgV)*0.5)]]
        CIgA = [sortfitgA[int(len(fitgA)*0.158655254)], sortfitgA[int(len(fitgA)*0.841344746)], sortfitgA[int(len(fitgA)*0.5)]]
        CIgV = [sortfitgV[int(len(fitgV)*0.158655254)], sortfitgV[int(len(fitgV)*0.841344746)], sortfitgV[int(len(fitgV)*0.5)]]
        CIE0 = [sortfitE0[int(len(fitE0)*0.158655254)], sortfitE0[int(len(fitE0)*0.841344746)], sortfitE0[int(len(fitE0)*0.5)]]
        print "gA/gV bootstrap uncertainty: %s +/- %s" %(np.mean(fitgAgV), np.std(fitgAgV))
        print "gA/gV median: %s" %CI[2]
        print "gA/gV CI: %s %s" %(CI[0], CI[1])
        print "dCI: %s" %((CI[1]-CI[0])*0.5)
        print "gA bootstrap uncertainty: %s +/- %s" %(np.mean(fitgA), np.std(fitgA))
        print "gA median: %s" %CIgA[2]
        print "gA CI: %s %s" %(CIgA[0], CIgA[1])
        print "dCI: %s" %((CIgA[1]-CIgA[0])*0.5)
        print "gV bootstrap uncertainty: %s +/- %s" %(np.mean(fitgV), np.std(fitgV))
        print "gV median: %s" %CIgV[2]
        print "gV CI: %s %s" %(CIgV[0], CIgV[1])
        print "dCI: %s" %((CIgV[1]-CIgV[0])*0.5)
        print "E0 bootstrap uncertainty: %s +/- %s" %(np.mean(fitE0), np.std(fitE0))
        print "E0 median: %s" %CIE0[2]
        print "E0 CI: %s %s" %(CIE0[0], CIE0[1])
        print "dCI: %s" %((CIE0[1]-CIE0[0])*0.5)
        f_bs = open('./fh_bootstrap/%s_gAgV_%s.pickle' %(params['gA_fit']['ens']['tag'],filetag), 'w+')
        pickle_result = {"gAgV": fitgAgV, "gA": fitgA, "gV": fitgV, "E0": fitE0}
        pickle.dump(pickle_result, f_bs)
        f_bs.flush()
        f_bs.close()
        # Find median and CI
        string = 'x, 2ss68plus, 2ss68minu, 2ps68plus, 2ps68minu, 3ss68plus, 3ss68minu, 3ps68plus, 3ps68minu, gVss68plus, gVss68minu, gVps68plus, gVps68minu\n'
        for i in range(len(x)):
            sort2ptss = np.sort(fit2ptss[:,i])
            sort2ptps = np.sort(fit2ptps[:,i])
            sort3ptss = np.sort(fit3ptss[:,i])
            sort3ptps = np.sort(fit3ptps[:,i])
            sortgVss = np.sort(fitgVss[:,i])
            sortgVps = np.sort(fitgVps[:,i])
            twoptssbias = b0mss[i] - sort2ptss[int(0.5*len(sort2ptss))]
            twoptpsbias = b0mps[i] - sort2ptps[int(0.5*len(sort2ptps))]
            thrptssbias = b03ptss[i] - sort3ptss[int(0.5*len(sort3ptss))]
            thrptpsbias = b03ptps[i] - sort3ptps[int(0.5*len(sort3ptps))]
            gVssbias = gV3ptss[i] - sortgVss[int(0.5*len(sortgVss))]
            gVpsbias = gV3ptps[i] - sortgVps[int(0.5*len(sortgVps))]
            twoptssplus = sort2ptss[int(0.841344746*len(sort2ptss))] + twoptssbias
            twoptssminu = sort2ptss[int(0.158655254*len(sort2ptss))] + twoptssbias
            twoptpsplus = sort2ptps[int(0.841344746*len(sort2ptps))] + twoptpsbias
            twoptpsminu = sort2ptps[int(0.158655254*len(sort2ptps))] + twoptpsbias
            thrptssplus = sort3ptss[int(0.841344746*len(sort3ptss))] + thrptssbias
            thrptssminu = sort3ptss[int(0.158655254*len(sort3ptss))] + thrptssbias
            thrptpsplus = sort3ptps[int(0.841344746*len(sort3ptps))] + thrptpsbias
            thrptpsminu = sort3ptps[int(0.158655254*len(sort3ptps))] + thrptpsbias
            gVssplus = sortgVss[int(0.841344746*len(sortgVss))] + gVssbias
            gVssminu = sortgVss[int(0.158655254*len(sortgVss))] + gVssbias
            gVpsplus = sortgVps[int(0.841344746*len(sortgVps))] + gVpsbias
            gVpsminu = sortgVps[int(0.158655254*len(sortgVps))] + gVpsbias
            string += '%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n' %(x[i],twoptssplus,twoptssminu,twoptpsplus,twoptpsminu,thrptssplus,thrptssminu,thrptpsplus,thrptpsminu,gVssplus,gVssminu,gVpsplus,gVpsminu)
        f_bs = open('./fh_bootstrap/%s_lsqfititer86_%s.csv' %(params['gA_fit']['ens']['tag'],filetag), 'w+')
        f_bs.write(string)
        f_bs.flush()
        f_bs.close()
        plt.figure()
        n, bins, patches = plt.hist(fitgAgV, 50, facecolor='green')
        x = np.delete(bins, -1)
        plt.plot(x, n)
        plt.xlabel('gA/gV')
        plt.ylabel('counts')
        plt.draw()
        plt.show()
