# fit for gA of the proton
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

def read_gA_bs(psql,params):
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
    fhbp = params[tag]['gA'][mq]
    print "reading for gA mq %s, basak %s, ens %s%s, Nbs %s, Mbs %s" %(str(mq),str(basak),str(tag),str(stream),str(Nbs),str(Mbs))
    print barp['meta_id']['SS'], barp['meta_id']['PS']
    print fhbp['meta_id']['SS'], fhbp['meta_id']['PS']
    # read two point
    SSl = np.array([psql.data('dwhisq_corr_baryon',idx) for idx in [barp['meta_id']['SS'][i] for i in basak]])
    PSl = np.array([psql.data('dwhisq_corr_baryon',idx) for idx in [barp['meta_id']['PS'][i] for i in basak]])
    T = len(SSl[0,0])
    # read fh correlator
    fhSSl = np.array([psql.data('dwhisq_fhcorr_baryon',idx) for idx in [fhbp['meta_id']['SS'][i] for i in basak]])
    fhPSl = np.array([psql.data('dwhisq_fhcorr_baryon',idx) for idx in [fhbp['meta_id']['PS'][i] for i in basak]])
    # concatenate and make gvars to preserve correlations
    SS = SSl[0]
    PS = PSl[0]
    for i in range(len(SSl)-1): # loop over basak operators
        SS = np.concatenate((SS,SSl[i+1]),axis=1)
        PS = np.concatenate((PS,PSl[i+1]),axis=1)
    fhSS = fhSSl[0]
    fhPS = fhPSl[0]
    for i in range(len(fhSSl)-1):
        fhSS = np.concatenate((fhSS,fhSSl[i+1]),axis=1)
        fhPS = np.concatenate((fhPS,fhPSl[i+1]),axis=1)
    boot0 = np.concatenate((SS, PS, fhSS, fhPS), axis=1)
    return boot0

def fit_proton(psql,params,gvboot0):
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
    spec = gvboot0[:len(gvboot0)/2]
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
            t = np.linspace(0, T/2, 100*T/2)
            b = params['gA_fit']['basak'][0]
            # output fit curve
            ss = fitfcn.dwhisq_twopt(t,p,b,'s','s')
            ps = fitfcn.dwhisq_twopt(t,p,b,'p','s')
            scaled_ss = np.sqrt(ss*np.exp(p['E0']*t))
            scaled_ps = ps*np.exp(p['E0']*t)/scaled_ss
            meffss = np.log(ss/np.roll(ss,-100))/t[100]
            meffps = np.log(ps/np.roll(ps,-100))/t[100]
            f_plot = open('./fh_fitline/%s_twopt.csv' %(tag), 'w+')
            string = 't, css, +-, cps, +-\n'
            for i in range(len(t)):
                string += '%s, %s, %s, %s, %s\n' %(t[i], meffss[i].mean, meffss[i].sdev, meffps[i].mean, meffps[i].sdev)
            f_plot.write(string)
            f_plot.flush()
            f_plot.close()
            f_plot = open('./fh_fitline/%s_scaled2pt.csv' %(tag), 'w+')
            # data scaled two point: propagate E0 error
            SSl = spec[:len(spec)/2].reshape((len(basak),T))[0]
            PSl = spec[len(spec)/2:].reshape((len(basak),T))[0]
            t = np.linspace(0,T-1,T)
            scaled_ss = np.sqrt(SSl*np.exp(p['E0']*t))
            scaled_ps = PSl*np.exp(p['E0']*t)/scaled_ss
            string = 't, zss, +-, zps, +-\n'
            for i in range(len(t)):
                string += '%s, %s, %s, %s, %s\n' %(t[i], scaled_ss[i].mean, scaled_ss[i].sdev, scaled_ps[i].mean, scaled_ps[i].sdev)
            f_plot = open('/Users/cchang5/Documents/Papers/FH/c51_p1/paper/figures/scaled_corr.csv', 'w+')
            f_plot.write(string)
            f_plot.flush()
            f_plot.close()
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
        # submit boot0 to db
        if False: #params['flags']['write']:
            corr_lst = np.array([[barp['meta_id']['SS'][i] for i in params['gA_fit']['basak']],[barp['meta_id']['PS'][i] for i in params['gA_fit']['basak']]]).flatten()
            fit_id = c51.select_fitid('baryon',nstates=nstates,basak=params['gA_fit']['basak'])
            for t in range(len(boot0fit['tmin'])):
                init_id = psql.initid(boot0fit['p0'][t])
                prior_id = psql.priorid(boot0fit['prior'][t])
                tmin = boot0fit['tmin'][t]
                tmax = boot0fit['tmax'][t]
                result = c51.make_result(boot0fit,t)
                psql.submit_boot0('proton',corr_lst,fit_id,tmin,tmax,init_id,prior_id,result,params['flags']['update'])
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
    tau = params['gA_fit']['tau']
    barp = params[tag]['proton'][mq]
    fhbp = params[tag]['gA'][mq]
    #print "nstates: %s" %nstates
    #print "fhstates: %s" %fhstates
    # make gvars
    spec = gvboot0[:len(gvboot0)/2]
    fh = gvboot0[len(gvboot0)/2:]
    T = len(fh)/(2*len(basak))
    # R(t)
    Rl = fh/spec
    dM = (np.roll(Rl,-tau)-Rl)/float(tau) #This is needed to plot gA correctly in the effective plots, but will give wrong data to fit with.
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
            # ground state nucleon mass prior central value
            E0 = barp['priors'][1]['E0'][0]
            # raw correlator dC_lambda/dlambda
            fhSS = fhSSl[b]
            fhPS = fhPSl[b]
            #c51.scatter_plot(np.arange(len(fhSS)), fhSS, '%s %s fh ss' %(basak[b],str(mq)))
            #c51.scatter_plot(np.arange(len(fhPS)), fhPS, '%s %s fh ps' %(basak[b],str(mq)))
            print "%s fhSS[1]*exp(E0):" %basak[b], fhSS[1]*np.exp(E0) #, "fhSS[1]:", fhSS[1]
            print "%s fhPS[1]*exp(E0):" %basak[b], fhPS[1]*np.exp(E0) #, "fhPS[1]:", fhPS[1]
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
        prior = c51.fhbaryon_priors(barp['priors'],fhbp['priors'],basak,nstates,fhstates)
        # read init
        try:
            #print "found init file"
            f_init = open('./fh_posterior/%s.yml' %(tag), 'r')
            init = yaml.load(f_init)
            f_init.close()
        except:
            init = None
        #print prior
        # read trange
        trange = barp['trange']
        fhtrange = fhbp['trange']
        # fit boot0
        boot0gv = gvboot0 #np.concatenate((spec, fh))
        #boot0gv = np.concatenate((spec,dM))
        boot0p = c51.dict_of_tuple_to_gvar(prior)
        fitfcn = c51.fit_function(T,nstates,fhstates,tau)
        boot0fit = c51.fitscript_v3(trange,fhtrange,T,boot0gv,boot0p,fitfcn.dwhisq_fh_ss_ps,basak=params['gA_fit']['basak'],init=None,bayes=params['flags']['bayes'])
        #boot0fit = c51.fitscript_v3(trange,fhtrange,T,boot0gv,boot0p,fitfcn.dwhisq_dm_ss_ps,basak=params['gA_fit']['basak'],init=None,bayes=params['flags']['bayes'])
        print boot0fit['rawoutput'][0]
        if params['flags']['fitline_plot']:
            p = boot0fit['rawoutput'][0].p
            t = np.linspace(0, T/2, 100*T/2)
            b = params['gA_fit']['basak'][0]
            # output fit curve
            ss = fitfcn.dwhisq_twopt(t,p,b,'s','s')
            ps = fitfcn.dwhisq_twopt(t,p,b,'p','s')
            fhss = fitfcn.dwhisq_fh(t,p,b,'s','s',False)
            fhps = fitfcn.dwhisq_fh(t,p,b,'p','s',False)
            rss = fhss/ss
            rps = fhps/ps
            yss = (np.roll(rss,-100)-rss)/t[100]
            yps = (np.roll(rps,-100)-rps)/t[100]
            #f_plot = open('./fh_fitline/%s.csv' %(tag), 'w+')
            f_plot = open('./fh_fitline/%s_fh.csv' %(tag), 'w+')
            string = 't, yss, +-, yps, +-\n'
            for i in range(len(t)):
                #string += '%s, %s, %s, %s, %s\n' %(t[i], yss[i].mean, yss[i].sdev, yps[i].mean, yps[i].sdev)
                string += '%s, %s, %s, %s, %s\n' %(t[i], fhss[i].mean, fhss[i].sdev, fhps[i].mean, fhps[i].sdev)
            f_plot.write(string)
            f_plot.flush()
            f_plot.close()
            meffss = np.log(ss/np.roll(ss,-100))/t[100]
            meffps = np.log(ps/np.roll(ps,-100))/t[100]
            f_plot = open('./fh_fitline/%s_twopt.csv' %(tag), 'w+')
            string = 't, css, +-, cps, +-\n'
            for i in range(len(t)):
                string += '%s, %s, %s, %s, %s\n' %(t[i], meffss[i].mean, meffss[i].sdev, meffps[i].mean, meffps[i].sdev)
            f_plot.write(string)
            f_plot.flush()
            f_plot.close()
            # output data
            dMSSl = fh[:len(fh)/2].reshape((len(basak),T))[0]
            dMPSl = fh[len(fh)/2:].reshape((len(basak),T))[0]
            #dMSSl = dM[:len(dM)/2].reshape((len(basak),T))[0]
            #dMPSl = dM[len(dM)/2:].reshape((len(basak),T))[0]
            t = np.linspace(0,30,31)
            f_data = open('./fh_fitline/%s_fh_dat.csv' %(tag), 'w+')
            #f_data = open('./fh_fitline/%s_dat.csv' %(tag), 'w+')
            string = 't_dat, yss_dat, +-, yps_dat, +-\n'
            for i in range(len(t)):
                string += '%s, %s, %s, %s, %s\n' %(t[i], dMSSl[i].mean, dMSSl[i].sdev, dMPSl[i].mean, dMPSl[i].sdev)
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
            f_dump = open('./fh_posterior/%s.yml' %(tag), 'w+')
            yaml.dump(fh_dump, f_dump)
            f_dump.flush()
            f_dump.close()
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
            tbl_print['E0'] = [boot0fit['pmean'][t]['E0'] for t in range(len(boot0fit['pmean']))]
            tbl_print['dE0'] = [boot0fit['psdev'][t]['E0'] for t in range(len(boot0fit['pmean']))]
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
            tbl_print['Q'] = boot0fit['Q']
            print tabulate(tbl_print, headers='keys')
            print "tmin, %sE0%s, +-, %sgA%s, +-, %schi2dof%s, %sQ%s, %slogGBF%s" %(fhstates, nstates, fhstates, nstates, fhstates, nstates, fhstates, nstates, fhstates, nstates)
            for i in range(len(boot0fit['fhtmin'])):
                print '%s, %s, %s, %s, %s, %s, %s, %s' %(str(boot0fit['fhtmin'][i]), str(boot0fit['pmean'][i]['E0']), str(boot0fit['psdev'][i]['E0']), str(boot0fit['pmean'][i]['gA00']), str(boot0fit['psdev'][i]['gA00']), boot0fit['chi2'][i]/boot0fit['dof'][i], boot0fit['Q'][i], boot0fit['logGBF'][i])
        # submit boot0 to db
        if False: #params['flags']['write']:
            corr_lst = np.array([[fhbp['meta_id']['SS'][i] for i in params['gA_fit']['basak']],[fhbp['meta_id']['PS'][i] for i in params['gA_fit']['basak']]]).flatten()
            fit_id = c51.select_fitid('fhbaryon',nstates=nstates,tau=params['gA_fit']['tau'])
            for t in range(len(boot0fit['tmin'])):
                baryon_corr_lst = np.array([[barp['meta_id']['SS'][i] for i in params['gA_fit']['basak']],[barp['meta_id']['PS'][i] for i in params['gA_fit']['basak']]]).flatten()
                baryon_fit_id = c51.select_fitid('baryon',nstates=nstates,basak=params['gA_fit']['basak'])
                baryon_tmin = trange['tmin'][0]
                baryon_tmax = trange['tmax'][0]
                baryon_p0 = {k: boot0fit['p0'][0][k] for k in [bk for n in range(nstates) for bk in barp['priors'][n+1].keys()]}
                baryon_init_id = psql.initid(baryon_p0)
                baryon_prior_id = psql.priorid(c51.dict_of_tuple_to_gvar(c51.baryon_priors(barp['priors'],basak,nstates)))
                baryon_id = psql.select_boot0("proton", baryon_corr_lst, baryon_fit_id, baryon_tmin, baryon_tmax, baryon_init_id, baryon_prior_id)
                init_id = psql.initid(boot0fit['p0'][t])
                prior_id = psql.priorid(boot0fit['prior'][t])
                tmin = boot0fit['fhtmin'][t]
                tmax = boot0fit['fhtmax'][t]
                result = c51.make_result(boot0fit,t)
                print tmin, tmax
                psql.submit_fhboot0('fhproton',corr_lst,baryon_id,fit_id,tmin,tmax,init_id,prior_id,result,params['flags']['update'])
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
    boot0 = read_gA_bs(psql,params)
    # bin
    boot0 = gv.dataset.bin_data(boot0,binsize=1)
    print np.shape(boot0)
    gvboot0 = c51.make_gvars(boot0)
    fit_proton(psql,params,gvboot0)
    res = fit_gA(psql,params,gvboot0)
    # bootstrap gA
    if params['flags']['bootstrap']:
    # read data
        b = params['gA_fit']['basak'][0]
        nstates = params['gA_fit']['nstates']
        fhstates = params['gA_fit']['fhstates']
        tau = params['gA_fit']['tau']
        fh = gvboot0[len(gvboot0)/2:]
        T = len(fh)/2
        fitfcn = c51.fit_function(T,nstates,fhstates,tau)
        x = np.linspace(0, T/2, 1000)
        fit2ptss = []
        fit2ptps = []
        fit3ptss = []
        fit3ptps = []
        p = res['gA_fit'].pmean
        b02ptss = fitfcn.dwhisq_twopt(x,p,b,'s','s')
        b02ptps = fitfcn.dwhisq_twopt(x,p,b,'p','s')
        b0mss = np.log(b02ptss/np.roll(b02ptss,-1))/x[1]
        b0mps = np.log(b02ptps/np.roll(b02ptps,-1))/x[1]
        b03ptss = fitfcn.dwhisq_dm(x,p,b,'s','s')
        b03ptps = fitfcn.dwhisq_dm(x,p,b,'p','s')
        for sfit in tqdm.tqdm(res['gA_fit'].bootstrapped_fit_iter(n=1000)):
            p = sfit.pmean
            ss2pt = fitfcn.dwhisq_twopt(x,p,b,'s','s')
            ps2pt = fitfcn.dwhisq_twopt(x,p,b,'p','s')
            mss = np.log(ss2pt/np.roll(ss2pt,-1))/x[1]
            mps = np.log(ps2pt/np.roll(ps2pt,-1))/x[1]
            fit2ptss.append(mss)
            fit2ptps.append(mps)
            fit3ptss.append(fitfcn.dwhisq_dm(x,p,b,'s','s'))
            fit3ptps.append(fitfcn.dwhisq_dm(x,p,b,'p','s'))
        std2ptss = np.std(fit2ptss,axis=0)
        std2ptps = np.std(fit2ptps,axis=0)
        std3ptss = np.std(fit3ptss,axis=0)
        std3ptps = np.std(fit3ptps,axis=0)
        string = 'x, 2ss, +-, 2ps, +-, 3ss, +-, 3ps, +-\n'
        for i in range(len(x)):
            string += '%s, %s, %s, %s, %s, %s, %s, %s, %s\n' %(x[i],b0mss[i],std2ptss[i],b0mps[i],std2ptps[i],b03ptss[i],std3ptss[i],b03ptps[i],std3ptps[i])
        f_bs = open('./fh_bootstrap/%s_lsqfititer.csv' %(params['gA_fit']['ens']['tag']), 'w+')
        f_bs.write(string)
        f_bs.flush()
        f_bs.close()
