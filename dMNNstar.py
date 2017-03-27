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
    fhbp = params[tag]['gA'][mq]
    # make gvars
    spec = gvboot0[:len(gvboot0)/2]
    fh = gvboot0[len(gvboot0)/2:]
    T = len(fh)/(2*len(basak))
    # R(t)
    Rl = fh/spec
    # dmeff [R(t+tau) - R(t)] / tau
    #dM = Rl
    dM = (np.roll(Rl,-tau)-Rl)/float(tau) #This is needed to plot gA correctly in the effective plots, but will give wrong data to fit with.
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
            #print stuff
            #print 'meff_ss'
            #for t in range(len(meff_ss)):
            #    print t, meff_ss[t].mean, meff_ss[t].sdev
            #print 'meff_ps'
            #for t in range(len(meff_ps)):
            #    print t, meff_ps[t].mean, meff_ps[t].sdev
            plt.show()
            # scaled correlator
            E0 = barp['priors'][1]['E0'][0]
            scaled_ss = eff.scaled_correlator(SS, E0, phase=1.0)
            scaled_ps = eff.scaled_correlator(PS, E0, phase=1.0)
            ylim = c51.find_yrange(scaled_ss, xlim[0], xlim[1])
            c51.scatter_plot(np.arange(len(scaled_ss)), scaled_ss, '%s %s ss scaled correlator (take sqrt to get Z0_s)' %(basak[b],str(mq)), xlim = xlim, ylim = ylim)
            ylim = c51.find_yrange(scaled_ps, xlim[0], xlim[1])
            c51.scatter_plot(np.arange(len(scaled_ps)), scaled_ps, '%s %s ps scaled correlator (divide by Z0_s to get Z0_p)' %(basak[b],str(mq)), xlim = xlim, ylim = ylim)
            #print stuff
            #print 'scaled_ss'
            #for t in range(len(scaled_ss)):
            #    print t, scaled_ss[t].mean, scaled_ss[t].sdev
            #print 'scaled_ps'
            #for t in range(len(meff_ps)):
            #    print t, scaled_ps[t].mean, scaled_ps[t].sdev
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
        boot0fit = c51.fitscript_v2(trange,T,boot0gv,boot0p,fitfcn.twopt_baryon_ss_ps,basak=params['gA_fit']['basak'])
        print boot0fit['rawoutput'][0]
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
            print tabulate(tbl_print, headers='keys')
        # submit boot0 to db
        if params['flags']['write']:
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
    tau = params['gA_fit']['tau']
    barp = params[tag]['proton'][mq]
    fhbp = params[tag]['gA'][mq]
    # make gvars
    spec = gvboot0[:len(gvboot0)/2]
    fh = gvboot0[len(gvboot0)/2:]
    T = len(fh)/(2*len(basak))
    # R(t)
    Rl = fh/spec
    # dmeff [R(t+tau) - R(t)] / tau
    #dM = Rl
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
        boot0fit = c51.fitscript_v3(trange,fhtrange,T,boot0gv,boot0p,fitfcn.baryon_dm_ss_ps,basak=params['gA_fit']['basak'])
        print boot0fit['rawoutput'][0]
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
        # submit boot0 to db
        if params['flags']['write']:
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
    user_flag = c51.user_list()
    f = open('./fhprotonmaster.yml.%s' %(user_flag),'r')
    params = yaml.load(f)
    f.close()
    # log in sql
    psqlpwd = pwd.passwd()
    psql = sql.pysql('cchang5','cchang5',psqlpwd)
    # fit gA
    boot0 = read_gA_bs(psql,params)
    gvboot0 = c51.make_gvars(boot0)
    fit_proton(psql,params,gvboot0)
    res = fit_gA(psql,params,gvboot0)
    #print res['gA']
