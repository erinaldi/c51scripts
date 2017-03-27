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

def gA_bs(psql,params):
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
    print "fitting for gA mq %s, basak %s, ens %s%s, Nbs %s, Mbs %s" %(str(mq),str(basak),str(tag),str(stream),str(Nbs),str(Mbs))
    # read two point
    SSl = np.array([psql.data('dwhisq_corr_baryon',idx) for idx in [barp['meta_id']['SS'][i] for i in basak]])
    PSl = np.array([psql.data('dwhisq_corr_baryon',idx) for idx in [barp['meta_id']['PS'][i] for i in basak]])
    T = len(SSl[0,0])
    # read fh correlator
    fhSSl = np.array([psql.data('dwhisq_fhcorr_baryon',idx) for idx in [fhbp['meta_id']['SS'][i] for i in basak]])
    fhPSl = np.array([psql.data('dwhisq_fhcorr_baryon',idx) for idx in [fhbp['meta_id']['PS'][i] for i in basak]])
    # check outliers
    if False:
        # corr [basak, config, time]
        temp = [SSl, PSl, fhSSl, fhPSl]
        ntemp = ['SSl', 'PSl', 'fhSSl', 'fhPSl']
        for j in range(len(temp)):
            print ntemp[j]
            corr = temp[j]
            for i in range(len(corr[0,0])):
                corrt = corr[0,:,i] #00 ss 30 ps
                minimum = np.amin(corrt)
                argmin = np.argmin(corrt)*5+300
                median = np.median(corrt)
                maximum = np.amax(corrt)
                argmax = np.argmax(corrt)*5+300
                print i, median - minimum, maximum-median, argmin, argmax
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
    # make gvars
    gvboot0 = c51.make_gvars(boot0)
    spec = gvboot0[:len(gvboot0)/2]
    fh = gvboot0[len(gvboot0)/2:]
    # R(t)
    Rl = fh/spec
    # dmeff [R(t+tau) - R(t)] / tau
    dM = (np.roll(Rl,-tau)-Rl)/float(tau)
    # plot data
    if params['flags']['plot_data']:
        for b in range(len(basak)):
            SS = SSl[b]
            PS = PSl[b]
            # raw correlator
            c51.scatter_plot(np.arange(len(SS[0])), c51.make_gvars(SS), '%s %s ss' %(basak[b],str(mq)))
            c51.scatter_plot(np.arange(len(PS[0])), c51.make_gvars(PS), '%s %s ps' %(basak[b],str(mq)))
            plt.show()
            # effective mass
            eff = c51.effective_plots(T)
            meff_ss = eff.effective_mass(c51.make_gvars(SS), 1, 'log')
            meff_ps = eff.effective_mass(c51.make_gvars(PS), 1, 'log')
            xlim = [2, len(meff_ss)/3-2]
            ylim = c51.find_yrange(meff_ss, xlim[0], xlim[1])
            c51.scatter_plot(np.arange(len(meff_ss)), meff_ss, '%s %s ss effective mass' %(basak[b],str(mq)), xlim = xlim, ylim = ylim)
            ylim = c51.find_yrange(meff_ps, xlim[0], xlim[1])
            c51.scatter_plot(np.arange(len(meff_ps)), meff_ps, '%s %s ps effective mass' %(basak[b],str(mq)), xlim = xlim, ylim = ylim)
            plt.show()
            # scaled correlator
            E0 = barp['priors'][1]['E0'][0]
            scaled_ss = eff.scaled_correlator(c51.make_gvars(SS), E0, phase=1.0)
            scaled_ps = eff.scaled_correlator(c51.make_gvars(PS), E0, phase=1.0)
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
        #boot0gv = spec
        boot0p = c51.dict_of_tuple_to_gvar(prior)
        fitfcn = c51.fit_function(T,nstates)
        #boot0fit = c51.fitscript_v2(trange,T,boot0gv,boot0p,fitfcn.twopt_baryon_ss_ps,basak=params['gA_fit']['basak'])
        # fit two point
        for b in range(len(basak)):
            SSspec = (spec[:len(spec)/2])[b*T:(b+1)*T]
            PSspec = (spec[len(spec)/2:])[b*T:(b+1)*T]
            boot0gv = np.concatenate((SSspec, PSspec))
            boot0fit = c51.fitscript_v2(trange,T,boot0gv,boot0p,fitfcn.twopt_baryon_ss_ps,basak=[basak[b]])
            boot0p = boot0fit['rawoutput'][0].p
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
                init_id = psql.initid(c51.baryon_initpriors(barp['priors'],basak,nstates))
                prior_id = psql.priorid(c51.dict_of_tuple_to_gvar(c51.baryon_priors(barp['priors'],basak,nstates)))
                tmin = boot0fit['tmin'][t]
                tmax = boot0fit['tmax'][t]
                result = c51.make_result(boot0fit,t)
                psql.submit_boot0('proton',corr_lst,fit_id,tmin,tmax,init_id,prior_id,result,params['flags']['update'])
    # plot fh correlator
    if params['flags']['plot_fhdata']:
        for b in range(len(basak)):
            # raw correlator dC_lambda/dlambda
            fhSS = fhSSl[b]
            fhPS = fhPSl[b]
            c51.scatter_plot(np.arange(len(fhSS[0])), c51.make_gvars(fhSS), '%s %s fh ss' %(basak[b],str(mq)))
            c51.scatter_plot(np.arange(len(fhPS[0])), c51.make_gvars(fhPS), '%s %s fh ps' %(basak[b],str(mq)))
            plt.show()
            # R(t)
            RSS = (Rl[:len(Rl)/2])[b*len(Rl)/(2*len(basak)):(b+1)*len(Rl)/(2*len(basak))]
            RPS = (Rl[len(Rl)/2:])[b*len(Rl)/(2*len(basak)):(b+1)*len(Rl)/(2*len(basak))]
            c51.scatter_plot(np.arange(len(RSS)), RSS, '%s %s R(t) ss' %(basak[b],str(mq)))
            c51.scatter_plot(np.arange(len(RPS)), RPS, '%s %s R(t) ps' %(basak[b],str(mq)))
            plt.show()
            # dmeff R(t+tau) - R(t)
            dMSS = (dM[:len(dM)/2])[b*len(dM)/(2*len(basak)):(b+1)*len(dM)/(2*len(basak))]
            dMPS = (dM[len(dM)/2:])[b*len(dM)/(2*len(basak)):(b+1)*len(dM)/(2*len(basak))]
            c51.scatter_plot(np.arange(len(dMSS)), dMSS, '%s %s [R(t+%s)-R(t)]/%s ss' %(basak[b],str(mq),str(tau),str(tau)))
            c51.scatter_plot(np.arange(len(dMPS)), dMPS, '%s %s [R(t+%s)-R(t)]/%s ps' %(basak[b],str(mq),str(tau),str(tau)))
            plt.show()
    # fit fh correlator
    if params['flags']['fit_gA']:
        # data concatenated previously
        # read priors
        prior = c51.fhbaryon_priors(barp['priors'],fhbp['priors'],basak,nstates)
        boot0p = c51.dict_of_tuple_to_gvar(prior)
        fitfcn = c51.fit_function(T,nstates,tau)
        # read trange
        trange = barp['trange']
        fhtrange = fhbp['trange']
        # fit two point
        for b in range(len(basak)):
            SSspec = (spec[:len(spec)/2])[b*T:(b+1)*T]
            PSspec = (spec[len(spec)/2:])[b*T:(b+1)*T]
            boot0gv = np.concatenate((SSspec, PSspec))
            boot0fit = c51.fitscript_v2(trange,T,boot0gv,boot0p,fitfcn.twopt_baryon_ss_ps,basak=[basak[b]])
            boot0p = boot0fit['rawoutput'][0].p
        # fit fhcorr
        for b in range(len(basak)):
            SSdM = (dM[:len(dM)/2])[b*T:(b+1)*T]
            PSdM = (dM[len(dM)/2:])[b*T:(b+1)*T]
            boot0gv = np.concatenate((SSdM, PSdM))
            boot0fit = c51.fitscript_v2(fhtrange,T,boot0gv,boot0p,fitfcn.fhbaryon_ss_ps,basak=[basak[b]])
            boot0p = boot0fit['rawoutput'][0].p
        if params['flags']['stability_plot']:
            c51.stability_plot(boot0fit,'E0','%s' %str(mq))
            c51.stability_plot(boot0fit,'gA00','%s' %str(mq))
            plt.show()
        if params['flags']['tabulate']:
            tbl_print = collections.OrderedDict()
            tbl_print['tmin'] = boot0fit['tmin']
            tbl_print['tmax'] = boot0fit['tmax']
            #tbl_print['fhtmin'] = boot0fit['fhtmin']
            #tbl_print['fhtmax'] = boot0fit['fhtmax']
            tbl_print['E0'] = [boot0fit['pmean'][t]['E0'] for t in range(len(boot0fit['pmean']))]
            tbl_print['dE0'] = [boot0fit['psdev'][t]['E0'] for t in range(len(boot0fit['pmean']))]
            tbl_print['gA00'] = [boot0fit['pmean'][t]['gA00'] for t in range(len(boot0fit['pmean']))]
            tbl_print['dgA00'] = [boot0fit['psdev'][t]['gA00'] for t in range(len(boot0fit['pmean']))]
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
            corr_lst = np.array([[fhbp['meta_id']['SS'][i] for i in params['gA_fit']['basak']],[fhbp['meta_id']['PS'][i] for i in params['gA_fit']['basak']]]).flatten()
            fit_id = c51.select_fitid('fhbaryon',nstates=nstates,tau=params['gA_fit']['tau'])
            for t in range(len(boot0fit['tmin'])):
                baryon_corr_lst = np.array([[barp['meta_id']['SS'][i] for i in params['gA_fit']['basak']],[barp['meta_id']['PS'][i] for i in params['gA_fit']['basak']]]).flatten()
                baryon_fit_id = c51.select_fitid('baryon',nstates=nstates,basak=params['gA_fit']['basak'])
                baryon_tmin = trange['tmin'][0]
                baryon_tmax = trange['tmax'][0]
                baryon_init_id = psql.initid(c51.baryon_initpriors(barp['priors'],basak,nstates))
                baryon_prior_id = psql.priorid(c51.dict_of_tuple_to_gvar(c51.baryon_priors(barp['priors'],basak,nstates)))
                baryon_id = psql.select_boot0("proton", baryon_corr_lst, baryon_fit_id, baryon_tmin, baryon_tmax, baryon_init_id, baryon_prior_id) 
                init_id = psql.initid(boot0fit['p0'][t])
                prior_id = psql.priorid(boot0fit['prior'][t])
                tmin = boot0fit['tmin'][t]
                tmax = boot0fit['tmax'][t]
                result = c51.make_result(boot0fit,t)
                psql.submit_fhboot0('fhproton',corr_lst,baryon_id,fit_id,tmin,tmax,init_id,prior_id,result,params['flags']['update'])
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
    gA_bs(psql,params)
