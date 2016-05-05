import sys
sys.path.append('$HOME/Physics/c51/scripts/')
import sqlc51lib as c51
import calsql as sql
import password_file as pwd
import numpy as np
import matplotlib.pyplot as plt
import gvar as gv
import multiprocessing as multi
from tabulate import tabulate
import yaml
import collections
import tqdm

def mres_bs(psql,params,mq):
    print 'fitting mres', mq
    tag = params['decay_ward_fit']['ens']['tag']
    stream = params['decay_ward_fit']['ens']['stream']
    Nbs = params['decay_ward_fit']['nbs']
    Mbs = params['decay_ward_fit']['mbs'] 
    mresp = params[tag]['mres'][mq]
    # read data
    mp = psql.data('dwhisq_corr_jmu',mresp['meta_id']['mp'])
    pp = psql.data('dwhisq_corr_jmu',mresp['meta_id']['pp'])
    T = len(pp[0])
    boot0 = mp/pp
    # plot mres
    if params['flags']['plot_data']:
        c51.scatter_plot(np.arange(len(boot0[0])), c51.make_gvars(boot0), '%s mres' %str(mq))
        plt.show()
    # read priors
    prior = mresp['priors']
    # read trange
    trange = mresp['trange']
    # fit boot0
    boot0gv = c51.make_gvars(boot0)
    boot0p = c51.dict_of_tuple_to_gvar(prior)
    fitfcn = c51.fit_function(T)
    boot0fit = c51.fitscript_v2(trange,T,boot0gv,boot0p,fitfcn.mres_fitfcn)
    if params['flags']['stability_plot']:
        c51.stability_plot(boot0fit,'mres',str(mq))
        plt.show()
    if params['flags']['tabulate']:
        tbl_print = collections.OrderedDict()
        tbl_print['tmin'] = boot0fit['tmin']
        tbl_print['tmax'] = boot0fit['tmax']
        tbl_print['mres'] = [boot0fit['pmean'][t]['mres'] for t in range(len(boot0fit['pmean']))]
        tbl_print['sdev'] = [boot0fit['psdev'][t]['mres'] for t in range(len(boot0fit['pmean']))]
        tbl_print['chi2/dof'] = np.array(boot0fit['chi2'])/np.array(boot0fit['dof'])
        tbl_print['logGBF'] = boot0fit['logGBF']
        print tabulate(tbl_print, headers='keys')
    # submit boot0 to db
    if params['flags']['write']:
        corr_lst = [mresp['meta_id']['mp'],mresp['meta_id']['pp']]
        fit_id = c51.select_fitid('mres',params)
        for t in range(len(boot0fit['tmin'])):
            init_id = psql.initid(boot0fit['p0'][t])
            prior_id = psql.priorid(boot0fit['prior'][t])
            tmin = boot0fit['tmin'][t]
            tmax = boot0fit['tmax'][t]
            result = c51.make_result(boot0fit,t)
            psql.submit_boot0('jmu',corr_lst,fit_id,tmin,tmax,init_id,prior_id,result,params['flags']['update'])
    if len(boot0fit['tmin'])!=1 or len(boot0fit['tmax'])!=1:
        print "sweeping over time range"
        print "skipping bootstrap"
        return 0
    else: pass
    if params['flags']['write']:
        bsresult = []
        psql.chkbsprior(tag,stream,Nbs,boot0fit['prior'][0])
        for g in tqdm.tqdm(range(Nbs)):
            # make bs gvar dataset
            n = g+1
            bs_id,draw = psql.fetchonebs(nbs=n,Mbs=Mbs,ens=tag,stream=stream)
            bootn = boot0[draw]
            bootngv = c51.make_gvars(bootn)
            # read randomized priors
            bsprior_id,bsp = psql.bspriorid(tag,stream,g,boot0fit['prior'][0])
            bsp = c51.dict_of_tuple_to_gvar(bsp)
            # read initial guess
            init = {"mres":boot0fit['pmean'][0]['mres']}
            #Fit
            bsfit = c51.fitscript_v2(trange,T,bootngv,bsp,fitfcn.mres_fitfcn,init)
            tmin = bsfit['tmin'][0]
            tmax = bsfit['tmax'][0]
            boot0_id = psql.select_boot0('jmu',corr_lst,fit_id,tmin,tmax,init_id,prior_id)
            bsinit_id = psql.initid(bsfit['p0'][0])
            result = c51.make_result(bsfit,0) #"""{"mres":%s, "chi2":%s, "dof":%s}""" %(bsfit['pmean'][t]['mres'],bsfit['chi2'][t],bsfit['dof'][t])
            psql.submit_bs('jmu_bs',boot0_id,bs_id,Mbs,bsinit_id,bsprior_id,result,params['flags']['update'])
    return 0

def decay_bs(psql,params,meson):
    if meson=='pion':
        mq1 = params['decay_ward_fit']['ml']
        mq2 = mq1
    elif meson == 'kaon':
        mq1 = params['decay_ward_fit']['ml']
        mq2 = params['decay_ward_fit']['ms']
    elif meson == 'etas':
        mq1 = params['decay_ward_fit']['ms']
        mq2 = mq1
    print 'fitting twopoint', mq1, mq2
    tag = params['decay_ward_fit']['ens']['tag']
    stream = params['decay_ward_fit']['ens']['stream']
    Nbs = params['decay_ward_fit']['nbs']
    Mbs = params['decay_ward_fit']['mbs']
    mesp = params[tag][meson]['%s_%s' %(str(mq1),str(mq2))]
    # read data
    SS = c51.fold(psql.data('dwhisq_corr_meson',mesp['meta_id']['SS']))
    PS = c51.fold(psql.data('dwhisq_corr_meson',mesp['meta_id']['PS']))
    T = len(SS[0])
    # plot effective mass / scaled correlator
    if params['flags']['plot_data']:
        # unfolded correlator data
        c51.scatter_plot(np.arange(len(SS[0])), c51.make_gvars(SS), '%s_%s ss folded' %(str(mq1),str(mq2)))
        c51.scatter_plot(np.arange(len(PS[0])), c51.make_gvars(PS), '%s_%s ps folded' %(str(mq1),str(mq2)))
        plt.show()
        # effective mass
        eff = c51.effective_plots(T)
        meff_ss = eff.effective_mass(c51.make_gvars(SS), 1, 'cosh')
        meff_ps = eff.effective_mass(c51.make_gvars(PS), 1, 'cosh')
        xlim = [3, len(meff_ss)/2-2]
        ylim = c51.find_yrange(meff_ss, xlim[0], xlim[1])
        c51.scatter_plot(np.arange(len(meff_ss)), meff_ss, '%s_%s ss effective mass' %(str(mq1),str(mq2)), xlim = xlim, ylim = ylim)
        ylim = c51.find_yrange(meff_ps, xlim[0], xlim[1])
        c51.scatter_plot(np.arange(len(meff_ps)), meff_ps, '%s_%s ps effective mass' %(str(mq1),str(mq2)), xlim = xlim, ylim = ylim)
        plt.show()
        # scaled correlator
        E0 = mesp['priors']['E0'][0]
        scaled_ss = eff.scaled_correlator(c51.make_gvars(SS), E0, phase=1.0)
        scaled_ps = eff.scaled_correlator(c51.make_gvars(PS), E0, phase=1.0)
        ylim = c51.find_yrange(scaled_ss, xlim[0], xlim[1])
        c51.scatter_plot(np.arange(len(scaled_ss)), scaled_ss, '%s_%s ss scaled correlator (take sqrt to get Z0_s)' %(str(mq1),str(mq2)), xlim = xlim, ylim = ylim)
        ylim = c51.find_yrange(scaled_ps, xlim[0], xlim[1])
        c51.scatter_plot(np.arange(len(scaled_ps)), scaled_ps, '%s_%s ps scaled correlator (divide by Z0_s to get Z0_p)' %(str(mq1),str(mq2)), xlim = xlim, ylim = ylim)
        plt.show()
    # concatenate data
    boot0 = np.concatenate((SS, PS), axis=1)
    # read priors
    prior = mesp['priors']
    # read trange
    trange = mesp['trange']
    # fit boot0
    boot0gv = c51.make_gvars(boot0)
    boot0p = c51.dict_of_tuple_to_gvar(prior)
    fitfcn = c51.fit_function(T,params['decay_ward_fit']['nstates'])
    boot0fit = c51.fitscript_v2(trange,T,boot0gv,boot0p,fitfcn.twopt_fitfcn_ss_ps)
    if params['flags']['stability_plot']:
        c51.stability_plot(boot0fit,'E0','%s_%s' %(str(mq1),str(mq2)))
        plt.show()
    if params['flags']['tabulate']:
        tbl_print = collections.OrderedDict()
        tbl_print['tmin'] = boot0fit['tmin']
        tbl_print['tmax'] = boot0fit['tmax']
        tbl_print['E0'] = [boot0fit['pmean'][t]['E0'] for t in range(len(boot0fit['pmean']))]
        tbl_print['dE0'] = [boot0fit['psdev'][t]['E0'] for t in range(len(boot0fit['pmean']))]
        tbl_print['Z0_s'] = [boot0fit['pmean'][t]['Z0_s'] for t in range(len(boot0fit['pmean']))]
        tbl_print['dZ0_s'] = [boot0fit['psdev'][t]['Z0_s'] for t in range(len(boot0fit['pmean']))]
        tbl_print['Z0_p'] = [boot0fit['pmean'][t]['Z0_p'] for t in range(len(boot0fit['pmean']))]
        tbl_print['dZ0_p'] = [boot0fit['psdev'][t]['Z0_p'] for t in range(len(boot0fit['pmean']))]
        tbl_print['chi2/dof'] = np.array(boot0fit['chi2'])/np.array(boot0fit['dof'])
        tbl_print['logGBF'] = boot0fit['logGBF']
        print tabulate(tbl_print, headers='keys')
    # submit boot0 to db
    if params['flags']['write']:
        corr_lst = [mesp['meta_id']['SS'],mesp['meta_id']['PS']]
        fit_id = c51.select_fitid('meson',params)
        for t in range(len(boot0fit['tmin'])):
            init_id = psql.initid(boot0fit['p0'][t])
            prior_id = psql.priorid(boot0fit['prior'][t])
            tmin = boot0fit['tmin'][t]
            tmax = boot0fit['tmax'][t]
            result = c51.make_result(boot0fit,t)
            psql.submit_boot0('meson',corr_lst,fit_id,tmin,tmax,init_id,prior_id,result,params['flags']['update'])
    if len(boot0fit['tmin'])!=1 or len(boot0fit['tmax'])!=1:
        print "sweeping over time range"
        print "skipping bootstrap"
        return 0
    else: pass
    if params['flags']['write']:
        bsresult = []
        psql.chkbsprior(tag,stream,Nbs,boot0fit['prior'][0])
        for g in tqdm.tqdm(range(Nbs)):
            # make bs gvar dataset
            n = g+1
            bs_id,draw = psql.fetchonebs(nbs=n,Mbs=Mbs,ens=tag,stream=stream)
            bootn = boot0[draw]
            bootngv = c51.make_gvars(bootn)
            # read randomized priors
            bsprior_id,bsp = psql.bspriorid(tag,stream,g,boot0fit['prior'][0])
            bsp = c51.dict_of_tuple_to_gvar(bsp)
            # read initial guess
            init = c51.read_init(boot0fit,0) #{"mres":boot0fit['pmean'][0]['mres']}
            #Fit
            bsfit = c51.fitscript_v2(trange,T,bootngv,bsp,fitfcn.twopt_fitfcn_ss_ps,init)
            tmin = bsfit['tmin'][0]
            tmax = bsfit['tmax'][0]
            boot0_id = psql.select_boot0('meson',corr_lst,fit_id,tmin,tmax,init_id,prior_id)
            bsinit_id = psql.initid(bsfit['p0'][0])
            result = c51.make_result(bsfit,0) #"""{"mres":%s, "chi2":%s, "dof":%s}""" %(bsfit['pmean'][t]['mres'],bsfit['chi2'][t],bsfit['dof'][t])
            psql.submit_bs('meson_bs',boot0_id,bs_id,Mbs,bsinit_id,bsprior_id,result,params['flags']['update'])
    return 0

def decay_constant(params, Z0_p, E0, mres_pion, mres_etas='pion'):
    ml = params.ml
    ms = params.ms
    if mres_etas == 'pion':
        constant = Z0_p*np.sqrt(2.)*(2.*ml+2.*mres_pion)/E0**(3./2.)
    else:
        constant = Z0_p*np.sqrt(2.)*(ml+ms+mres_pion+mres_etas)/E0**(3./2.)
    return constant

if __name__=='__main__':
    # read master
    user_flag = c51.user_list()
    f = open('./sqlmaster.yml.%s' %(user_flag),'r')
    params = yaml.load(f)
    f.close()
    # yaml entires
    fitmeta = params['decay_ward_fit']
    # log in sql
    psqlpwd = pwd.passwd()
    psql = sql.pysql('cchang5','cchang5',psqlpwd)
    # fit mres
    #mres_bs(psql,params,fitmeta['ml'])
    #mres_bs(psql,params,fitmeta['ms'])
    ## bootstrap decay constant
    #decay_bs(psql,params,'pion')
    decay_bs(psql,params,'kaon')
    ## calculate boot0 decay constant
    #fpi = decay_constant(params, decay_pion_proc.read_boot0('Z0_p'), decay_pion_proc.read_boot0('E0'), mres_pion_proc.read_boot0('mres'))
    #fk = decay_constant(params, decay_kaon_proc.read_boot0('Z0_p'), decay_kaon_proc.read_boot0('E0'), mres_pion_proc.read_boot0('mres'), mres_etas_proc.read_boot0('mres'))
    #ratio = fk/fpi
    #print 'fk/fpi:', ratio
    ## calculate bootstrap error
    #fpi_bs = decay_constant(params, decay_pion_proc.read_bs('Z0_p','on'), decay_pion_proc.read_bs('E0','on'), mres_pion_proc.read_bs('mres','on'))
    #fk_bs = decay_constant(params, decay_kaon_proc.read_bs('Z0_p','on'), decay_kaon_proc.read_bs('E0','on'), mres_pion_proc.read_bs('mres','on'), mres_etas_proc.read_bs('mres','on'))
    #plttbl = collections.OrderedDict()
    #plttbl['tmin'] = decay_pion_proc.tmin
    #plttbl['tmax'] = decay_pion_proc.tmax
    #plttbl['fk'] = fk
    #plttbl['fk_bserr'] = np.std(fk_bs, axis=0)
    #plttbl['fpi'] = fpi
    #plttbl['fpi_bserr'] = np.std(fpi_bs, axis=0)
    #plttbl['fk/fpi'] = fk/fpi
    #plttbl['fk/fpi_bserr(%)'] = np.std(fk_bs/fpi_bs, axis=0)*100
    #print tabulate(plttbl, headers='keys')
    #if params.plot_hist_flag == 'on':
    #    fpi_bs = decay_constant(params, decay_pion_proc.read_bs('Z0_p'), decay_pion_proc.read_bs('E0'), mres_pion_proc.read_bs('mres'))
    #    fk_bs = decay_constant(params, decay_kaon_proc.read_bs('Z0_p'), decay_kaon_proc.read_bs('E0'), mres_pion_proc.read_bs('mres'), mres_etas_proc.read_bs('mres'))
    #    c51.histogram_plot(fpi_bs, xlabel='fpi')
    #    c51.histogram_plot(fk_bs, xlabel='fk')
    #plt.show()
