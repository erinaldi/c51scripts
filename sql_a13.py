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

def corrfit(psql,params,meson):
    mq = params['a13_fit']['ml']
    print 'fitting a13', mq
    tag = params['a13_fit']['ens']['tag']
    stream = params['a13_fit']['ens']['stream']
    Nbs = params['a13_fit']['nbs']
    Mbs = params['a13_fit']['mbs']
    mesp = params[tag][meson][mq]
    # read data
    SS = c51.fold(psql.data('dwhisq_corr_meson',mesp['meta_id']['SS']))
    PS = c51.fold(psql.data('dwhisq_corr_meson',mesp['meta_id']['PS']))
    T = len(SS[0])
    # plot effective mass / scaled correlator
    if params['flags']['plot_data']:
        # unfolded correlator data
        c51.scatter_plot(np.arange(len(SS[0])), c51.make_gvars(SS), '%s ss folded' %(str(mq)))
        c51.scatter_plot(np.arange(len(PS[0])), c51.make_gvars(PS), '%s ps folded' %(str(mq)))
        plt.show()
        # effective mass
        eff = c51.effective_plots(T)
        meff_ss = eff.effective_mass(c51.make_gvars(SS), 1, 'cosh')
        meff_ps = eff.effective_mass(c51.make_gvars(PS), 1, 'cosh')
        xlim = [2, 12]
        ylim = [0, 2] #c51.find_yrange(meff_ss, xlim[0], xlim[1])
        c51.scatter_plot(np.arange(len(meff_ss)), meff_ss, '%s ss effective mass' %(str(mq)), xlim = xlim, ylim = ylim)
        ylim = c51.find_yrange(meff_ps, xlim[0], xlim[1])
        c51.scatter_plot(np.arange(len(meff_ps)), meff_ps, '%s ps effective mass' %(str(mq)), xlim = xlim, ylim = ylim)
        plt.show()
        # scaled correlator
        E0 = mesp['priors'][1]['E0'][0]
        scaled_ss = eff.scaled_correlator(c51.make_gvars(SS), E0, phase=1.0)
        scaled_ps = eff.scaled_correlator(c51.make_gvars(PS), E0, phase=1.0)
        ylim = c51.find_yrange(scaled_ss, xlim[0], xlim[1])
        c51.scatter_plot(np.arange(len(scaled_ss)), scaled_ss, '%s ss scaled correlator (take sqrt to get Z0_s)' %(str(mq)), xlim = xlim, ylim = ylim)
        ylim = c51.find_yrange(scaled_ps, xlim[0], xlim[1])
        c51.scatter_plot(np.arange(len(scaled_ps)), scaled_ps, '%s ps scaled correlator (divide by Z0_s to get Z0_p)' %(str(mq)), xlim = xlim, ylim = ylim)
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

if __name__=='__main__':
    # read master
    f = open('./a13master.yml','r')
    params = yaml.load(f)
    f.close()
    # yaml entires
    fitmeta = params['a13_fit']
    # log in sql
    psqlpwd = pwd.passwd()
    psql = sql.pysql('cchang5','cchang5',psqlpwd)
    # fit mres
    corrfit(psql,params,'a1_3')
