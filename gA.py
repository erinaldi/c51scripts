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
    barp = params[tag]['proton'][mq]
    print "fitting for gA mq %s, basak %s, ens %s%s, Nbs %s, Mbs %s" %(str(mq),str(basak),str(tag),str(stream),str(Nbs),str(Mbs))
    # read two point
    SSl = np.array([psql.data('dwhisq_corr_baryon',idx) for idx in [barp['meta_id']['SS'][i] for i in basak]])
    PSl = np.array([psql.data('dwhisq_corr_baryon',idx) for idx in [barp['meta_id']['PS'][i] for i in basak]])
    T = len(SSl[0,0])
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
    # concatenate data
    SS = SSl[0]
    PS = PSl[0]
    for i in range(len(SSl)-1):
        SS = np.concatenate((SS,SSl[i+1]),axis=1)
        PS = np.concatenate((PS,PSl[i+1]),axis=1)
    boot0 = np.concatenate((SS, PS), axis=1)
    # read priors
    prior = c51.baryon_priors(barp['priors'],basak,nstates)
    ## read trange
    trange = barp['trange']
    ## fit boot0
    boot0gv = c51.make_gvars(boot0)
    boot0p = c51.dict_of_tuple_to_gvar(prior)
    fitfcn = c51.fit_function(T,nstates)
    boot0fit = c51.fitscript_v2(trange,T,boot0gv,boot0p,fitfcn.twopt_baryon_ss_ps,basak=params['gA_fit']['basak'])
    print boot0fit['rawoutput'][0]

if __name__=='__main__':
    # read master
    user_flag = c51.user_list()
    f = open('./fhprotonmaster.yml.%s' %(user_flag),'r')
    params = yaml.load(f)
    f.close()
    # yaml entires
    fitmeta = params['gA_fit']
    # log in sql
    psqlpwd = pwd.passwd()
    psql = sql.pysql('cchang5','cchang5',psqlpwd)
    # fit gA
    gA_bs(psql,params)
