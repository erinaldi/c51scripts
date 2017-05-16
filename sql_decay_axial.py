import sys
sys.path.append('$HOME/c51/scripts/')
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
import sql_decay_ward as decay

def read_axial(psql,params,meson):
    ml = params['axial_fit']['ml']
    ms = params['axial_fit']['ms']
    print 'reading %s' %(meson)
    tag = params['axial_fit']['ens']['tag']
    stream = params['axial_fit']['ens']['stream']
    Nbs = params['axial_fit']['nbs']
    Mbs = params['axial_fit']['mbs']
    if meson == 'axial_ll':
        axialp = params[tag][meson]['%s_%s' %(ml, ml)]
    elif meson == 'axial_ls':
        axialp = params[tag][meson]['%s_%s' %(ml, ms)]
    # read data
    data= c51.fold(psql.data('dwhisq_corr_meson',axialp['meta_id']['PS']),-1)
    #data= psql.data('dwhisq_corr_meson',axialp['meta_id']['PS'])
    return data

def fit_axial(psql,params,dparams,meson,datagv,inherit_prior=None,flow=False):
    if meson=='axial_ll':
        mq1 = params['axial_fit']['ml']
        mq2 = mq1
        dmeson = 'pion'
    elif meson == 'axial_ls':
        mq1 = params['axial_fit']['ml']
        mq2 = params['axial_fit']['ms']
        dmeson = 'kaon'
    print 'fitting axial two point', mq1, mq2
    tag = params['axial_fit']['ens']['tag']
    stream = params['axial_fit']['ens']['stream']
    nstates = params['axial_fit']['nstates']
    axialp = params[tag][meson]['%s_%s' %(str(mq1),str(mq2))]
    mesonp = dparams[tag][dmeson]['%s_%s' %(str(mq1),str(mq2))]
    T = len(datagv)/3
    SS = datagv[:T]
    PS = datagv[T:2*T]
    ax = datagv[2*T:]
    # plot effective mass / scaled correlator
    if params['flags']['plot_data']:
        # unfolded correlator data
        c51.scatter_plot(np.arange(T), ax, '%s_%s ps' %(str(mq1),str(mq2)))
        plt.show()
        # effective mass
        eff = c51.effective_plots(T)
        meff_ps = eff.effective_mass(ax[1:T/2], 1, 'cosh')
        xlim = [2, len(meff_ps)/2-2]
        ylim = c51.find_yrange(meff_ps, xlim[0], xlim[1])
        c51.scatter_plot(np.arange(len(meff_ps)), meff_ps, '$%s %s$ ps effective mass' %(str(mq1),str(mq2)), xlim = xlim, ylim = ylim)
        plt.show()
        # scaled correlator
        E0 = mesonp['priors'][1]['E0'][0]
        print E0
        z_s = np.sqrt(eff.scaled_correlator_v2(SS, E0, phase=1.0))
        scaled_ax = eff.scaled_correlator_v2(ax, E0, phase=1.0)/z_s
        ylim = c51.find_yrange(scaled_ax, xlim[0], xlim[1])
        c51.scatter_plot(np.arange(len(scaled_ax)), scaled_ax, '$%s %s$ f0 scaled axial correlator' %(str(mq1),str(mq2)), xlim = xlim, ylim = ylim)
        plt.show()
    # read priors
    if inherit_prior==None:
        prior = c51.meson_priors(mesonp['priors'],nstates)
        axp = c51.meson_priors(axialp['priors'],nstates)
        for k in axp.keys():
            prior[k] = axp[k]
        prior = c51.dict_of_tuple_to_gvar(prior)
    else:
        prior = c51.meson_priors(mesonp['priors'],nstates)
        axp = c51.meson_priors(axialp['priors'],nstates)
        for k in axp.keys():
            prior[k] = axp[k]
        prior = c51.dict_of_tuple_to_gvar(prior)
        for k in inherit_prior.keys():
            prior[k] = inherit_prior[k]
    # read trange
    trange  = axialp['trange']
    dtrange = mesonp['trange']
    print "PRIOR:", prior
    # fit
    fitfcn = c51.fit_function(T,params['axial_fit']['nstates'])
    if flow:
        boot0fit = c51.fitscript_v3(dtrange,trange,T,datagv,prior,fitfcn.dwhisq_twopt_osc_axial,axial=True)
    else:
        boot0fit = c51.fitscript_v3(dtrange,trange,T,datagv,prior,fitfcn.dwhisq_twopt_axial,axial=True)
    #print boot0fit['rawoutput'][0]
    if params['flags']['stability_plot']:
        c51.stability_plot(boot0fit,'E0','%s_%s' %(str(mq1),str(mq2)))
        c51.stability_plot(boot0fit,'F0','%s_%s' %(str(mq1),str(mq2)))
        plt.show()
    if params['flags']['tabulate']:
        tbl_print = collections.OrderedDict()
        tbl_print['tmin'] = boot0fit['tmin']
        tbl_print['tmax'] = boot0fit['tmax']
        tbl_print['Fphi'] = [fphi(boot0fit['post'][t]['F0'], boot0fit['post'][t]['E0']).mean for t in range(len(boot0fit['pmean']))]
        tbl_print['dFphi'] = [fphi(boot0fit['post'][t]['F0'], boot0fit['post'][t]['E0']).sdev for t in range(len(boot0fit['pmean']))]
        tbl_print['E0'] = [boot0fit['pmean'][t]['E0'] for t in range(len(boot0fit['pmean']))]
        tbl_print['dE0'] = [boot0fit['psdev'][t]['E0'] for t in range(len(boot0fit['pmean']))]
        tbl_print['Z0_s'] = [boot0fit['pmean'][t]['Z0_s'] for t in range(len(boot0fit['pmean']))]
        tbl_print['dZ0_s'] = [boot0fit['psdev'][t]['Z0_s'] for t in range(len(boot0fit['pmean']))]
        tbl_print['Z0_p'] = [boot0fit['pmean'][t]['Z0_p'] for t in range(len(boot0fit['pmean']))]
        tbl_print['dZ0_p'] = [boot0fit['psdev'][t]['Z0_p'] for t in range(len(boot0fit['pmean']))]
        tbl_print['F0'] = [boot0fit['pmean'][t]['F0'] for t in range(len(boot0fit['pmean']))]
        tbl_print['dF0'] = [boot0fit['psdev'][t]['F0'] for t in range(len(boot0fit['pmean']))]
        tbl_print['F0*Z0_s'] = [(boot0fit['rawoutput'][t].p['F0']*boot0fit['rawoutput'][t].p['Z0_s']).mean for t in range(len(boot0fit['rawoutput']))]
        tbl_print['dF0*Z0_s'] = [(boot0fit['rawoutput'][t].p['F0']*boot0fit['rawoutput'][t].p['Z0_s']).sdev for t in range(len(boot0fit['rawoutput']))]
        tbl_print['chi2/dof'] = np.array(boot0fit['chi2'])/np.array(boot0fit['dof'])
        tbl_print['logGBF'] = boot0fit['logGBF']
        print tabulate(tbl_print, headers='keys')
    return {'axial_fit': boot0fit['rawoutput'][0]}

def fphi(gvF0, gvE0):
    return -1.0*gvF0*np.sqrt(1.0/gvE0)

if __name__=='__main__':
    # read parameters
    f = open('./axial_flow.yml','r')
    params = yaml.load(f)
    f.close()
    f = open('./decay_flow.yml','r')
    dparams = yaml.load(f)
    dparams['decay_ward_fit']['ens'] = params['axial_fit']['ens']
    dparams['decay_ward_fit']['ml'] = params['axial_fit']['ml']
    dparams['decay_ward_fit']['ms'] = params['axial_fit']['ms']
    f.close()

    # yaml entires
    fitmeta = params['axial_fit']
    # log in sql
    psqlpwd = pwd.passwd()
    psql = sql.pysql('cchang5','cchang5',psqlpwd)
    # read data
    ll = read_axial(psql, params, 'axial_ll')
    ls = read_axial(psql, params, 'axial_ls')
    # read two-point
    pionSS, pionPS = decay.read_decay_bs(psql,dparams,'pion')
    kaonSS, kaonPS = decay.read_decay_bs(psql,dparams,'kaon')
    # axial_ll
    print np.shape(pionSS), np.shape(pionPS), np.shape(ll)
    if params['flags']['fit_axial_ll']:
        ll = np.concatenate((pionSS, pionPS, ll), axis=1)
        llgv = c51.make_gvars(ll)
        fpi = fit_axial(psql,params,dparams,'axial_ll',llgv,flow=True)
        print fphi(fpi['axial_fit'].p['F0'],fpi['axial_fit'].p['E0'])
    # axial_ls
    if params['flags']['fit_axial_ls']:
        ls = np.concatenate((kaonSS, kaonPS, ls), axis=1)
        lsgv = c51.make_gvars(ls)
        fka = fit_axial(psql,params,dparams,'axial_ls',lsgv,flow=True)
        print fphi(fka['axial_fit'].p['F0'],fka['axial_fit'].p['E0'])
