import sys
sys.path.append('$HOME/c51/scripts/')
import h5py as h5
import c51lib as c51
import numpy as np
import gvar as gv
import matplotlib.pyplot as plt
from tabulate import tabulate
import yaml
import collections

#Notes: for tuning fits, do single state no prior fit

def read_data(params, meson, plot_flag='off'):
    ens = params['current_fit']['ens']
    ml = params['current_fit']['ml']
    ms = params['current_fit']['ms']
    loc = params[ens][str(ml)+'_'+str(ms)]['data_loc']
    # read data
    twopt_dat = c51.fold(c51.open_data(loc['file_loc'], loc['twopt_'+meson]))
    T = 2*len(twopt_dat[0])
    if plot_flag == 'on':
        # unfolded correlator data
        c51.scatter_plot(np.arange(len(twopt_dat[0])), c51.make_gvars(twopt_dat), meson+' folded')
        # effective mass
        eff = c51.effective_plots(T)
        meff = eff.effective_mass(c51.make_gvars(twopt_dat), 1, 'cosh')
        xlim = [3, len(meff)]
        ylim = c51.find_yrange(meff, xlim[0], xlim[1])
        c51.scatter_plot(np.arange(len(meff)), meff, meson+' effective mass', xlim = xlim, ylim = ylim)
        # scaled correlator
        E0 = dict()
        E0['pion'] = 0.09
        E0['kaon'] = 0.41
        scaled = eff.scaled_correlator(c51.make_gvars(twopt_dat), E0[meson], phase=1.0)
        c51.scatter_plot(np.arange(len(scaled)), scaled, meson+' scaled correlator (A0 = Z0_s*Z0_p)')
    return twopt_dat, T

def fittwo_pt(params, meson, twopt_dat, T):
    ens = params['current_fit']['ens']
    ml = params['current_fit']['ml']
    ms = params['current_fit']['ms']
    # make data
    twopt_gv = c51.make_gvars(twopt_dat)
    # priors
    priors = params[ens][str(ml)+'_'+str(ms)]['priors'][meson]
    priors_gv = c51.dict_of_tuple_to_gvar(priors)
    # read trange
    trange = params[ens][str(ml)+'_'+str(ms)]['trange']
    # fit
    fitfcn = c51.fit_function(T, nstates=1)
    fit = c51.fitscript_v2(trange, T, twopt_gv, priors_gv, fitfcn.twopt_fitfcn_phiqq, result_flag='off')
    return np.array([[0, fit]])

if __name__=='__main__':
    # switches
    meson = 'pion'
    meson = 'kaon'
    effective_plot_flag = 'off'
    stability_plot_flag = 'on'
    fit_tbl_flag = 'on'
    # read parameters
    f = open('./tuneqq.yml','r')
    params = yaml.load(f)
    f.close()
    # read data
    twopt_dat, T = read_data(params, meson, plot_flag=effective_plot_flag)
    # fit two point
    fit = fittwo_pt(params, meson, twopt_dat, T)
    # process fit
    fit_proc = c51.process_bootstrap(fit)
    fit_boot0, fit_bs = fit_proc()
    if stability_plot_flag == 'on':
        c51.stability_plot(fit_boot0, 'E0', meson+' E0 ')
    if fit_tbl_flag == 'on':
        tbl = collections.OrderedDict()
        tbl['tmin'] = fit_proc.tmin
        tbl['tmax'] = fit_proc.tmax
        tbl['A0'] = fit_proc.read_boot0('A0')
        tbl['A0 err'] = fit_proc.read_boot0_sdev('A0')
        tbl['E0'] = fit_proc.read_boot0('E0')
        tbl['E0 err'] = fit_proc.read_boot0_sdev('E0')
        tbl['chi2/dof'] = fit_proc.chi2dof
        print tabulate(tbl, headers='keys')
    plt.show()
