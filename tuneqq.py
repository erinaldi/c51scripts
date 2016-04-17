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

def read_data(params):
    # read data
    twopt_dat = c51.fold(c51.open_data(params.data_loc['file_loc'], params.data_loc['phiqq_'+params.hadron]))
    T = 2*len(twopt_dat[0])
    if params.plot_data_flag == 'on':
        # unfolded correlator data
        c51.scatter_plot(np.arange(len(twopt_dat[0])), c51.make_gvars(twopt_dat), params.hadron+' folded')
        # effective mass
        eff = c51.effective_plots(T)
        meff = eff.effective_mass(c51.make_gvars(twopt_dat), 1, 'cosh')
        xlim = [2, len(meff)]
        ylim = c51.find_yrange(meff, xlim[0], xlim[1])
        ylim = [0.0, 0.7]
        c51.scatter_plot(np.arange(len(meff)), meff, params.hadron+' effective mass', xlim = xlim, ylim = ylim)
        # scaled correlator
        E0 = params.priors[params.hadron]['E0'][0]
        scaled = eff.scaled_correlator(c51.make_gvars(twopt_dat), E0, phase=1.0)
        c51.scatter_plot(np.arange(len(scaled)), scaled, params.hadron+' scaled correlator (A0 = Z0_s*Z0_p)')
    return twopt_dat, T

def fittwo_pt(params, twopt_dat, T):
    # make data
    twopt_gv = c51.make_gvars(twopt_dat)
    # priors
    priors = params.priors[params.hadron]
    priors_gv = c51.dict_of_tuple_to_gvar(priors)
    # read trange
    trange = params.trange['twopt']
    # fit
    fitfcn = c51.fit_function(T, nstates=1)
    fit = c51.fitscript_v2(trange, T, twopt_gv, priors_gv, fitfcn.twopt_fitfcn_phiqq, params.print_fit_flag)
    return np.array([[0, fit]])

if __name__=='__main__':
    # read parameters
    params = c51.process_params()
    # read data
    twopt_dat, T = read_data(params)
    # fit two point
    fit = fittwo_pt(params, twopt_dat, T)
    # process fit
    fit_proc = c51.process_bootstrap(fit)
    fit_boot0, fit_bs = fit_proc()
    if params.plot_stab_flag == 'on':
        c51.stability_plot(fit_boot0, 'E0', params.hadron+' E0 ')
        c51.stability_plot(fit_boot0, 'A0', params.hadron+' A0 ')
    if params.print_tbl_flag == 'on':
        tbl = c51.tabulate_result(fit_proc, ['A0', 'E0'])
        print tbl
    plt.show()
