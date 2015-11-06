import sys
sys.path.append('$HOME/Physics/c51/scripts/')
import c51lib as c51
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
import gvar as gv
import multiprocessing as multi
from tabulate import tabulate
import yaml
import collections

def mres_bs(params, meson, draws):
    #Read data
    mp = c51.open_data(params.data_loc['file_loc'], params.data_loc['mres_'+meson+'_mp'])
    pp = c51.open_data(params.data_loc['file_loc'], params.data_loc['mres_'+meson+'_pp'])
    T = len(pp)
    mres_dat = mp/pp
    # plot mres
    if params.plot_data_flag == 'on':
        c51.scatter_plot(np.arange(len(mres_dat[0])), c51.make_gvars(mres_dat), meson+' folded mres')
    #Read priors
    prior = params.priors[meson]
    #Read trange
    print params.priors
    trange = params.trange['mres']
    print trange
    #Fit
    args = ((g, trange, T, mres_dat, prior, draws) for g in range(len(draws)))
    pool = multi.Pool()
    p = pool.map_async(mres_fit, args)
    # sort by bootstrap number
    output = np.sort(np.array(p.get()), axis=0)
    return output

def mres_fit(args):
    g, trange, T, mres_dat, prior, draws = args
    #Resample mres data
    mres_dat_bs = mres_dat[draws[g]]
    #mres_dat_bs = c51.make_gvars(mres_dat_bs)
    mres_mean = np.average(mres_dat_bs, axis=0)
    mres_sdev = np.std(mres_dat_bs, axis=0)
    mres_dat_bs = gv.gvar(mres_mean, mres_sdev)
    #Randomize priors
    bsp = c51.dict_of_tuple_to_gvar(prior) #{'mres': gv.gvar(prior[0]+prior[1]*np.random.randn(), prior[1])}
    #Fit
    fitfcn = c51.fit_function(T)
    fit = c51.fitscript_v2(trange, T, mres_dat_bs, bsp, fitfcn.mres_fitfcn, result_flag='off')
    result = [g, fit]
    return result

if __name__=='__main__':
    # read parameters
    params = c51.process_params()
    # generate bootstrap list
    draw_n = 0
    draws = params.bs_draws(draw_n)
    # bootstrap mres
    mres_pion_fit = mres_bs(params, 'pion', draws)
    #mres_etas_fit = mres_bs(params, 'etas', draws, mres_data_flag)
    # process bootstrap
    mres_pion_proc = c51.process_bootstrap(mres_pion_fit)
    #mres_etas_proc = c51.process_bootstrap(mres_etas_fit)
    # plot mres stability
    if params.plot_stab_flag == 'on':
        mres_pion_0, mres_pion_n = mres_pion_proc()
        #mres_etas_0, mres_etas_n = mres_etas_proc()
        c51.stability_plot(mres_pion_0, 'mres', 'pion mres')
        #c51.stability_plot(mres_etas_0, 'mres', 'etas mres')
        plt.show()
    # print results
    if params.print_tbl_flag == 'on':
        tbl_print = collections.OrderedDict()
        tbl_print['tmin'] = mres_pion_proc.tmin
        tbl_print['tmax'] = mres_pion_proc.tmax
        tbl_print['mres_pion_boot0'] = mres_pion_proc.read_boot0('mres')
        tbl_print['mres_pion_sdev'] = mres_pion_proc.read_boot0_sdev('mres')
        tbl_print['pion_chi2/dof'] = mres_pion_proc.chi2dof
        #tbl_print['mres_etas_boot0'] = mres_etas_proc.read_boot0('mres')
        #tbl_print['mres_etas_sdev'] = mres_etas_proc.read_boot0_sdev('mres')
        #tbl_print['etas_chi2/dof'] = mres_etas_proc.chi2dof
        print params['current_fit']['ens']
        print tabulate(tbl_print, headers='keys')
