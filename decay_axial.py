import sys
sys.path.append('$HOME/c51/scripts/')
import c51lib as c51
import numpy as np
import matplotlib.pyplot as plt
import gvar as gv
import multiprocessing as multi
from tabulate import tabulate
import yaml
import collections

def decay_axial(params, meson, draws, plot_flag='off'):
    if meson == 'pion':
        shortname = 'll'
    elif meson == 'kaon':
        shortname = 'ls'
    ens = params['current_fit']['ens']
    ml = params['current_fit']['ml']
    ms = params['current_fit']['ms']
    loc = params[ens][str(ml)+'_'+str(ms)]['data_loc']
    # read data
    twopt_dat = c51.open_data(loc['file_loc'], loc[meson+'_corr'])
    twopt_ss = np.squeeze(twopt_dat[:,:,0,:])
    twopt_ps = np.squeeze(twopt_dat[:,:,1,:])
    axial_dat = np.squeeze(c51.open_data(loc['file_loc'], loc['axial_'+shortname]))
    T = len(twopt_ss[0])
    if plot_flag == 'on':
        # unfolded correlator data
        c51.scatter_plot(np.arange(len(twopt_ss[0])), c51.make_gvars(twopt_ss), 'two point no fold')
        c51.scatter_plot(np.arange(len(axial_dat[0])), c51.make_gvars(axial_dat), 'axial_'+shortname+' no fold')
    # fold data
    axial_dat = c51.fold(axial_dat, phase=-1.0)
    twopt_ss = c51.fold(twopt_ss, phase=1.0)
    twopt_ps = c51.fold(twopt_ps, phase=1.0)
    if plot_flag == 'on':
        # folded correlator data
        c51.scatter_plot(np.arange(len(twopt_ss[0])), c51.make_gvars(twopt_ss), 'two point folded')
        c51.scatter_plot(np.arange(len(axial_dat[0])), c51.make_gvars(axial_dat), 'axial_'+shortname+' folded')
        # effective mass
        eff = c51.effective_plots(T)
        meff = eff.effective_mass(c51.make_gvars(twopt_ss), 1, 'cosh')
        meff_axial = eff.effective_mass(c51.make_gvars(axial_dat)[1:], 1, 'cosh')
        xlim = [3, len(meff)]
        ylim = c51.find_yrange(meff, xlim[0], xlim[1])
        c51.scatter_plot(np.arange(len(meff)), meff, 'two pt effective mass', xlim = xlim, ylim = ylim)
        ylim = c51.find_yrange(meff, xlim[0], xlim[1])
        c51.scatter_plot(np.arange(len(meff_axial))+1, meff_axial, 'axial_'+shortname+' effective mass', xlim = xlim, ylim = ylim)
        # scaled correlator
        E0 = dict()
        E0['pion'] = 0.165
        E0['kaon'] = 0.1
        scaled = eff.scaled_correlator(c51.make_gvars(twopt_ss), E0[meson], phase=1.0)
        scaled_axial = eff.scaled_correlator(c51.make_gvars(axial_dat)[1:], E0[meson], phase=-1.0)
        c51.scatter_plot(np.arange(len(scaled)), scaled, 'two pt scaled correlator (take sqrt to get Z0)')
        c51.scatter_plot(np.arange(len(scaled_axial))+1, scaled_axial, 'axial_'+shortname+' scaled correlator (divide by Z0 to get F0)')
    # concatenate
    decay_axial_ss = np.concatenate((axial_dat, twopt_ss), axis=1)
    # read priors
    priors = params[ens][str(ml)+'_'+str(ms)]['priors'][meson]
    # read trange
    trange  = params[ens][str(ml)+'_'+str(ms)]['trange']
    # fit
    args = ((g, trange, T, decay_axial_ss, priors, draws) for g in range(len(draws)))
    pool = multi.Pool()
    p = pool.map_async(decay_axial_fit, args)
    # sort via bootstrap number
    output = np.sort(np.array(p.get()), axis=0)
    return output

# bootstrap routine
def decay_axial_fit(args):
    g, trange, T, decay_axial_ss, priors, draws = args
    # resample
    decay_axial_ss = c51.make_gvars(decay_axial_ss[draws[g]])
    # make priors
    bsp = c51.dict_of_tuple_to_gvar(priors)
    # fit
    fitfcn = c51.fit_function(T, nstates=2)
    fit = c51.fitscript_v2(trange, T, decay_axial_ss, bsp, fitfcn.axial_twoptss_fitfcn, result_flag='off')
    # record nbs
    fit = [g, fit]
    return fit 

if __name__=='__main__':
    stab_plot_flag = 'off'
    bootstrap_histo_flag = 'on'
    # read parameters
    f = open('./decay_axial.yml','r')
    params = yaml.load(f)
    f.close()
    # bootstrap draws
    # if draw number = 0, draws boot0
    draw_n = 1000
    draws = c51.bs_draws(params, draw_n)
    # bootstrap axial_ll
    fpi = decay_axial(params, 'pion', draws, plot_flag='off')
    # bootstrap axial_ls
    fk = decay_axial(params, 'kaon', draws, plot_flag='off')
    # process output and plot distribution
    fpi_proc = c51.process_bootstrap(fpi)
    fk_proc = c51.process_bootstrap(fk)
    fpi_boot0, fpi_bs = fpi_proc()
    fk_boot0, fk_bs = fk_proc()
    print fpi_boot0
    if bootstrap_histo_flag == 'on':
        c51.histogram_plot(fpi_bs, 'F0', 'pion F0')
        c51.histogram_plot(fpi_bs, 'E0', 'pion E0')
        c51.histogram_plot(fk_bs, 'F0', 'kaon F0')
        c51.histogram_plot(fk_bs, 'E0', 'kaon F0')
    # plot stability for boot0 fits
    if stab_plot_flag == 'on':
        c51.stability_plot(fpi_boot0, 'F0', 'boot0 fpi')
        c51.stability_plot(fpi_boot0, 'E0', 'boot0 m_pi')
        c51.stability_plot(fk_boot0, 'F0', 'boot0 fk')
        c51.stability_plot(fk_boot0, 'E0', 'boot0 m_k')
    # boot0 result
    F0_pi = fpi_proc.read_boot0('F0') 
    E0_pi = fpi_proc.read_boot0('E0')
    F0_k = fk_proc.read_boot0('F0')
    E0_k = fk_proc.read_boot0('E0')
    # calculate decay constant
    fpi = -1.0*F0_pi*np.sqrt(2.0/E0_pi)
    fk = -1.0*F0_k*np.sqrt(2.0/E0_k)
    table_print = collections.OrderedDict()
    table_print['tmin'] = fpi_boot0['tmin']
    table_print['tmax'] = fpi_boot0['tmax']
    table_print['fk'] = fk
    table_print['fpi'] = fpi
    table_print['fk/fpi'] = fk/fpi
    print tabulate(table_print, headers='keys')
    # bootstrap errors
    F0_pi_bs_mean = fpi_proc.read_bs('F0') 
    E0_pi_bs_mean = fpi_proc.read_bs('E0')
    F0_k_bs_mean = fk_proc.read_bs('F0')
    E0_k_bs_mean = fk_proc.read_bs('E0')
    fpi_bserror = -1.0*F0_pi_bs_mean*np.sqrt(2.0/E0_pi_bs_mean)
    fk_bserror = -1.0*F0_k_bs_mean*np.sqrt(2.0/E0_k_bs_mean)
    if bootstrap_histo_flag == 'on':
        c51.histogram_plot(fpi_bserror, xlabel='fpi')
        c51.histogram_plot(fk_bserror, xlabel='fk')
    # result
    fpi_boot0_avg = np.average(fpi) 
    fpi_error = np.std(fpi_bserror)
    fk_boot0_avg = np.average(fk) 
    fk_error = np.std(fk_bserror)
    print 'fk: ', gv.gvar(fk_boot0_avg, fk_error)
    print 'fpi: ', gv.gvar(fpi_boot0_avg, fpi_error)
    fkfpi_boot0_avg = np.average(fk/fpi) 
    fkfpi_error = np.std(fk_bserror/fpi_bserror)
    print 'fk/fpi: ', gv.gvar(fkfpi_boot0_avg, fkfpi_error)
    plt.show()
