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
    print 'mres', meson
    ens = params.ens
    ml = params.ml
    ms = params.ms
    loc = params.data_loc
    #Read data
    mp = c51.open_data(loc['file_loc'], loc['mres_'+meson+'_mp'])
    pp = c51.open_data(loc['file_loc'], loc['mres_'+meson+'_pp'])
    T = len(pp[0])
    mres_dat = mp/pp
    # plot mres
    if params.plot_data_flag == 'on':
        c51.scatter_plot(np.arange(len(mres_dat[0])), c51.make_gvars(mres_dat), meson+' mres')
        plt.show()
    #Read priors
    prior = params.priors[meson]
    #Read trange
    trange = params.trange['mres']
    #Fit
    #args = ((g, trange, T, mres_dat, prior, draws) for g in range(len(draws)))
    #pool = multi.Pool()
    #p = pool.map_async(mres_fit, args)
    # sort by bootstrap number
    #output = np.sort(np.array(p.get()), axis=0)
    result = []
    for g in range(len(draws)):
        mres_dat_bs = mres_dat[draws[g]]
        mres_dat_bs = c51.make_gvars(mres_dat_bs)
        #Randomize priors
        bsp = c51.dict_of_tuple_to_gvar(prior) #{'mres': gv.gvar(prior[0]+prior[1]*np.random.randn(), prior[1])}
        #Fit
        fitfcn = c51.fit_function(T)
        fit = c51.fitscript_v2(trange, T, mres_dat_bs, bsp, fitfcn.mres_fitfcn, result_flag='off')
        result.append([g, fit])
    output = np.sort(np.array(result), axis=0)
    return output

def mres_fit(args):
    g, trange, T, mres_dat, prior, draws = args
    print g
    #Resample mres data
    mres_dat_bs = mres_dat[draws[g]]
    mres_dat_bs = c51.make_gvars(mres_dat_bs)
    print 'bs'
    #Randomize priors
    bsp = c51.dict_of_tuple_to_gvar(prior) #{'mres': gv.gvar(prior[0]+prior[1]*np.random.randn(), prior[1])}
    #Fit
    fitfcn = c51.fit_function(T)
    fit = c51.fitscript_v2(trange, T, mres_dat_bs, bsp, fitfcn.mres_fitfcn, result_flag='off')
    result = [g, fit]
    return result

def decay_bs(params, meson, draws):
    print 'decay ', meson
    ens = params.ens
    ml = params.ml
    ms = params.ms
    loc = params.data_loc
    # read data
    decay_dat = c51.fold(c51.open_data(loc['file_loc'], loc['spec_'+meson]))
    decay_ss = np.squeeze(decay_dat[:,:,0,:])
    decay_ps = np.squeeze(decay_dat[:,:,1,:])
    T = 2*len(decay_ss[0])
    if params.plot_data_flag == 'on':
        # unfolded correlator data
        c51.scatter_plot(np.arange(len(decay_ss[0])), c51.make_gvars(decay_ss), meson+' ss folded')
        c51.scatter_plot(np.arange(len(decay_ps[0])), c51.make_gvars(decay_ps), meson+' ps folded')
        # effective mass
        eff = c51.effective_plots(T)
        meff_ss = eff.effective_mass(c51.make_gvars(decay_ss), 1, 'cosh')
        meff_ps = eff.effective_mass(c51.make_gvars(decay_ps), 1, 'cosh')
        xlim = [3, len(meff_ss)-2]
        ylim = c51.find_yrange(meff_ss, xlim[0], xlim[1])
        c51.scatter_plot(np.arange(len(meff_ss)), meff_ss, meson+' ss effective mass', xlim = xlim, ylim = ylim)
        ylim = c51.find_yrange(meff_ps, xlim[0], xlim[1])
        c51.scatter_plot(np.arange(len(meff_ps)), meff_ps, meson+' ps effective mass', xlim = xlim, ylim = ylim)
        # scaled correlator
        E0 = params.priors[meson]['E0'][0]
        scaled_ss = eff.scaled_correlator(c51.make_gvars(decay_ss), E0, phase=1.0)
        scaled_ps = eff.scaled_correlator(c51.make_gvars(decay_ps), E0, phase=1.0)
        ylim = c51.find_yrange(scaled_ss, xlim[0], xlim[1])
        c51.scatter_plot(np.arange(len(scaled_ss)), scaled_ss, meson+' ss scaled correlator (take sqrt to get Z0_s)', xlim = xlim, ylim = ylim)
        ylim = c51.find_yrange(scaled_ps, xlim[0], xlim[1])
        c51.scatter_plot(np.arange(len(scaled_ps)), scaled_ps, meson+' ps scaled correlator (divide by Z0_s to get Z0_p)', xlim = xlim, ylim = ylim)
        plt.show()
    # concatenate data
    decay_ss_ps = np.concatenate((decay_ss, decay_ps), axis=1)
    # priors
    priors = params.priors[meson]
    # read trange
    trange = params.trange['twopt']
    #Fit
    #args = ((g, trange, T, decay_ss_ps, priors, draws) for g in range(len(draws)))
    #pool = multi.Pool()
    #p = pool.map_async(decay_fit, args)
    ## sort via bootstrap number
    #output = np.sort(np.array(p.get()), axis=0)
    result = []
    for g in range(len(draws)):
        # resample decay data
        decay_bs = decay_ss_ps[draws[g]]
        decay_bs = c51.make_gvars(decay_bs)
        # priors
        bsp = c51.dict_of_tuple_to_gvar(priors)
        #Fit
        fitfcn = c51.fit_function(T, nstates=2)
        fit = c51.fitscript_v2(trange, T, decay_bs, bsp, fitfcn.twopt_fitfcn_ss_ps, result_flag='off')
        fit = [g, fit]
        result.append(fit)
    output = np.sort(np.array(result), axis=0)
    return output

def decay_fit(args):
    g, trange, T, decay_ss_sp_dat, priors, draws = args
    # resample decay data
    decay_bs = decay_ss_sp_dat[draws[g]]
    decay_bs = c51.make_gvars(decay_bs)
    # priors
    bsp = c51.dict_of_tuple_to_gvar(priors)
    #Fit
    fitfcn = c51.fit_function(T, nstates=2)
    fit = c51.fitscript_v2(trange, T, decay_bs, bsp, fitfcn.twopt_fitfcn_ss_ps, result_flag='off')
    fit = [g, fit]
    return fit

def decay_constant(params, Z0_p, E0, mres_pion, mres_etas='pion'):
    ml = params.ml
    ms = params.ms
    if mres_etas=='pion':
        constant = Z0_p*np.sqrt(2.)*(2.*ml+2.*mres_pion)/E0**(3./2.)
    else:
        constant = Z0_p*np.sqrt(2.)*(ml+ms+mres_pion+mres_etas)/E0**(3./2.)
    return constant

if __name__=='__main__':
    mres_data_flag = 'on'
    mres_tbl_flag = 'on'
    decay_constant_flag = 'off'
    decay_histogram_flag = 'on'
    # read parameters
    #f = open('./master.yml','r')
    #params = yaml.load(f)
    #f.close()
    params = c51.process_params()
    # generate bootstrap list
    draw_n = params.nbs
    draws = params.bs_draws(draw_n)
    # bootstrap mres
    mres_pion_fit = mres_bs(params, 'pion', draws)
    mres_etas_fit = mres_bs(params, 'etas', draws)
    # process bootstrap
    mres_pion_proc = c51.process_bootstrap(mres_pion_fit)
    mres_etas_proc = c51.process_bootstrap(mres_etas_fit)
    if params.print_fit_flag == 'on':
        print mres_pion_proc()[0]['rawoutput']
        print mres_etas_proc()[0]['rawoutput']
    # plot mres stability
    if params.plot_stab_flag == 'on':
        mres_pion_0, mres_pion_n = mres_pion_proc()
        mres_etas_0, mres_etas_n = mres_etas_proc()
        c51.stability_plot(mres_pion_0, 'mres', 'pion mres')
        c51.stability_plot(mres_etas_0, 'mres', 'etas mres')
        #plt.show()
    # print results
    if mres_tbl_flag == 'on':
        tbl_print = collections.OrderedDict()
        tbl_print['tmin'] = mres_pion_proc.tmin
        tbl_print['tmax'] = mres_pion_proc.tmax
        tbl_print['mres_pion_boot0'] = mres_pion_proc.read_boot0('mres')
        tbl_print['mres_pion_sdev'] = mres_pion_proc.read_boot0_sdev('mres')
        tbl_print['pion_chi2/dof'] = mres_pion_proc.chi2dof
        tbl_print['mres_etas_boot0'] = mres_etas_proc.read_boot0('mres')
        tbl_print['mres_etas_sdev'] = mres_etas_proc.read_boot0_sdev('mres')
        tbl_print['etas_chi2/dof'] = mres_etas_proc.chi2dof
        print tabulate(tbl_print, headers='keys')
    # bootstrap decay constant
    decay_pion_fit = decay_bs(params, 'pion', draws)
    decay_kaon_fit = decay_bs(params, 'kaon', draws)
    print decay_pion_fit[0][1]['rawoutput']
    print decay_kaon_fit[0][1]['rawoutput']
    # process bootstrap
    decay_pion_proc = c51.process_bootstrap(decay_pion_fit)
    decay_kaon_proc = c51.process_bootstrap(decay_kaon_fit)
    if params.plot_stab_flag == 'on':
        decay_pion_0, decay_pion_n = decay_pion_proc()
        decay_kaon_0, decay_kaon_n = decay_kaon_proc()
        c51.stability_plot(decay_pion_0, 'E0', 'pion E0')
        c51.stability_plot(decay_pion_0, 'Z0_p', 'pion Z0_p')
        c51.stability_plot(decay_kaon_0, 'E0', 'kaon E0')
        c51.stability_plot(decay_pion_0, 'Z0_s', 'pion Z0_s')
        plt.show()
    # calculate boot0 decay constant
    fpi = decay_constant(params, decay_pion_proc.read_boot0('Z0_p'), decay_pion_proc.read_boot0('E0'), mres_pion_proc.read_boot0('mres'))
    fk = decay_constant(params, decay_kaon_proc.read_boot0('Z0_p'), decay_kaon_proc.read_boot0('E0'), mres_pion_proc.read_boot0('mres'), mres_etas_proc.read_boot0('mres'))
    ratio = fk/fpi
    print 'fk/fpi:', ratio
    # calculate bootstrap error
    fpi_bs = decay_constant(params, decay_pion_proc.read_bs('Z0_p','on'), decay_pion_proc.read_bs('E0','on'), mres_pion_proc.read_bs('mres','on'))
    fk_bs = decay_constant(params, decay_kaon_proc.read_bs('Z0_p','on'), decay_kaon_proc.read_bs('E0','on'), mres_pion_proc.read_bs('mres','on'), mres_etas_proc.read_bs('mres','on'))
    plttbl = collections.OrderedDict()
    plttbl['tmin'] = decay_pion_proc.tmin
    plttbl['tmax'] = decay_pion_proc.tmax
    plttbl['fk'] = fk
    plttbl['fk_bserr'] = np.std(fk_bs, axis=0)
    plttbl['fpi'] = fpi
    plttbl['fpi_bserr'] = np.std(fpi_bs, axis=0)
    plttbl['fk/fpi'] = fk/fpi
    plttbl['fk/fpi_bserr(%)'] = np.std(fk_bs/fpi_bs, axis=0)*100
    print tabulate(plttbl, headers='keys')
    if decay_histogram_flag == 'on':
        fpi_bs = decay_constant(params, decay_pion_proc.read_bs('Z0_p'), decay_pion_proc.read_bs('E0'), mres_pion_proc.read_bs('mres'))
        fk_bs = decay_constant(params, decay_kaon_proc.read_bs('Z0_p'), decay_kaon_proc.read_bs('E0'), mres_pion_proc.read_bs('mres'), mres_etas_proc.read_bs('mres'))
        c51.histogram_plot(fpi_bs, xlabel='fpi')
        c51.histogram_plot(fk_bs, xlabel='fk')
    plt.show()
