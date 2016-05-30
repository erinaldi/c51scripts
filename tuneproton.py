import sys
sys.path.append('$HOME/c51/scripts/')
import c51lib as c51
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
import gvar as gv
import multiprocessing as multi
from tabulate import tabulate
import yaml
import collections

def read_proton(pr):
    # read data
    p_up = c51.open_data(pr.data_loc['file_loc'], pr.data_loc['proton_spin_up'])
    p_dn = c51.open_data(pr.data_loc['file_loc'], pr.data_loc['proton_spin_dn'])
    pnp_up = c51.open_data(pr.data_loc['file_loc'], pr.data_loc['proton_np_spin_up'])
    pnp_dn = c51.open_data(pr.data_loc['file_loc'], pr.data_loc['proton_np_spin_dn'])
    p_savg = c51.ispin_avg(p_up, p_dn)
    pnp_savg = c51.ispin_avg(pnp_up, pnp_dn)
    p_avg = c51.parity_avg(p_savg, pnp_savg, phase=-1.0)
    T = len(p_avg[0])
    p_avg = p_avg[:, :T/2] # keep only first half of data ('folded' length)
    if pr.plot_data_flag == 'on':
        # folded correlator data
        p_ss = p_avg[:,:,0,0]
        p_ps = p_avg[:,:,3,0]
        c51.scatter_plot(np.arange(len(p_ss[0])), c51.make_gvars(p_ss), 'proton ss folded')
        c51.scatter_plot(np.arange(len(p_ps[0])), c51.make_gvars(p_ps), 'proton ps folded')
        # effective mass
        eff = c51.effective_plots(T)
        meff_ss = eff.effective_mass(c51.make_gvars(p_ss)[1:], 1, 'cosh')
        meff_ps = eff.effective_mass(c51.make_gvars(p_ps)[1:], 1, 'cosh')
        xlim = [1, len(meff_ss)] #*2/5]
        #ylim = [0.7, 1.1]
        ylim = [0.3, 0.9]
        #ylim = c51.find_yrange(meff_ss, xlim[0], xlim[1])
        c51.scatter_plot(np.arange(len(meff_ss))+1, meff_ss, 'proton ss effective mass', xlim = xlim, ylim = ylim)
        #ylim = c51.find_yrange(meff_ps, xlim[0], xlim[1])
        c51.scatter_plot(np.arange(len(meff_ps))+1, meff_ps, 'proton ps effective mass', xlim = xlim, ylim = ylim)
        # scaled correlator
        E0 = pr.priors['proton']['E0'][0]
        scaled_ss = eff.scaled_correlator(c51.make_gvars(p_ss), E0, phase=1.0)
        scaled_ps = eff.scaled_correlator(c51.make_gvars(p_ps), E0, phase=1.0)
        ylim = c51.find_yrange(scaled_ss, xlim[0], xlim[1]/2)
        c51.scatter_plot(np.arange(len(scaled_ss)), scaled_ss, 'proton ss scaled correlator (take sqrt to get Z0_s)', xlim = xlim, ylim = ylim)
        ylim = c51.find_yrange(scaled_ps, xlim[0], xlim[1]/2)
        c51.scatter_plot(np.arange(len(scaled_ps)), scaled_ps, 'proton ps scaled correlator (divide by Z0_s to get Z0_p)', xlim = xlim, ylim = ylim)
        plt.show()
    return p_avg, T

def fit_proton(pr, p_avg, T):
    # priors
    priors = c51.dict_of_tuple_to_gvar(pr.priors['proton'])
    # trange
    trange = pr.trange['proton']
    # make data
    # ss + ps
    p_data = np.concatenate((p_avg[:,:,0,0], p_avg[:,:,3,0]), axis=1)
    # ss
    #p_data = p_avg[:,:,0,0]
    # ps
    #p_data = p_avg[:,:,3,0]
    p_data = c51.make_gvars(p_data)
    # fit
    grandfit = dict()
    for n in range(2,3):
        fitfcn = c51.fit_function(T, nstates=n)
        # ss + ps
        fit = c51.fitscript_v2(trange, T, p_data, priors, fitfcn.twopt_fitfcn_ss_ps, pr.print_fit_flag)
        # ss
        #fit = c51.fitscript_v2(trange, T, p_data, priors, fitfcn.twopt_fitfcn_ss, pr.print_fit_flag)
        # ps
        #fit = c51.fitscript_v2(trange, T, p_data, priors, fitfcn.twopt_fitfcn_ps, pr.print_fit_flag)
        for k in fit.keys():
            try:
                grandfit[k] = np.concatenate((grandfit[k], fit[k]), axis=0)
            except:
                grandfit[k] = fit[k]
        try:
            grandfit['nstates'] = np.concatenate((grandfit['nstates'], n*np.ones(len(fit[k]))), axis=0)
        except:
            grandfit['nstates'] = n*np.ones(len(fit[k]))
    return np.array([[0, grandfit]])

if __name__=='__main__':
    # read params
    pr = c51.process_params()
    # read data
    p_avg, T = read_proton(pr)
    # fit
    fit = fit_proton(pr, p_avg, T)
    # process fit
    fit_proc = c51.process_bootstrap(fit)
    fit_boot0, fit_bs = fit_proc()
    if pr.plot_stab_flag == 'on':
        c51.stability_plot(fit_boot0, 'E0', 'proton E0 ')
        c51.stability_plot(fit_boot0, 'Z0_p', 'proton Z0_p ')
        c51.stability_plot(fit_boot0, 'Z0_s', 'proton Z0_s ')
    if pr.print_tbl_flag == 'on':
        tbl = c51.tabulate_result(fit_proc, ['Z0_s', 'Z0_p', 'E0'])
        print tbl
    #c51.heatmap(fit_proc.nstates, fit_proc.tmin, fit_proc.normbayesfactor, [0,1], 'Bayes Factor', 'nstates', 'tmin')
    #c51.heatmap(fit_proc.nstates, fit_proc.tmin, fit_proc.chi2dof, [0,3], 'chi2/dof', 'nstates', 'tmin')
    # nstate stability
    c51.nstate_stability_plot(fit_boot0, 'E0', 'proton E0 ')
    c51.nstate_stability_plot(fit_boot0, 'Z0_p', 'proton Z0_p ')
    c51.nstate_stability_plot(fit_boot0, 'Z0_s', 'proton Z0_s ')
    # model averaging
    #bma = c51.bayes_model_avg(fit_proc, ['Z0_p', 'Z0_s', 'E0'])
    #print bma
    # look at small t values
    #fcn_cls = c51.fit_function(T,nstates=1)
    #fitraw = fit[0,1]['rawoutput']
    #fit_y = fcn_cls.twopt_fitfcn_ss_ps(fitraw.x, fitraw.p)
    #data_y = fitraw.y
    #diff = data_y - fit_y
    #c51.scatter_plot(fitraw.x, diff[:len(diff)/2], title='ss difference', xlim=[fitraw.x[0]-1, fitraw.x[-1]+1])
    #c51.scatter_plot(fitraw.x, diff[len(diff)/2:], title='ps difference', xlim=[fitraw.x[0]-1, fitraw.x[-1]+1])
    plt.show()
