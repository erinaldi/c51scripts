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

def read_baryon(pr):
    # read data
    baryon = pr.baryon2pt
    if baryon in ['proton','sigma','xi','lambda']:
        corr_up = c51.open_data(pr.data_loc['file_loc'], pr.data_loc[baryon+'_spin_up'])
        corr_dn = c51.open_data(pr.data_loc['file_loc'], pr.data_loc[baryon+'_spin_dn'])
        corrnp_up = c51.open_data(pr.data_loc['file_loc'], pr.data_loc[baryon+'_np_spin_up'])
        corrnp_dn = c51.open_data(pr.data_loc['file_loc'], pr.data_loc[baryon+'_np_spin_dn'])
        # spin avg
        corr = 0.5*(corr_up+corr_dn)
        corrnp = 0.5*(corrnp_up+corrnp_dn)
    elif baryon in ['delta','sigma_st','xi_st','omega']:
        corr_upup = c51.open_data(pr.data_loc['file_loc'], pr.data_loc[baryon+'_spin_upup'])
        corr_up = c51.open_data(pr.data_loc['file_loc'], pr.data_loc[baryon+'_spin_up'])
        corr_dn = c51.open_data(pr.data_loc['file_loc'], pr.data_loc[baryon+'_spin_dn'])
        corr_dndn = c51.open_data(pr.data_loc['file_loc'], pr.data_loc[baryon+'_spin_dndn'])
        corrnp_upup = c51.open_data(pr.data_loc['file_loc'], pr.data_loc[baryon+'_np_spin_upup'])
        corrnp_up = c51.open_data(pr.data_loc['file_loc'], pr.data_loc[baryon+'_np_spin_up'])
        corrnp_dn = c51.open_data(pr.data_loc['file_loc'], pr.data_loc[baryon+'_np_spin_dn'])
        corrnp_dndn = c51.open_data(pr.data_loc['file_loc'], pr.data_loc[baryon+'_np_spin_dndn'])
        # spin avg
        corr = 0.25*(corr_upup+corr_up+corr_dn+corr_dndn)
        corrnp = 0.25*(corrnp_upup+corrnp_up+corrnp_dn+corrnp_dndn)
    # parity avg
    corr_avg = c51.parity_avg(corr, corrnp, phase=-1.0)
    T = len(corr_avg[0])
    corr_avg = corr_avg[:, :T/2] # keep only first half of data ('folded' length)
    if pr.plot_data_flag == 'on':
        for src in range(len(corr_avg[0,0,0])):
            for snk in range(len(corr_avg[0,0])):
                # folded correlator data
                corr_i = corr_avg[:,:,snk,src]
                #c51.scatter_plot(np.arange(len(corr_i[0])), c51.make_gvars(corr_i), '%s snk_%s src_%s folded' %(baryon,str(snk),str(src)))
                # effective mass
                eff = c51.effective_plots(T)
                meff = eff.effective_mass(c51.make_gvars(corr_i)[1:], 1, 'cosh')
                xlim = [1, len(meff)]
                ylim = c51.find_yrange(meff, xlim[0], xlim[1]*2/5)
                c51.scatter_plot(np.arange(len(meff))+1, meff, '%s snk_%s src_%s effective mass' %(baryon,str(snk),str(src)), xlim = xlim, ylim = ylim)
                # scaled correlator
                E0 = pr.priors[baryon]['E0'][0]
                scaled = eff.scaled_correlator(c51.make_gvars(corr_i), E0, phase=1.0)
                ylim = c51.find_yrange(scaled, xlim[0], xlim[1]*2/5)
                c51.scatter_plot(np.arange(len(scaled)), scaled, '%s snk_%s src_%s scaled correlator (take sqrt to get Z0_s)' %(baryon,str(snk),str(src)), xlim = xlim, ylim = ylim)
                plt.show()
    return corr_avg, T

def chain_fit(pr, corr_avg, T):
    # parameters
    baryon = pr.baryon2pt
    nstates = pr.nstates
    priors = c51.dict_of_tuple_to_gvar(pr.priors['proton'])
    # trange
    trange = pr.trange[baryon]
    #p_data = c51.make_gvars(p_data)
    # fit
    grandfit = dict()
    fitfcn = c51.fit_function(T, nstates=nstates)
    snk_len = len(corr_avg[0,0])
    src_len = len(corr_avg[0,0,0])
    corr_ss = corr_avg[:,:,:snk_len/2,:]
    corr_ps = corr_avg[:,:,snk_len/2:,:]
    for snk in range(snk_len):
        for src in range(src_len):
            pass
    #    fit = c51.fitscript_v2(trange, T, p_data, priors, fitfcn.twopt_fitfcn_ss_ps, pr.print_fit_flag)
    #    # ss
    #    #fit = c51.fitscript_v2(trange, T, p_data, priors, fitfcn.twopt_fitfcn_ss, pr.print_fit_flag)
    #    # ps
    #    #fit = c51.fitscript_v2(trange, T, p_data, priors, fitfcn.twopt_fitfcn_ps, pr.print_fit_flag)
    #    for k in fit.keys():
    #        try:
    #            grandfit[k] = np.concatenate((grandfit[k], fit[k]), axis=0)
    #        except:
    #            grandfit[k] = fit[k]
    #    try:
    #        grandfit['nstates'] = np.concatenate((grandfit['nstates'], n*np.ones(len(fit[k]))), axis=0)
    #    except:
    #        grandfit['nstates'] = n*np.ones(len(fit[k]))
    return np.array([[0, grandfit]])

if __name__=='__main__':
    # read params
    pr = c51.process_params()
    # read data
    corr_avg, T = read_baryon(pr)
    # fit
    fit = chain_fit(pr, corr_avg, T)
    raise SystemExit
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
