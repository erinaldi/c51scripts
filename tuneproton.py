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

<<<<<<< HEAD
=======
##Read spin averaged positive parity proton
#filename = 'l3248f211b580m00235m0647m831a_avg.h5'
#path = 'l3248f211b580m00235m0647m831/wf1p0_m51p3_l524_a53p5_smrw4p5_n60/spectrum/ml0p00211_ms0p0902/'
#
#datapath = path+'proton/spin_up'
#data_up = c51.read_data(filename, datapath, 0, 0)
#datapath = path+'proton/spin_dn'
#data_dn = c51.read_data(filename, datapath, 0, 0)
#data_pos = c51.ispin_avg(data_up, data_dn)
##Read spin averaged negative parity proton
#datapath = path+'proton_np/spin_up'
#data_up = c51.read_data(filename, datapath, 0, 0)
#datapath = path+'proton_np/spin_dn'
#data_dn = c51.read_data(filename, datapath, 0, 0)
#data_neg = c51.ispin_avg(data_up, data_dn)
##Parity average
#data_ss = c51.parity_avg(data_pos, data_neg, -1)
#datapath = path+'proton/spin_up'
#data_up = c51.read_data(filename, datapath, 3, 0)
#datapath = path+'proton/spin_dn'
#data_dn = c51.read_data(filename, datapath, 3, 0)
#data_pos = c51.ispin_avg(data_up, data_dn)
##Read spin averaged negative parity proton
#datapath = path+'proton_np/spin_up'
#data_up = c51.read_data(filename, datapath, 3, 0)
#datapath = path+'proton_np/spin_dn'
#data_dn = c51.read_data(filename, datapath, 3, 0)
#data_neg = c51.ispin_avg(data_up, data_dn)
##Parity average
#data_ps = c51.parity_avg(data_pos, data_neg, -1)
#
#data = np.concatenate((data_ss, data_ps), axis=1)
#data = c51.make_gvars(data)
##data = data_ps
#
##Plot effective mass
#T = len(data)*0.5
#meff = c51.effective_mass(data, 1)
#x = np.arange(len(meff))
#ylim = c51.find_yrange(meff, 1, 10)
##ylim = [0.47, 0.57]
#xr = [1,15]
#c51.scatter_plot(x, meff, 'effective mass', xlim=[xr[0],xr[1]], ylim=ylim)
##ylim = c51.find_yrange(meff, 65, 79)
#c51.scatter_plot(x, meff, 'effective mass ps', xlim=[T+xr[0],T+xr[1]], ylim=ylim)
#
##Fit
#inputs = c51.read_yaml('temp.yml')
#prior = c51.dict_of_tuple_to_gvar(inputs['prior'])
#trange = inputs['trange']
#fitfcn = c51.fit_function(T, nstates=2)
#fit = c51.fitscript(trange, data, prior, fitfcn.twopt_fitfcn_ss_ps, sets=2, result_flag='on')
#c51.stability_plot(fit, 'E0')
#c51.stability_plot(fit, 'Z0_s')
#plt.show()

>>>>>>> merge_yaml_edit
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
        p_ps = p_avg[:,:,0,0]
        p_ss = p_avg[:,:,3,0]
        c51.scatter_plot(np.arange(len(p_ss[0])), c51.make_gvars(p_ss), 'proton ss folded')
        c51.scatter_plot(np.arange(len(p_ps[0])), c51.make_gvars(p_ps), 'proton ps folded')
        # effective mass
        eff = c51.effective_plots(T)
        meff_ss = eff.effective_mass(c51.make_gvars(p_ss)[1:], 1, 'cosh')
        meff_ps = eff.effective_mass(c51.make_gvars(p_ps)[1:], 1, 'cosh')
        xlim = [1, len(meff_ss)*4/5]
        ylim = c51.find_yrange(meff_ss, xlim[0], xlim[1])
        c51.scatter_plot(np.arange(len(meff_ss))+1, meff_ss, 'proton ss effective mass', xlim = xlim, ylim = ylim)
        ylim = c51.find_yrange(meff_ps, xlim[0], xlim[1])
        c51.scatter_plot(np.arange(len(meff_ps))+1, meff_ps, 'proton ps effective mass', xlim = xlim, ylim = ylim)
        # scaled correlator
        E0 = pr.priors['proton']['E0'][0]
        scaled_ss = eff.scaled_correlator(c51.make_gvars(p_ss), E0, phase=1.0)
        scaled_ps = eff.scaled_correlator(c51.make_gvars(p_ps), E0, phase=1.0)
        ylim = c51.find_yrange(scaled_ss, xlim[0], xlim[1])
        c51.scatter_plot(np.arange(len(scaled_ss)), scaled_ss, 'proton ss scaled correlator (take sqrt to get Z0_s)', xlim = xlim, ylim = ylim)
        ylim = c51.find_yrange(scaled_ps, xlim[0], xlim[1])
        c51.scatter_plot(np.arange(len(scaled_ps)), scaled_ps, 'proton ps scaled correlator (divide by Z0_s to get Z0_p)', xlim = xlim, ylim = ylim)
    return p_avg, T

def fit_proton(pr, p_avg, T):
    # priors
    priors = c51.dict_of_tuple_to_gvar(pr.priors['proton'])
    # trange
    trange = pr.trange['proton']
    # make data
    p_data = np.concatenate((p_avg[:,:,3,0], p_avg[:,:,0,0]), axis=1)
    p_data = c51.make_gvars(p_data)
    # fit
    fitfcn = c51.fit_function(T, nstates=2)
    fit = c51.fitscript_v2(trange, T, p_data, priors, fitfcn.twopt_fitfcn_ss_ps, pr.print_fit_flag)
    return np.array([[0, fit]])

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
    if pr.print_tbl_flag == 'on':
        tbl = collections.OrderedDict()
        tbl['tmin'] = fit_proc.tmin
        tbl['tmax'] = fit_proc.tmax
        tbl['Z0_s'] = fit_proc.read_boot0('Z0_s')
        tbl['Z0_s err'] = fit_proc.read_boot0_sdev('Z0_s')
        tbl['Z0_p'] = fit_proc.read_boot0('Z0_p')
        tbl['Z0_p err'] = fit_proc.read_boot0_sdev('Z0_p')
        tbl['E0'] = fit_proc.read_boot0('E0')
        tbl['E0 err'] = fit_proc.read_boot0_sdev('E0')
        tbl['chi2/dof'] = fit_proc.chi2dof
        print tabulate(tbl, headers='keys')
    plt.show()
