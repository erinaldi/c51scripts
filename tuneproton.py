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

def read_data(params, baryon, plot_flag='off'):
    ens = params['current_fit']['ens']
    ml = params['current_fit']['ml']
    ms = params['current_fit']['ms']
    loc = params[ens][str(ml)+'_'+str(ms)]['data_loc']
    # read data
    twopt_dat = c51.fold(c51.open_data(loc['file_loc'], loc['twopt_'+meson]))
    twopt_ss = np.squeeze(twopt_dat[:,:,0,:])
    twopt_ps = np.squeeze(twopt_dat[:,:,1,:])
    T = 2*len(twopt_ss[0])
    if plot_flag == 'on':
        # unfolded correlator data
        c51.scatter_plot(np.arange(len(twopt_ss[0])), c51.make_gvars(twopt_ss), meson+' ss folded')
        c51.scatter_plot(np.arange(len(twopt_ps[0])), c51.make_gvars(twopt_ps), meson+' ps folded')
        # effective mass
        eff = c51.effective_plots(T)
        meff_ss = eff.effective_mass(c51.make_gvars(twopt_ss), 1, 'cosh')
        meff_ps = eff.effective_mass(c51.make_gvars(twopt_ps), 1, 'cosh')
        xlim = [3, len(meff_ss)]
        ylim = c51.find_yrange(meff_ss, xlim[0], xlim[1])
        c51.scatter_plot(np.arange(len(meff_ss)), meff_ss, meson+' ss effective mass', xlim = xlim, ylim = ylim)
        ylim = c51.find_yrange(meff_ps, xlim[0], xlim[1])
        c51.scatter_plot(np.arange(len(meff_ps)), meff_ps, meson+' ps effective mass', xlim = xlim, ylim = ylim)
        # scaled correlator
        E0 = dict()
        E0['pion'] = 0.165
        E0['kaon'] = 0.1
        scaled_ss = eff.scaled_correlator(c51.make_gvars(twopt_ss), E0[meson], phase=1.0)
        scaled_ps = eff.scaled_correlator(c51.make_gvars(twopt_ps), E0[meson], phase=1.0)
        c51.scatter_plot(np.arange(len(scaled_ss)), scaled_ss, meson+' ss scaled correlator (take sqrt to get Z0_s)')
        c51.scatter_plot(np.arange(len(scaled_ps)), scaled_ps, meson+' ps scaled correlator (divide by Z0_s to get Z0_p)')
    return twopt_ss, twopt_ps, T

if __name__=='__main__':
