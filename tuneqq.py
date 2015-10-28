#Fit for decay constants
import h5py as h5
import c51lib as c51
import numpy as np
import gvar as gv
import matplotlib.pyplot as plt

#Notes: for tuning fits, do single state no prior fit

def read_data(filename, dpath_pion, dpath_kaon, plot='off'):
    pion = c51.open_data(filename, dpath_pion)
    kaon = c51.open_data(filename, dpath_kaon)

    #Fold meson data
    pion = c51.fold(pion)
    kaon = c51.fold(kaon)

    pion_gv = c51.make_gvars(pion)
    kaon_gv = c51.make_gvars(kaon)

    if plot=='on':
        #Pion
        dat = pion_gv
        E0 = 0.13402
        meff = c51.effective_mass(dat,1,'cosh')
        x = np.arange(len(meff))
        xlim = [2,len(meff)-2]
        ylim = c51.find_yrange(meff, xlim[0], xlim[1])
        c51.scatter_plot(x, meff, 'pion meff', xlim = xlim, ylim = ylim)
        scaled2pt = c51.scaled_correlator(dat, E0)
        ylim = c51.find_yrange(scaled2pt, xlim[0], xlim[1])
        c51.scatter_plot(x, scaled2pt, 'pion scaled 2pt', xlim=xlim, ylim=ylim)
        #ylim = c51.find_yrange(dat, xlim[0], xlim[1])
        #c51.scatter_plot(x, dat, 'data', xlim=xlim, ylim=ylim)
        #Kaon
        dat = kaon_gv
        E0 = 0.41559
        meff = c51.effective_mass(dat,1,'cosh')
        x = np.arange(len(meff))
        ylim = c51.find_yrange(meff, xlim[0], xlim[1])
        c51.scatter_plot(x, meff, 'kaon meff', xlim = xlim, ylim = ylim)
        scaled2pt = c51.scaled_correlator(dat, E0)
        ylim = c51.find_yrange(scaled2pt, xlim[0], xlim[1])
        c51.scatter_plot(x, scaled2pt, 'kaon scaled 2pt', xlim=xlim, ylim=ylim)
    else: pass

    return pion_gv, kaon_gv

def decay(pion_gv, kaon_gv):
    prior = dict()
    prior['Z0_s'] = gv.gvar(0.0, 10.0)
    #prior['Z1_s'] = gv.gvar(0.06, 0.05)
    #prior['Z2_s'] = gv.gvar(0.025, 0.035)
    #prior['Z0_p'] = gv.gvar(0.27, 0.15)
    #prior['Z1_p'] = gv.gvar(0.27, 0.35)
    #prior['Z2_p'] = gv.gvar(0.27, 0.35)
    prior['E0'] = gv.gvar(0.13, 10.0)
    #prior['E1'] = gv.gvar(0.0, 1.0)
    #prior['E2'] = gv.gvar(0.0, 1.0)
    trange = dict()
    trange['tmin'] = [5,5]
    trange['tmax'] = [20,20]
    T = len(pion_gv)*2
    fitfcn = c51.fit_function(T=T, nstates=1)
    fit = c51.fitscript(trange, pion_gv, prior, fitfcn.twopt_fitfcn_ss, sets=1, result_flag='on')
    print "pion"
    #c51.stability_plot(fit, 'Z0_p')
    c51.stability_plot(fit, 'E0')
    #ml = 0.0158
    #ms = 0.0902
    #mres_pi = gv.gvar(0.0009633, 0.0000065)
    #Z0_p = fit['post'][0]['Z0_p']
    #E0 = fit['post'][0]['E0']
    #fpi = Z0_p*np.sqrt(2.)*(2.*ml+2.*mres_pi)/E0**(3./2.)
    #print 'fpi:', fpi

    print "kaon"
    prior['Z0_s'] = gv.gvar(0.0, 10.0)
    #prior['Z1_s'] = gv.gvar(0.037, 0.03)
    #prior['Z2_s'] = gv.gvar(0.02, 0.03)
    #prior['Z0_p'] = gv.gvar(0.2, 0.1)
    #prior['Z1_p'] = gv.gvar(0.2, 0.3)
    #prior['Z2_p'] = gv.gvar(0.2, 0.3)
    prior['E0'] = gv.gvar(0.415, 10.0)
    #prior['E1'] = gv.gvar(0.0, 1.0)
    #prior['E2'] = gv.gvar(0.0, 1.0)
    trange['tmin'] = [5,5]
    trange['tmax'] = [20,20]
    fitfcn = c51.fit_function(T=T, nstates=1)
    fit = c51.fitscript(trange, kaon_gv, prior, fitfcn.twopt_fitfcn_ss, sets=1, result_flag='on')
    #c51.stability_plot(fit, 'Z0_p')
    c51.stability_plot(fit, 'E0')
    #mres_kaon = gv.gvar(0.0006685, 0.0000044)
    #Z0_p = fit['post'][0]['Z0_p']
    #E0 = fit['post'][0]['E0']
    #fk = Z0_p*np.sqrt(2.)*(ml+ms+mres_pi+mres_kaon)/E0**(3./2.)
    #print 'fk:', fk
    #fkfpi = fk/fpi
    #print 'fk/fpi:', fkfpi

if __name__=='__main__':
    #filename = '/home/cchang5/c51/data/l3264f211b600m00507m0507m628a_avg.h5'
    #dpath_pion = 'l3264f211b600m00507m0507m628/wf1p0_m51p2_l512_a52p0_smrw5p5_n75/phi_qq/mq0p00600/corr'
    #dpath_kaon = 'l3264f211b600m00507m0507m628/wf1p0_m51p2_l512_a51p8_smrw5p5_n75/phi_qq/mq0p0693/corr'
    filename = '/home/cchang5/c51/data/l3248f211b580m00235m0647m831a_avg.h5'
    dpath_pion = 'l3248f211b580m00235m0647m831a/wf1p0_m51p3_l524_a53p5_smrw4p5_n60/phi_qq/mq0p00211/corr'
    dpath_kaon = 'l3248f211b580m00235m0647m831a/wf1p0_m51p3_l524_a53p5_smrw4p5_n60/phi_qq/mq0p00211/corr'
    #filename = '/Users/cchang5/c51/data/l2464f211b600m0102m0509m635a_avg.h5'
    #dpath_pion = 'l2464f211b600m0102m0509m635/wf1p0_m51p2_l58_a51p5_smrw5p0_n75/spectrum/ml0p0126_ms0p0693/pion/corr'
    #dpath_kaon = 'l2464f211b600m0102m0509m635/wf1p0_m51p2_l58_a51p5_smrw5p0_n75/spectrum/ml0p0126_ms0p0693/kaon/corr'
    pion, kaon = read_data(filename, dpath_pion, dpath_kaon, 'on')
    decay(pion, kaon)

    plt.show()
