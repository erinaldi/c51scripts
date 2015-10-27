#Fit for decay constants
import h5py as h5
import c51lib as c51
import numpy as np
import gvar as gv
import matplotlib.pyplot as plt

def read_data(filename, dpath_pion, dpath_kaon, plot='off'):
    pion = c51.open_data(filename, dpath_pion)
    kaon = c51.open_data(filename, dpath_kaon)
    
    #Fold meson data
    pion = c51.fold(pion)
    kaon = c51.fold(kaon)

    pion_ss = c51.make_gvars(pion[:,:,0,0])
    pion_ps = c51.make_gvars(pion[:,:,1,0])
    kaon_ss = c51.make_gvars(kaon[:,:,0,0])
    kaon_ps = c51.make_gvars(kaon[:,:,1,0])
   
    if plot=='on':
        dat = pion_ps
        E0 = 0.237
        meff = c51.effective_mass(dat,1,'cosh')
        x = np.arange(len(meff))
        xlim = [4,15]
        ylim = c51.find_yrange(meff, xlim[0], xlim[1])
        c51.scatter_plot(x, meff, 'meff', xlim = xlim, ylim = ylim)
        scaled2pt = c51.scaled_correlator(dat, E0)
        ylim = c51.find_yrange(scaled2pt, xlim[0], xlim[1])
        c51.scatter_plot(x, scaled2pt, 'scaled 2pt', xlim=xlim, ylim=ylim)
        ylim = c51.find_yrange(dat, xlim[0], xlim[1])
        c51.scatter_plot(x, dat, 'data', xlim=xlim, ylim=ylim)
    else: pass

    pion_ss_ps = np.concatenate((pion[:,:,0,0],pion[:,:,1,0]), axis=1)
    kaon_ss_ps = np.concatenate((kaon[:,:,0,0],kaon[:,:,1,0]), axis=1)
    pion_ss_ps_gv = c51.make_gvars(pion_ss_ps)
    kaon_ss_ps_gv = c51.make_gvars(kaon_ss_ps)
    
    return pion_ss_ps_gv, kaon_ss_ps_gv

def decay(pion_ss_ps_gv, kaon_ss_ps_gv):
    prior = dict()
    prior['Z0_s'] = gv.gvar(0.025, 0.01)
    prior['Z1_s'] = gv.gvar(0.025, 0.035)
    prior['Z2_s'] = gv.gvar(0.025, 0.035)
    prior['Z0_p'] = gv.gvar(0.27, 0.15)
    prior['Z1_p'] = gv.gvar(0.27, 0.35)
    prior['Z2_p'] = gv.gvar(0.27, 0.35)
    prior['E0'] = gv.gvar(0.23, 0.2)
    prior['E1'] = gv.gvar(0.0, 1.0)
    prior['E2'] = gv.gvar(0.0, 1.0)
    trange = dict()
    trange['tmin'] = [6,6]
    trange['tmax'] = [20,20]
    T = len(pion_ss_ps_gv)
    fitfcn = c51.fit_function(T=T, nstates=2)
    fit = c51.fitscript(trange, pion_ss_ps_gv, prior, fitfcn.twopt_fitfcn_ss_ps, sets=2, result_flag='off')
    print "pion"
    c51.stability_plot(fit, 'Z0_p')
    c51.stability_plot(fit, 'E0')
    ml = 0.0158
    ms = 0.0902
    mres_pi = gv.gvar(0.0009633, 0.0000065)
    Z0_p = fit['post'][0]['Z0_p']
    E0 = fit['post'][0]['E0']
    fpi = Z0_p*np.sqrt(2.)*(2.*ml+2.*mres_pi)/E0**(3./2.)
    print 'fpi:', fpi

    print "kaon"
    prior['Z0_s'] = gv.gvar(0.02, 0.01)
    prior['Z1_s'] = gv.gvar(0.02, 0.03)
    prior['Z2_s'] = gv.gvar(0.02, 0.03)
    prior['Z0_p'] = gv.gvar(0.2, 0.1)
    prior['Z1_p'] = gv.gvar(0.2, 0.3)
    prior['Z2_p'] = gv.gvar(0.2, 0.3)
    prior['E0'] = gv.gvar(0.404, 0.2)
    prior['E1'] = gv.gvar(0.0, 1.0)
    prior['E2'] = gv.gvar(0.0, 1.0)
    fit = c51.fitscript(trange, kaon_ss_ps_gv, prior, fitfcn.twopt_fitfcn_ss_ps, sets=2, result_flag='off')
    c51.stability_plot(fit, 'Z0_p')
    c51.stability_plot(fit, 'E0')
    mres_kaon = gv.gvar(0.0006685, 0.0000044)
    Z0_p = fit['post'][0]['Z0_p']
    E0 = fit['post'][0]['E0']
    fk = Z0_p*np.sqrt(2.)*(ml+ms+mres_pi+mres_kaon)/E0**(3./2.)
    print 'fk:', fk
    fkfpi = fk/fpi
    print 'fk/fpi:', fkfpi

if __name__=='__main__':
    filename = '/home/cchang5/c51/data/l1648f211b580m013m065m838a_avg.h5'
    dpath_pion = 'l1648f211b580m013m065m838a/wf1p0_m51p3_l512_a52p0_smrw4p5_n60/spec_ml0p0158_ms0p0902/pion/corr'
    dpath_kaon = 'l1648f211b580m013m065m838a/wf1p0_m51p3_l512_a52p0_smrw4p5_n60/spec_ml0p0158_ms0p0902/kaon/corr'
    pion, kaon = read_data(filename, dpath_pion, dpath_kaon, 'on')
    decay(pion, kaon)

    plt.show()
