#Calculate mres
import c51lib as c51
import numpy as np
import h5py as h5
import gvar as gv
import matplotlib.pyplot as plt

def read_data(filename, dpath_pion_mp, dpath_pion_pp, dpath_etas_mp, dpath_etas_pp, plot='off'):
    pion_mp = c51.fold(c51.open_data(filename, dpath_pion_mp))
    pion_pp = c51.fold(c51.open_data(filename, dpath_pion_pp))
    etas_mp = c51.fold(c51.open_data(filename, dpath_etas_mp))
    etas_pp = c51.fold(c51.open_data(filename, dpath_etas_pp))

    #pion_mp = c51.open_data(filename, dpath_pion_mp)
    #pion_pp = c51.open_data(filename, dpath_pion_pp)
    #etas_mp = c51.open_data(filename, dpath_etas_mp)
    #etas_pp = c51.open_data(filename, dpath_etas_pp)

    pion = pion_mp/pion_pp
    etas = etas_mp/etas_pp

    pion_gv = c51.make_gvars(pion)
    etas_gv = c51.make_gvars(etas)

    pion_gv = gv.gvar(np.array([pion_gv[i].mean for i in range(len(pion_gv))]), np.array([pion_gv[j].sdev for j in range(len(pion_gv))]))
    etas_gv = gv.gvar(np.array([etas_gv[i].mean for i in range(len(pion_gv))]), np.array([etas_gv[j].sdev for j in range(len(pion_gv))]))

    if plot=='on':
        x = np.arange(len(pion_gv))
        c51.scatter_plot(x, pion_gv, title='pion mres')
        c51.scatter_plot(x, etas_gv, title='etas mres')
    else: pass

    return pion_gv, etas_gv

def mres(pion_gv, etas_gv):
    prior = dict()
    prior['mres'] = gv.gvar(0.0, 1.0)
    x = np.arange(len(pion_gv))
    
    c51.scatter_plot(x,pion_gv)
    trange = dict()
    T = len(pion_gv)-2
    trange['tmin'] = [2, 2]
    trange['tmax'] = [T, T]
    pionfit = c51.fitscript(trange, pion_gv, prior, c51.mres_fitfcn, result_flag='off')
    print "Pion"
    c51.stability_plot(pionfit,'mres','pion')

    c51.scatter_plot(x,etas_gv)
    etasfit = c51.fitscript(trange, etas_gv, prior, c51.mres_fitfcn, result_flag='off')
    print "Etas"
    c51.stability_plot(etasfit,'mres','kaon')
    #print kaonfit['post']
    return 0

if __name__=='__main__':
    ens = 'l3264f211b600m00507m0507m628'
    flow= '1p0'
    M5 = '1p2'
    L5 = '12'
    alpha5 = '2p0'
    smear_sigma = '5p5'
    smear_n = '75'
    val_ml = '0p00605'
    val_ms = '0p0693'
    filename = '/Users/cchang5/c51/data/l3264f211b600m00507m0507m628a_avg.h5'
    dpath_pion_mp = ens+'/wf'+flow+'_m5'+M5+'_l5'+L5+'_a5'+alpha5+'_smrw'+smear_sigma+'_n'+smear_n+'/dwf_jmu/mq'+val_ml+'/midpoint_pseudo'
    dpath_pion_pp = ens+'/wf'+flow+'_m5'+M5+'_l5'+L5+'_a5'+alpha5+'_smrw'+smear_sigma+'_n'+smear_n+'/dwf_jmu/mq'+val_ml+'/pseudo_pseudo'
    dpath_etas_mp = ens+'/wf'+flow+'_m5'+M5+'_l5'+L5+'_a5'+alpha5+'_smrw'+smear_sigma+'_n'+smear_n+'/dwf_jmu/mq'+val_ms+'/midpoint_pseudo'
    dpath_etas_pp = ens+'/wf'+flow+'_m5'+M5+'_l5'+L5+'_a5'+alpha5+'_smrw'+smear_sigma+'_n'+smear_n+'/dwf_jmu/mq'+val_ms+'/pseudo_pseudo'
    pion, etas = read_data(filename, dpath_pion_mp, dpath_pion_pp, dpath_etas_mp, dpath_etas_pp, 'on')
    mres(pion, etas)
    plt.show()
