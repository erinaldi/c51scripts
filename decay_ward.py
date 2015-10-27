import c51lib as c51
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
import gvar as gv
import multiprocessing as multi
import mres
import decay
import yaml

def mres_bs(params, meson, draws):
    ens = params['current_fit']['ens']
    ml = params['current_fit']['ml']
    ms = params['current_fit']['ms']
    loc = params[ens]['data_loc']

    #Read data
    mp = c51.fold(c51.open_data(loc['file_loc'], loc['mres_'+meson+'_mp']))
    pp = c51.fold(c51.open_data(loc['file_loc'], loc['mres_'+meson+'_pp']))
    mres_dat = mp/pp

    #Read priors
    prior = params[ens][str(ml)+'_'+str(ms)]['mres_'+meson]

    #Read trange
    trange = params[ens]['trange']

    #Fit
    args = ((g, trange, mres_dat, prior, draws) for g in range(len(draws)))
    pool = multi.Pool()
    p = pool.map_async(mres_fit, args)
    output = np.array(p.get())
    output = np.transpose(output[np.argsort(output[:,0])]) #Sort by bootstrap number
    #Flatten into a 1D array
    output = np.ravel(output[1:])
    return output

def mres_fit(args):
    g, trange, mres_dat, prior, draws = args

    #Resample mres data
    mres_dat_bs = mres_dat[draws[g]]
    mres_dat_bs = c51.make_gvars(mres_dat_bs)

    #Randomize priors
    p = {'mres': gv.gvar(prior[0]+prior[1]*np.random.randn(), prior[1])}

    #Fit
    fit = c51.fitscript(trange, mres_dat_bs, p, c51.mres_fitfcn)
    result = np.append(g, [fit['post'][i]['mres'].mean for i in range(len(fit['tmax']))])
  
    return result

def decay_bs(params, meson, draws):
    ens = params['current_fit']['ens']
    ml = params['current_fit']['ml']
    ms = params['current_fit']['ms']
    loc = params[ens]['data_loc']

    #Read data
    decay_dat = c51.fold(c51.open_data(loc['file_loc'], loc['decay_'+meson]))
    decay_ss_sp_dat = np.concatenate((decay_dat[:,:,0,0], decay_dat[:,:,1,0]), axis=1)

    #Read prior
    priors = dict()
    ploc = params[ens][str(ml)+'_'+str(ms)]
    priors['Z0_s'] = ploc['Z0s_'+meson]
    priors['Z0_p'] = ploc['Z0p_'+meson]
    priors['Z1_s'] = ploc['Z1s_'+meson]
    priors['Z1_p'] = ploc['Z1p_'+meson]
    priors['E0'] = ploc['E0_'+meson]
    priors['E1'] = ploc['E1_'+meson]

    #Read trange
    trange = params[ens]['trange']

    #Fit
    args = ((g, trange, decay_ss_sp_dat, priors, draws) for g in range(len(draws)))
    pool = multi.Pool()
    p = pool.map_async(decay_fit, args)
    output = np.array(p.get())
    output_Z = output[:,0]
    output_E = output[:,1]
    #Can output pvalue/chi^2 here at output[:,2] if nessecary for scaling
    output_Z = np.transpose(output_Z[np.argsort(output_Z[:,0])]) #Sort by bootstrap number
    output_E = np.transpose(output_E[np.argsort(output_E[:,0])])
    #Flatten into a 1D array
    output_Z = np.ravel(output_Z[1:])
    output_E = np.ravel(output_E[1:])
    output = [output_Z, output_E]
    return output

def decay_fit(args):
    g, trange, decay_ss_sp_dat, priors, draws = args

    #Resample decay data
    decay_bs = decay_ss_sp_dat[draws[g]]
    decay_bs = c51.make_gvars(decay_bs)

    #Randomize priors
    bsp = dict()
    for key in priors.keys():
        #bsp[key] = gv.gvar(priors[key][0]+priors[key][1]*np.random.randn(), priors[key][1])
        bsp[key] = gv.gvar(priors[key][0], priors[key][1])
    #Fit
    fitfcn = c51.fit_function(T=len(decay_bs), nstates=2)
    fit = c51.fitscript(trange, decay_bs, bsp, fitfcn.twopt_fitfcn_ss_ps, sets=2)
    #result = [int(g), fit['post'][0]['Z0_p'].mean, fit['post'][0]['E0'].mean]
    result_Z = np.append(g, [fit['post'][i]['Z0_p'].mean for i in range(len(fit['tmax']))])
    result_E = np.append(g, [fit['post'][i]['E0'].mean for i in range(len(fit['tmax']))])
    return result_Z, result_E

def decay_constant(params, decay, mres_pion, mres_etas='pion'):
    ml = params['current_fit']['ml']
    ms = params['current_fit']['ms']
    Z0 = decay[0]
    mres_ll = mres_pion[1]
    E0 = decay[1]
    if mres_etas=='pion':
        constant = Z0*np.sqrt(2.)*(2.*ml+2.*mres_ll)/E0**(3./2.)
    else:
        mres_ss = mres_etas[1]
        constant = Z0*np.sqrt(2.)*(ml+ms+mres_ll+mres_ss)/E0**(3./2.)
    return constant

if __name__=='__main__':
    #Read parameters
    f = open('./decay_ward.yml','r')
    params = yaml.load(f)
    f.close()
    #Generate bootstrap list
    draw_n = 2
    draws = c51.bs_draws(params, draw_n)
    #Bootstrap mres
    mres_pion = mres_bs(params, 'pion', draws)
    mres_etas = mres_bs(params, 'etas', draws)
    print "mres_pion:", np.average(mres_pion)
    print "mres_etas:", np.average(mres_etas)

    #Bootstrap decay constant
    decay_pion = decay_bs(params, 'pion', draws)
    decay_kaon = decay_bs(params, 'kaon', draws)

    fpi = decay_constant(params, decay_pion, mres_pion)
    fk = decay_constant(params, decay_kaon, mres_pion, mres_etas)

    print "fk:", np.mean(fk), '+/-', np.std(fk)
    print "fpi:", np.mean(fpi), '+/-', np.std(fpi)

    ratio = fk/fpi
    ratio_mean = np.mean(ratio)
    ratio_sdev = np.std(ratio)
    print "fk/fpi:", ratio_mean, '+/-', ratio_sdev
    n, bins, patches = plt.hist(ratio)
    x = np.delete(bins, -1)
    plt.draw()
    plt.plot(x, n)
    plt.show()
