# fit for gA of the proton
import sys
sys.path.append('$HOME/c51/scripts/')
import fitlib as c51
import numpy as np
import matplotlib.pyplot as plt
import gvar as gv
from tabulate import tabulate
import yaml
import collections
import copy
import tqdm


def gA_bs(psql,params):
    # read data
    mq = params['gA_fit']['ml']
    tag = params['gA_fit']['ens']['tag']
    stream = params['gA_fit']['ens']['stream']
    Nbs = params['gA_fit']['nbs']
    Mbs = params['gA_fit']['mbs']
    nstates = params['gA_fit']['nstates']
    barp = params[tag]['proton'][mq]
    print "fitting for gA mq %s, basak %s, ens %s%s, Nbs %s, Mbs %s" %(str(mq),str(basak),str(tag),str(stream),str(Nbs),str(Mbs))
    # read two point

    # READ YOUR DATA HERE IN WHATEVER FORM IT'S SAVED IN as a [cfg x t] numpy array
    # I named it as SS and PS for two smearing; if you want to keep the same notation you can add SP and PP

    T = len(SS[0]) #SET WHAT T IS (T LENGTH OF LATTICE)
    # concatenate and make gvars to preserve correlations
    boot0 = np.concatenate((SS, PS, fhSS, fhPS), axis=1) #Here concatenate the 4 src/snk combinations
    # make gvars
    boot0gv = c51.make_gvars(boot0)
    # plot data: This plots the folded data, meff and scaled two point (get rid of the dependence on basak operators or else the code will puke at you)
    # you can edit the bottom sections by following the notation to plot the sp and pp data if you want
    if params['flags']['plot_data']:
        # raw correlator
        c51.scatter_plot(np.arange(len(SS[0])), c51.make_gvars(SS), '%s %s ss' %(basak[b],str(mq)))
        c51.scatter_plot(np.arange(len(PS[0])), c51.make_gvars(PS), '%s %s ps' %(basak[b],str(mq)))
        plt.show()
        # effective mass
        eff = c51.effective_plots(T)
        meff_ss = eff.effective_mass(c51.make_gvars(SS), 1, 'log')
        meff_ps = eff.effective_mass(c51.make_gvars(PS), 1, 'log')
        xlim = [2, len(meff_ss)/3-2]
        ylim = c51.find_yrange(meff_ss, xlim[0], xlim[1])
        c51.scatter_plot(np.arange(len(meff_ss)), meff_ss, '%s %s ss effective mass' %(basak[b],str(mq)), xlim = xlim, ylim = ylim)
        ylim = c51.find_yrange(meff_ps, xlim[0], xlim[1])
        c51.scatter_plot(np.arange(len(meff_ps)), meff_ps, '%s %s ps effective mass' %(basak[b],str(mq)), xlim = xlim, ylim = ylim)
        plt.show()
        # scaled correlator
        E0 = barp['priors'][1]['E0'][0] # make sure your prior central value for E0 is accurate for these plots to make sense
        scaled_ss = eff.scaled_correlator(c51.make_gvars(SS), E0, phase=1.0)
        scaled_ps = eff.scaled_correlator(c51.make_gvars(PS), E0, phase=1.0)
        ylim = c51.find_yrange(scaled_ss, xlim[0], xlim[1])
        c51.scatter_plot(np.arange(len(scaled_ss)), scaled_ss, '%s %s ss scaled correlator (take sqrt to get Z0_s)' %(basak[b],str(mq)), xlim = xlim, ylim = ylim)
        ylim = c51.find_yrange(scaled_ps, xlim[0], xlim[1])
        c51.scatter_plot(np.arange(len(scaled_ps)), scaled_ps, '%s %s ps scaled correlator (divide by Z0_s to get Z0_p)' %(basak[b],str(mq)), xlim = xlim, ylim = ylim)
        plt.show()
    if params['flags']['fit_twopt']: ## +++THIS SECTION CHANGED+++
        # data already concatenated previously
        ## read trange
        trange = barp['trange']
        ## fit boot0
        boot0fit = c51.firscript_v2(trange,nstates,T,boot0gv,barp['priors']) #Change the fitfunction you enter in here to yours.
        if params['flags']['stability_plot']: #you can have it print different stability plots here
            c51.stability_plot(boot0fit,'E0','%s' %str(mq))
            plt.show()
        if params['flags']['tabulate']: #This tabulates the results.  Add your own columns for more values
            tbl_print = collections.OrderedDict()
            tbl_print['tmin'] = boot0fit['tmin']
            tbl_print['tmax'] = boot0fit['tmax']
            tbl_print['E0'] = [boot0fit['pmean'][t]['E0'] for t in range(len(boot0fit['pmean']))]
            tbl_print['dE0'] = [boot0fit['psdev'][t]['E0'] for t in range(len(boot0fit['pmean']))]
            tbl_print['chi2/dof'] = np.array(boot0fit['chi2'])/np.array(boot0fit['dof'])
            tbl_print['logGBF'] = boot0fit['logGBF']
            print tabulate(tbl_print, headers='keys')
    # bootstrap
    if len(boot0fit['tmin'])!=1 or len(boot0fit['tmax'])!=1: # This skips bootstrapping if you are scanning over tmin / tmax
        print "sweeping over time range"
        print "skipping bootstrap"
        return 0
    else: 
        # SET YOUR SEED HERE
        # import the random number generator that is used in other projects
        # so I never import a RNG because when I bootstrap, I actually save
        # the list of configuration number draws, and reuse the same list if
        # I want to preserve correlations.
        bsresult = []
        for g in tqdm.tqdm(range(Nbs)):
            # make bs gvar dataset
            n = g+1
            # HERE DRAW YOUR BOOTSTRAPS
            # The output should be a numpy array of length Ncfg
            # of integer values which set what configuration to draw
            bootn = boot0[draw]
            bootngv = c51.make_gvars(bootn)
            # read randomized priors
            # HERE DEFINE YOUR PRIORS (or read them in from master.yml)
            # The priors are a dictionary of tuples. Example: {'E0': [0.8, 0.3], 'Z0': [1.0, 0.5]}
            # So the same as when reading from master.yml
            # you may also want to randomize their central values
            bsp = c51.dict_of_tuple_to_gvar(bsp) #this converts to the lsqfit datatype)
            # read initial guess
            init = c51.read_init(boot0fit,0) # this makes the initial guess equal to your boot0 fit posterior
            #Fit
            bsfit = c51.fitscript_v2(trange,T,bootngv,bsp,fitfcn.mres_fitfcn,init)
            tmin = bsfit['tmin'][0]
            tmax = bsfit['tmax'][0]
            result = c51.make_result(bsfit,0) #"""{"mres":%s, "chi2":%s, "dof":%s}""" %(bsfit['pmean'][t]['mres'],bsfit['chi2'][t],bsfit['dof'][t])
            # So try printing bsfit and/or result to see if those outputs are useful for you.
            # We can work on how you'd like this outputted if you have any ideas
if __name__=='__main__':
    # read master
    f = open('./master.yml' ,'r')
    params = yaml.load(f)
    f.close()
    # fit gA
    gA_bs(psql,params)
