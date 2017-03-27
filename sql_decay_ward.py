import sys
sys.path.append('$HOME/Physics/c51/scripts/')
import sqlc51lib as c51
import calsql as sql
import password_file as pwd
import numpy as np
import matplotlib.pyplot as plt
import gvar as gv
import multiprocessing as multi
from tabulate import tabulate
import yaml
import collections
import tqdm

def read_mres_bs(psql,params,mq):
    print 'reading mres', mq
    tag = params['decay_ward_fit']['ens']['tag']
    stream = params['decay_ward_fit']['ens']['stream']
    Nbs = params['decay_ward_fit']['nbs']
    Mbs = params['decay_ward_fit']['mbs'] 
    mresp = params[tag]['mres'][mq]
    # read data
    mp = psql.data('dwhisq_corr_jmu',mresp['meta_id']['mp'])
    pp = psql.data('dwhisq_corr_jmu',mresp['meta_id']['pp'])
    return mp, pp

def fit_mres_bs(psql,params,mq,gv_mp,gv_pp,bootn_fit=False,g=None):
    tag = params['decay_ward_fit']['ens']['tag']
    stream = params['decay_ward_fit']['ens']['stream']
    Nbs = params['decay_ward_fit']['nbs']
    Mbs = params['decay_ward_fit']['mbs']
    mresp = params[tag]['mres'][mq]
    T = len(gv_pp)
    boot0gv = gv_mp/gv_pp
    # plot mres
    if params['flags']['plot_data']:
        c51.scatter_plot(np.arange(len(boot0gv)), boot0gv, '%s mres' %str(mq))
        plt.show()
    # read priors
    if bootn_fit:
        if g == 0:
            prior = mresp['priors']
        else:
            prior = dict()
            for k in mresp['priors'].keys():
                prior[k] = [np.random.normal(mresp['priors'][k][0],mresp['priors'][k][1]), mresp['priors'][k][1]]
    else:
        prior = mresp['priors']
    # read init
    try:
        tstring = '%s_%s' %(mresp['trange']['tmin'][0], mresp['trange']['tmax'][0])
        f_init = open('./jmu_posterior/jmu_%s_%s.yml' %(tag,tstring), 'r')
        init = yaml.load(f_init)
        f_init.close()
    except:
        init = None
    # read trange
    trange = mresp['trange']
    # fit boot0
    boot0p = c51.dict_of_tuple_to_gvar(prior)
    fitfcn = c51.fit_function(T)
    boot0fit = c51.fitscript_v2(trange,T,boot0gv,boot0p,fitfcn.mres_fitfcn,init=init,bayes=params['flags']['bayes'])
    if params['flags']['boot0_update']:
        post = boot0fit['rawoutput'][0].pmean
        res_dump = dict()
        tstring = "%s_%s" %(trange['tmin'][0],trange['tmax'][0])
        for k in post.keys():
            res_dump[k] = float(post[k])
        f_dump = open('./jmu_posterior/jmu_%s_%s.yml' %(tag,tstring), 'w+')
        yaml.dump(res_dump, f_dump)
        f_dump.flush()
        f_dump.close()
    if params['flags']['stability_plot']:
        c51.stability_plot(boot0fit,'mres',str(mq))
        plt.show()
    if params['flags']['tabulate']:
        tbl_print = collections.OrderedDict()
        tbl_print['tmin'] = boot0fit['tmin']
        tbl_print['tmax'] = boot0fit['tmax']
        tbl_print['mres'] = [boot0fit['pmean'][t]['mres'] for t in range(len(boot0fit['pmean']))]
        tbl_print['sdev'] = [boot0fit['psdev'][t]['mres'] for t in range(len(boot0fit['pmean']))]
        tbl_print['chi2/dof'] = np.array(boot0fit['chi2'])/np.array(boot0fit['dof'])
        tbl_print['logGBF'] = boot0fit['logGBF']
        print tabulate(tbl_print, headers='keys')
    return {'mres_fit': boot0fit['rawoutput'][0]}

def read_decay_bs(psql,params,meson):
    if meson=='pion':
        mq1 = params['decay_ward_fit']['ml']
        mq2 = mq1
    elif meson == 'kaon':
        mq1 = params['decay_ward_fit']['ml']
        mq2 = params['decay_ward_fit']['ms']
    elif meson == 'etas':
        mq1 = params['decay_ward_fit']['ms']
        mq2 = mq1
    print 'reading meson two point', mq1, mq2
    tag = params['decay_ward_fit']['ens']['tag']
    stream = params['decay_ward_fit']['ens']['stream']
    mesp = params[tag][meson]['%s_%s' %(str(mq1),str(mq2))]
    # read data
    SS = c51.fold(psql.data('dwhisq_corr_meson',mesp['meta_id']['SS']))
    PS = c51.fold(psql.data('dwhisq_corr_meson',mesp['meta_id']['PS']))
    return SS, PS

def fit_decay_bs(psql,params,meson, gv_SS, gv_PS,bootn_fit=False,g=None):
    if meson=='pion':
        mq1 = params['decay_ward_fit']['ml']
        mq2 = mq1
    elif meson == 'kaon':
        mq1 = params['decay_ward_fit']['ml']
        mq2 = params['decay_ward_fit']['ms']
    elif meson == 'etas':
        mq1 = params['decay_ward_fit']['ms']
        mq2 = mq1
    tag = params['decay_ward_fit']['ens']['tag']
    stream = params['decay_ward_fit']['ens']['stream']
    nstates = params['decay_ward_fit']['nstates']
    mesp = params[tag][meson]['%s_%s' %(str(mq1),str(mq2))]
    T = len(gv_SS)
    # plot effective mass / scaled correlator
    if params['flags']['plot_data']:
        # unfolded correlator data
        c51.scatter_plot(np.arange(len(gv_SS)), gv_SS, '%s_%s ss folded' %(str(mq1),str(mq2)))
        c51.scatter_plot(np.arange(len(gv_PS)), gv_PS, '%s_%s ps folded' %(str(mq1),str(mq2)))
        plt.show()
        # effective mass
        eff = c51.effective_plots(T)
        meff_ss = eff.effective_mass(gv_SS, 1, 'cosh')
        meff_ps = eff.effective_mass(gv_PS, 1, 'cosh')
        xlim = [3, len(meff_ss)/2-2]
        ylim = c51.find_yrange(meff_ss, xlim[0], xlim[1])
        c51.scatter_plot(np.arange(len(meff_ss)), meff_ss, '%s_%s ss effective mass' %(str(mq1),str(mq2)), xlim = xlim, ylim = ylim)
        ylim = c51.find_yrange(meff_ps, xlim[0], xlim[1])
        c51.scatter_plot(np.arange(len(meff_ps)), meff_ps, '%s_%s ps effective mass' %(str(mq1),str(mq2)), xlim = xlim, ylim = ylim)
        plt.show()
        # scaled correlator
        E0 = mesp['priors'][1]['E0'][0]
        scaled_ss = np.sqrt(eff.scaled_correlator(gv_SS, E0, phase=1.0))
        scaled_ps = eff.scaled_correlator(gv_PS, E0, phase=1.0)/scaled_ss
        ylim = c51.find_yrange(scaled_ss, xlim[0], xlim[1])
        c51.scatter_plot(np.arange(len(scaled_ss)), scaled_ss, '%s_%s ss scaled correlator (take sqrt to get Z0_s)' %(str(mq1),str(mq2)), xlim = xlim, ylim = ylim)
        ylim = c51.find_yrange(scaled_ps, xlim[0], xlim[1])
        c51.scatter_plot(np.arange(len(scaled_ps)), scaled_ps, '%s_%s ps scaled correlator (divide by Z0_s to get Z0_p)' %(str(mq1),str(mq2)), xlim = xlim, ylim = ylim)
        plt.show()
    # concatenate data
    boot0gv = np.concatenate((gv_SS, gv_PS))
    # read priors
    if bootn_fit:
        if g == 0:
            priorn = mesp['priors']
        else:
            priorn = dict()
            for n in mesp['priors'].keys():
                priorn[n] = {}
                for k in mesp['priors'][n].keys():
                    priorn[n][k] = [np.random.normal(mesp['priors'][n][k][0],mesp['priors'][n][k][1]), mesp['priors'][n][k][1]]
    else:
        priorn = mesp['priors']
    prior = c51.meson_priors(priorn,nstates)
    # read init
    try:
        tstring = '%s_%s' %(mesp['trange']['tmin'][0], mesp['trange']['tmax'][0])
        f_init = open('./meson_posterior/meson_%s_%s.yml' %(tag,tstring), 'r')
        init = yaml.load(f_init)
        f_init.close()
    except:
        init = None
    # read trange
    trange = mesp['trange']
    # fit boot0
    boot0p = c51.dict_of_tuple_to_gvar(prior)
    fitfcn = c51.fit_function(T,params['decay_ward_fit']['nstates'])
    boot0fit = c51.fitscript_v2(trange,T,boot0gv,boot0p,fitfcn.twopt_fitfcn_ss_ps,init=init,bayes=params['flags']['bayes'])
    #boot0fit = c51.fitscript_v2(trange,T,boot0gv,boot0p,fitfcn.dwhisq_twopt_osc_ss_ps)
    #print boot0fit['rawoutput'][0]
    if params['flags']['boot0_update']:
        post = boot0fit['rawoutput'][0].pmean
        mes_dump = dict()
        tstring = "%s_%s" %(trange['tmin'][0],trange['tmax'][0])
        for k in post.keys():
            mes_dump[k] = float(post[k])
        f_dump = open('./meson_posterior/meson_%s_%s.yml' %(tag,tstring), 'w+')
        yaml.dump(mes_dump, f_dump)
        f_dump.flush()
        f_dump.close()
    if params['flags']['stability_plot']:
        c51.stability_plot(boot0fit,'E0','%s_%s' %(str(mq1),str(mq2)))
        plt.show()
    if params['flags']['tabulate']:
        tbl_print = collections.OrderedDict()
        tbl_print['tmin'] = boot0fit['tmin']
        tbl_print['tmax'] = boot0fit['tmax']
        tbl_print['E0'] = [boot0fit['pmean'][t]['E0'] for t in range(len(boot0fit['pmean']))]
        tbl_print['dE0'] = [boot0fit['psdev'][t]['E0'] for t in range(len(boot0fit['pmean']))]
        tbl_print['Z0_s'] = [boot0fit['pmean'][t]['Z0_s'] for t in range(len(boot0fit['pmean']))]
        tbl_print['dZ0_s'] = [boot0fit['psdev'][t]['Z0_s'] for t in range(len(boot0fit['pmean']))]
        tbl_print['Z0_p'] = [boot0fit['pmean'][t]['Z0_p'] for t in range(len(boot0fit['pmean']))]
        tbl_print['dZ0_p'] = [boot0fit['psdev'][t]['Z0_p'] for t in range(len(boot0fit['pmean']))]
        tbl_print['chi2/dof'] = np.array(boot0fit['chi2'])/np.array(boot0fit['dof'])
        tbl_print['logGBF'] = boot0fit['logGBF']
        print tabulate(tbl_print, headers='keys')
    return {'meson_fit': boot0fit['rawoutput'][0]}

def decay_constant(params, Z0_p, E0, mres_pion, mres_etas='pion'):
    ml = params['decay_ward_fit']['ml']
    if mres_etas == 'pion':
        constant = Z0_p*np.sqrt(2.)*(2.*ml+2.*mres_pion)/E0**(3./2.)
    else:
        ms = params['decay_ward_fit']['ms']
        constant = Z0_p*np.sqrt(2.)*(ml+ms+mres_pion+mres_etas)/E0**(3./2.)
    return constant

def concatgv(corr1, corr2):
    concat = np.concatenate((corr1,corr2),axis=1)
    concat_gv = c51.make_gvars(concat)
    corr1 = concat_gv[:len(concat_gv)/2]
    corr2 = concat_gv[len(concat_gv)/2:]
    return corr1, corr2

if __name__=='__main__':
    # read master
    f = open('./decay.yml','r')
    params = yaml.load(f)
    f.close()
    # yaml entires
    fitmeta = params['decay_ward_fit']
    # log in sql
    psqlpwd = pwd.passwd()
    psql = sql.pysql('cchang5','cchang5',psqlpwd)
    # fit mres
    # ml mres
    mpl, ppl = read_mres_bs(psql,params,fitmeta['ml'])
    gv_mp, gv_pp = concatgv(mpl, ppl)
    resl = fit_mres_bs(psql,params,fitmeta['ml'],gv_mp,gv_pp)
    print resl['mres_fit']
    #buffdict = gv.BufferDict()
    #buffdict = res['mres_fit'].p
    # ms mres
    #mps, pps = read_mres_bs(psql,params,fitmeta['ms'])
    #gv_mp, gv_pp = concatgv(mps, pps)
    #ress = fit_mres_bs(psql,params,fitmeta['ms'],gv_mp,gv_pp)
    #print ress['mres_fit']

    ## bootstrap decay constant
    ## fit pion
    SSpi, PSpi = read_decay_bs(psql,params,'pion')
    gv_SS, gv_PS = concatgv(SSpi, PSpi)
    resp = fit_decay_bs(psql,params,'pion', gv_SS, gv_PS)
    print resp['meson_fit']
    ## fit kaon
    #SSka, PSka = read_decay_bs(psql,params,'kaon')
    #gv_SS, gv_PS = concatgv(SSka, PSka)
    #resk = fit_decay_bs(psql,params,'kaon', gv_SS, gv_PS)
    ## fit etas
    ##SS, PS = read_decay_bs(psql,params,'etas')
    ##gv_SS, gv_PS = concatgv(SS, PS)
    ##res = fit_decay_bs(psql,params,'etas', gv_SS, gv_PS)
    ##print res['meson_fit']
    #
    ##print "decay constants"
    #fk = decay_constant(params, resk['meson_fit'].p['Z0_p'], resk['meson_fit'].p['E0'], resl['mres_fit'].p['mres'], ress['mres_fit'].p['mres']) / np.sqrt(2)
    #fp = decay_constant(params, resp['meson_fit'].p['Z0_p'], resp['meson_fit'].p['E0'], resl['mres_fit'].p['mres'], mres_etas='pion') / np.sqrt(2)
    #print "fk:", fk
    #print "fp:", fp
    #print "fk/fp:", fk/fp

    if params['flags']['bootstrap']:
        params['flags']['plot_data'] = False
        params['flags']['stability_plot'] = False
        params['flags']['tabulate'] = False
        Nbs = params['decay_ward_fit']['nbs']
        Nbsmin = params['decay_ward_fit']['nbsmin']
        bslist = psql.fetch_draws(params['decay_ward_fit']['ens']['tag'], Nbs)
        # mres l
        # update boot0
        if params['flags']['bootn_mresl']:
            params['flags']['boot0_update'] = True
            gv_mp, gv_pp = concatgv(mpl, ppl)
            fit_mres_bs(psql,params,fitmeta['ml'],gv_mp,gv_pp)
            params['flags']['boot0_update'] = False
            print "updated resl boot0"
            psql.insert_mres_v1_meta(params,'ml')
            for g in tqdm.tqdm(range(Nbsmin,Nbs+1)): # THIS FITS boot0 ALSO
                draws = np.array(bslist[g][1])
                mpn = mpl[draws,:]
                ppn = ppl[draws,:]
                gv_mpn, gv_ppn = concatgv(mpn, ppn)
                sfit = fit_mres_bs(psql,params,fitmeta['ml'],gv_mpn,gv_ppn,True,g)['mres_fit']
                p = sfit.pmean
                p['Q'] = sfit.Q
                psql.insert_mres_v1_result(params,g,len(draws),p,'ml')
        # mres s
        # update boot0
        if params['flags']['bootn_mress']:
            params['flags']['boot0_update'] = True
            gv_mp, gv_pp = concatgv(mps, pps)
            fit_mres_bs(psql,params,fitmeta['ms'],gv_mp,gv_pp)
            params['flags']['boot0_update'] = False
            print "updated ress boot0"
            psql.insert_mres_v1_meta(params,'ms')
            for g in tqdm.tqdm(range(Nbsmin,Nbs+1)): # THIS FITS boot0 ALSO
                draws = np.array(bslist[g][1])
                mpn = mps[draws,:]
                ppn = pps[draws,:]
                gv_mpn, gv_ppn = concatgv(mpn, ppn)
                sfit = fit_mres_bs(psql,params,fitmeta['ms'],gv_mpn,gv_ppn,True,g)['mres_fit']
                p = sfit.pmean
                p['Q'] = sfit.Q
                psql.insert_mres_v1_result(params,g,len(draws),p,'ms')
        # pion
        # update boot0
        if params['flags']['bootn_pion']:
            params['flags']['boot0_update'] = True
            gv_ss, gv_ps = concatgv(SSpi, PSpi)
            result = fit_decay_bs(psql,params,'pion',gv_ss,gv_ps)
            params['flags']['boot0_update'] = False
            print result['meson_fit']
            print "updated pion boot0"
            psql.insert_meson_v1_meta(params,'pion')
            for g in tqdm.tqdm(range(Nbsmin,Nbs+1)): # THIS FITS boot0 ALSO
                draws = np.array(bslist[g][1])
                SSn = SSpi[draws,:]
                PSn = PSpi[draws,:]
                gv_ssn, gv_psn = concatgv(SSn, PSn)
                sfit = fit_decay_bs(psql,params,'pion',gv_ssn,gv_psn,True,g)['meson_fit']
                p = sfit.pmean
                p['Q'] = sfit.Q
                psql.insert_meson_v1_result(params,g,len(draws),p,'pion')
