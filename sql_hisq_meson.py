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

def read_hisq_meson(psql,params,meson):
    if meson in ['phi_jj_5', 'phi_jj_I']:
        mq1 = params['hisq_fit']['ml']
        mq2 = mq1
    elif meson in ['phi_jr_5', 'phi_jr_I']:
        mq1 = params['hisq_fit']['ml']
        mq2 = params['hisq_fit']['ms']
    elif meson in ['phi_rr_5', 'phi_rr_I']:
        mq1 = params['hisq_fit']['ms']
        mq2 = mq1
    print 'reading meson two point', mq1, mq2
    tag = params['hisq_fit']['ens']['tag']
    stream = params['hisq_fit']['ens']['stream']
    mesp = params[tag][meson]['%s_%s' %(str(mq1),str(mq2))]
    # read data
    data = c51.fold(psql.data('dwhisq_corr_meson',mesp['meta_id']))
    return data

def fit_hisq_meson(psql,params,meson, datagv,bootn_fit=False,g=None):
    if meson in ['phi_jj_5', 'phi_jj_I']:
        mq1 = params['hisq_fit']['ml']
        mq2 = mq1
    elif meson in ['phi_jr_5', 'phi_jr_I']:
        mq1 = params['hisq_fit']['ml']
        mq2 = params['hisq_fit']['ms']
    elif meson in ['phi_rr_5', 'phi_rr_I']:
        mq1 = params['hisq_fit']['ms']
        mq2 = mq1
    print 'fitting two point', mq1, mq2
    tag = params['hisq_fit']['ens']['tag']
    stream = params['hisq_fit']['ens']['stream']
    nstates = params['hisq_fit']['nstates']
    mesp = params[tag][meson]['%s_%s' %(str(mq1),str(mq2))]
    T = len(datagv)
    # plot effective mass / scaled correlator
    if params['flags']['plot_data']:
        # unfolded correlator data
        c51.scatter_plot(np.arange(len(datagv)), datagv, '%s_%s ss' %(str(mq1),str(mq2)))
        plt.show()
        # effective mass
        eff = c51.effective_plots(T)
        meff = eff.effective_mass(datagv, 1, 'cosh')
        xlim = [3, len(meff)/2-2]
        ylim = c51.find_yrange(meff, xlim[0], xlim[1])
        c51.scatter_plot(np.arange(len(meff)), meff, '%s_%s effective mass' %(str(mq1),str(mq2)), xlim = xlim, ylim = ylim)
        plt.show()
        # scaled correlator
        E0 = mesp['priors'][1]['E0'][0]
        scaled = eff.scaled_correlator_v2(datagv, E0, phase=1.0)
        ylim = c51.find_yrange(scaled, xlim[0], xlim[1])
        c51.scatter_plot(np.arange(len(scaled)), scaled, '%s_%s scaled correlator (A0)' %(str(mq1),str(mq2)), xlim = xlim, ylim = ylim)
        plt.show()
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
        f_init = open('./hisq_posterior/hisq_%s_%s.yml' %(tag,tstring), 'r')
        init = yaml.load(f_init)
        f_init.close()
    except:
        init = None
    # read trange
    trange = mesp['trange']
    # fit boot0
    boot0gv = datagv
    boot0p = c51.dict_of_tuple_to_gvar(prior)
    fitfcn = c51.fit_function(T,params['hisq_fit']['nstates'])
    boot0fit = c51.fitscript_v2(trange,T,boot0gv,boot0p,fitfcn.mixed_twopt_osc,init=init,bayes=params['flags']['bayes'])
    if params['flags']['boot0_update']:
        print boot0fit['rawoutput'][0]
        post = boot0fit['rawoutput'][0].pmean
        mes_dump = dict()
        tstring = "%s_%s" %(trange['tmin'][0],trange['tmax'][0])
        for k in post.keys():
            mes_dump[k] = float(post[k])
        f_dump = open('./hisq_posterior/hisq_%s_%s.yml' %(tag,tstring), 'w+')
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
        tbl_print['A0'] = [boot0fit['pmean'][t]['A0'] for t in range(len(boot0fit['pmean']))]
        tbl_print['dA0'] = [boot0fit['psdev'][t]['A0'] for t in range(len(boot0fit['pmean']))]
        tbl_print['E1'] = [boot0fit['pmean'][t]['E1'] for t in range(len(boot0fit['pmean']))]
        tbl_print['dE1'] = [boot0fit['psdev'][t]['E1'] for t in range(len(boot0fit['pmean']))]
        tbl_print['A1'] = [boot0fit['pmean'][t]['A1'] for t in range(len(boot0fit['pmean']))]
        tbl_print['dA1'] = [boot0fit['psdev'][t]['A1'] for t in range(len(boot0fit['pmean']))]
        tbl_print['chi2/dof'] = [boot0fit['chi2'][t]/boot0fit['dof'][t] for t in range(len(boot0fit['pmean']))]
        tbl_print['logGBF'] = boot0fit['logGBF']
        print tabulate(tbl_print, headers='keys')
        if params['flags']['fitline_plot']:
            p = boot0fit['rawoutput'][0].p
            t = np.linspace(0, 30, 1000)
            # output fit curve
            fitline = fitfcn.mixed_twopt_osc(t,p)
            meff = np.arccosh( (np.roll(fitline,-1)+np.roll(fitline,1)) / (2*fitline) )
            f_plot = open('./hisq_fitline/%s_mixed_twopt.csv' %(tag), 'w+')
            string = 't, y, +-\n'
            for i in range(len(t)):
                string += '%s, %s, %s\n' %(t[i], meff[i].mean, meff[i].sdev)
            f_plot.write(string)
            f_plot.flush()
            f_plot.close()
    return {'hisq_meson_fit': boot0fit['rawoutput'][0]}

if __name__=='__main__':
    # read master
    f = open('./hisq.yml','r')
    params = yaml.load(f)
    f.close()
    # yaml entires
    fitmeta = params['hisq_fit']
    # log in sql
    psqlpwd = pwd.passwd()
    psql = sql.pysql('cchang5','cchang5',psqlpwd)

    ## bootstrap decay constant
    # fit phi_jj_5
    datajj = read_hisq_meson(psql,params,'phi_jj_5')
    datagv = c51.make_gvars(datajj)
    resp = fit_hisq_meson(psql,params,'phi_jj_5', datagv)
    # fit phi_jr_5
    #data = read_hisq_meson(psql,params,'phi_jr_5')
    #datagv = c51.make_gvars(data)
    #resp = fit_hisq_meson(psql,params,'phi_jr_5', datagv)
    # fit phi_rr_5
    datarr = read_hisq_meson(psql,params,'phi_rr_5')
    datagv = c51.make_gvars(datarr)
    resp = fit_hisq_meson(psql,params,'phi_rr_5', datagv)
    #print res['meson_fit']
    #buffdict.update(res['meson_fit'].p)

    if params['flags']['bootstrap']:
        params['flags']['plot_data'] = False
        params['flags']['stability_plot'] = False
        params['flags']['tabulate'] = False
        Nbs = params['hisq_fit']['nbs']
        Nbsmin = params['hisq_fit']['nbsmin']
        bslist = psql.fetch_draws(params['hisq_fit']['ens']['tag'], Nbs)
        # pion
        # update boot0
        if params['flags']['bootn_jj']:
            params['flags']['boot0_update'] = True
            gvdatajj = c51.make_gvars(datajj)
            result = fit_hisq_meson(psql,params,'phi_jj', gvdatajj)
            params['flags']['boot0_update'] = False
            print result['hisq_meson_fit']
            print "updated phi_jj boot0"
            psql.insert_mixed_v1_meta(params,'phi_ju')
            for g in tqdm.tqdm(range(Nbsmin,Nbs+1)): # THIS FITS boot0 ALSO
                draws = np.array(bslist[g][1])
                jjn = datajj[draws,:]
                gvjjn = gv.dataset.avg_data(jjn)
                sfit = fit_hisq_meson(psql,params,'phi_jj',gvjjn,True,g)['hisq_meson_fit']
                p = sfit.pmean
                p['Q'] = sfit.Q
                psql.insert_mixed_v1_result(params,g,len(draws),p,'phi_jj')
