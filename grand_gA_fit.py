# Fit the shit out of everything
import sys
sys.path.append('$HOME/c51/scripts/')
import gvar as gv
import cPickle as pickle
import calsql as sql
import sqlc51lib as c51
import password_file as pwd
import numpy as np
import yaml
# import various analysis scripts
import sql_decay_ward as decay
import dM as gA

if __name__=="__main__":
    # read yaml #
    # master
    f = open('./grand_gA_fit.yml' ,'r')
    params = yaml.load(f)
    f.close()
    # decay parameter file
    user_flag = c51.user_list()
    f = open('./sqlmaster.yml.%s' %(user_flag),'r')
    decay_params = yaml.load(f)
    f.close()
    decay_params['decay_ward_fit']['ens'] = params['grand_ensemble']['ens']
    decay_params['decay_ward_fit']['ml'] = params['grand_ensemble']['ml']
    # gA parameter file
    f = open('./fhprotonmaster.yml.%s' %(user_flag),'r')
    gA_params = yaml.load(f)
    f.close()
    gA_params['gA_fit']['ens'] = params['grand_ensemble']['ens']
    gA_params['gA_fit']['ml'] = params['grand_ensemble']['ml']
    # log in sql #
    psqlpwd = pwd.passwd()
    psql = sql.pysql('cchang5','cchang5',psqlpwd)
    # read data #
    # mres for ml
    mp, pp = decay.read_mres_bs(psql,decay_params,decay_params['decay_ward_fit']['ml'])
    # pion correlator
    ss_pion, ps_pion = decay.read_decay_bs(psql,decay_params,'pion')
    # gA parameter file
    gAboot0 = gA.read_gA_bs(psql,gA_params)
    # concatenate and build grand covariance matrix #
    # index
    x_mp = np.arange(len(mp[0]))
    x_pp = np.arange(len(pp[0]))+len(x_mp)
    x_ss_pion = np.arange(len(ss_pion[0]))+len(x_mp)+len(x_pp)
    x_ps_pion = np.arange(len(ps_pion[0]))+len(x_mp)+len(x_pp)+len(x_ss_pion)
    x_gAboot0 = np.arange(len(gAboot0[0]))+len(x_mp)+len(x_pp)+len(x_ss_pion)+len(x_ps_pion)
    print "len mp:", np.shape(mp)
    print "len pp:", np.shape(pp)
    print "len ss_pion:", np.shape(ss_pion)
    print "len ps_pion:", np.shape(ps_pion)
    print "len gAboot0:", np.shape(gAboot0)
    # concatenate
    boot0 = np.concatenate((mp,pp,ss_pion,ps_pion,gAboot0),axis=1)
    # correlated covariance
    gvboot0 = c51.make_gvars(boot0)
    # split to different project subsets
    gvmp = gvboot0[x_mp]
    gvpp = gvboot0[x_pp]
    gvss_pion = gvboot0[x_ss_pion]
    gvps_pion = gvboot0[x_ps_pion]
    gvgAboot0 = gvboot0[x_gAboot0]
    # fit #
    # mres
    mresl = decay.fit_mres_bs(psql,decay_params,decay_params['decay_ward_fit']['ml'],gvmp,gvpp)
    # meson
    pifit = decay.fit_decay_bs(psql,decay_params,'pion',gvss_pion,gvps_pion)
    # mN and gA
    #gAfit = gA.fit_gA(psql,gA_params,gvgAboot0)
    gAfit = gA.fit_proton(psql,gA_params,gvgAboot0)
    # chipt parameters
    priors = gv.BufferDict()
    priors['mpi'] = pifit['meson_fit'].p['E0']
    priors['fpi'] = decay.decay_constant(decay_params,pifit['meson_fit'].p['Z0_p'],pifit['meson_fit'].p['E0'],mresl['mres_fit'].p['mres'])
    #priors['mN'] = gAfit['gA_fit'].p['E0']
    priors['mN'] = gAfit['nucleon_fit'].p['E0']
    print priors
    #print gv.evalcorr([pifit['meson_fit'].p['E0'],decay.decay_constant(decay_params,pifit['meson_fit'].p['Z0_p'],pifit['meson_fit'].p['E0'],mresl['mres_fit'].p['mres']),gAfit['gA_fit'].p['E0'],gAfit['gA_fit'].p['gA00']])
    print priors['mN']/priors['fpi']
    data = gv.BufferDict()
    data['gA'] = gAfit['gA_fit'].p['gA00']
    # write output
    result = {params['grand_ensemble']['ens']['tag']:{'data':data, 'priors':priors}}
    #print result
    pickle.dump(result, open('./pickle_result/gA_%s.pickle' %params['grand_ensemble']['ens']['tag'], 'wb'))
    #g = pickle.load(open('./pickle_result/gA_%s.pickle' %params['grand_ensemble']['ens']['tag'], 'rb'))
    #print g
