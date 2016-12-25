# Fit the shit out of the decay constants
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

if __name__=='__main__':
    # read yaml #
    f = open('./decay.yml','r')
    decay_params = yaml.load(f)
    f.close()
    ensemble = decay_params['decay_ward_fit']['ens']['tag']
    # log in sql #
    psqlpwd = pwd.passwd()
    psql = sql.pysql('cchang5','cchang5',psqlpwd)
    # read data #
    # mres for ml
    mpl, ppl = decay.read_mres_bs(psql,decay_params,decay_params['decay_ward_fit']['ml'])
    # ms
    mps, pps = decay.read_mres_bs(psql,decay_params,decay_params['decay_ward_fit']['ms'])
    # pion correlator
    ss_pion, ps_pion = decay.read_decay_bs(psql,decay_params,'pion')
    ss_kaon, ps_kaon = decay.read_decay_bs(psql,decay_params,'kaon')
    # index
    x_mpl = np.arange(len(mpl[0]))
    x_ppl = len(x_mpl)+np.arange(len(ppl[0]))
    x_mps = len(x_ppl)+len(x_mpl)+np.arange(len(mps[0]))
    x_pps = len(x_mps)+len(x_ppl)+len(x_mpl)+np.arange(len(pps[0]))
    x_ss_pion = len(x_pps)+len(x_mps)+len(x_ppl)+len(x_mpl)+np.arange(len(ss_pion[0]))
    x_ps_pion = len(x_ss_pion)+len(x_pps)+len(x_mps)+len(x_ppl)+len(x_mpl)+np.arange(len(ps_pion[0]))
    x_ss_kaon = len(x_ps_pion)+len(x_ss_pion)+len(x_pps)+len(x_mps)+len(x_ppl)+len(x_mpl)+np.arange(len(ss_kaon[0]))
    x_ps_kaon = len(x_ss_kaon)+len(x_ps_pion)+len(x_ss_pion)+len(x_pps)+len(x_mps)+len(x_ppl)+len(x_mpl)+np.arange(len(ps_kaon[0]))
    print "len mpl:", np.shape(mpl)
    print "len ppl:", np.shape(ppl)
    print "len mps:", np.shape(mps)
    print "len pps:", np.shape(pps)
    print "len ss_pion:", np.shape(ss_pion)
    print "len ps_pion:", np.shape(ps_pion)
    print "len ss_kaon:", np.shape(ss_kaon)
    print "len ps_kaon:", np.shape(ps_kaon)
    # concatenate
    boot0 = np.concatenate((mpl,ppl,mps,pps,ss_pion,ps_pion,ss_kaon,ps_kaon),axis=1)
    # correlated covariance
    gvboot0 = c51.make_gvars(boot0)
    # split for projects
    gvmpl = gvboot0[x_mpl]
    gvppl = gvboot0[x_ppl]
    gvmps = gvboot0[x_mps]
    gvpps = gvboot0[x_pps]
    gvss_pion = gvboot0[x_ss_pion]
    gvps_pion = gvboot0[x_ps_pion]
    gvss_kaon = gvboot0[x_ss_kaon]
    gvps_kaon = gvboot0[x_ps_kaon]
    # fit #
    # mres
    mresl = decay.fit_mres_bs(psql,decay_params,decay_params['decay_ward_fit']['ml'],gvmpl,gvppl)
    mress = decay.fit_mres_bs(psql,decay_params,decay_params['decay_ward_fit']['ms'],gvmps,gvpps)
    # meson
    pionfit = decay.fit_decay_bs(psql,decay_params,'pion',gvss_pion,gvps_pion)
    kaonfit = decay.fit_decay_bs(psql,decay_params,'kaon',gvss_kaon,gvps_kaon)
    # make 5D decay constant
    fkaon = decay.decay_constant(decay_params, kaonfit['meson_fit'].p['Z0_p'], kaonfit['meson_fit'].p['E0'], mresl['mres_fit'].p['mres'], mress['mres_fit'].p['mres'])
    fpion = decay.decay_constant(decay_params, pionfit['meson_fit'].p['Z0_p'], pionfit['meson_fit'].p['E0'], mresl['mres_fit'].p['mres'])
    mkaon = kaonfit['meson_fit'].p['E0']
    mpion = pionfit['meson_fit'].p['E0']
    # result
    #print mresl['mres_fit'].p
    #print mress['mres_fit'].p
    #print pionfit['meson_fit'].p
    #print kaonfit['meson_fit'].p
    #print gv.evalcorr([mresl['mres_fit'].p['mres'], mress['mres_fit'].p['mres'], pionfit['meson_fit'].p['E0'], kaonfit['meson_fit'].p['E0']])
    print fkaon/fpion
    #print gv.evalcorr([fkaon,fpion])
    # chipt parameters
    priors = gv.BufferDict()
    priors['mpion2fpion2'] = (mpion/fpion)**2
    priors['mkaon2fpion2'] = (mkaon/fpion)**2
    data = gv.BufferDict()
    data['fkfpi'] = fkaon/fpion
    # write output
    result = {ensemble:{'data':data, 'priors':priors}}
    print result
    pickle.dump(result, open('./pickle_result/fkfpi_%s.pickle' %ensemble, 'wb'))
    g = pickle.load(open('./pickle_result/fkfpi_%s.pickle' %ensemble, 'rb'))
    print g
