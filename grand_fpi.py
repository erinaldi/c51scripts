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
    # pion correlator
    ss_pion, ps_pion = decay.read_decay_bs(psql,decay_params,'pion')
    # index
    x_mpl = np.arange(len(mpl[0]))
    x_ppl = len(x_mpl)+np.arange(len(ppl[0]))
    x_ss_pion = len(x_ppl)+len(x_mpl)+np.arange(len(ss_pion[0]))
    x_ps_pion = len(x_ss_pion)+len(x_ppl)+len(x_mpl)+np.arange(len(ps_pion[0]))
    print "len mpl:", np.shape(mpl)
    print "len ppl:", np.shape(ppl)
    print "len ss_pion:", np.shape(ss_pion)
    print "len ps_pion:", np.shape(ps_pion)
    # concatenate
    boot0 = np.concatenate((mpl,ppl,ss_pion,ps_pion),axis=1)
    # correlated covariance
    gvboot0 = c51.make_gvars(boot0)
    # split for projects
    gvmpl = gvboot0[x_mpl]
    gvppl = gvboot0[x_ppl]
    gvss_pion = gvboot0[x_ss_pion]
    gvps_pion = gvboot0[x_ps_pion]
    # fit #
    # mres
    mresl = decay.fit_mres_bs(psql,decay_params,decay_params['decay_ward_fit']['ml'],gvmpl,gvppl)
    # meson
    pionfit = decay.fit_decay_bs(psql,decay_params,'pion',gvss_pion,gvps_pion)
    # make 5D decay constant
    fpion = decay.decay_constant(decay_params, pionfit['meson_fit'].p['Z0_p'], pionfit['meson_fit'].p['E0'], mresl['mres_fit'].p['mres'])
    mpion = pionfit['meson_fit'].p['E0']
    # result
    #print mresl['mres_fit'].p
    #print mress['mres_fit'].p
    #print pionfit['meson_fit'].p
    #print kaonfit['meson_fit'].p
    #print gv.evalcorr([mresl['mres_fit'].p['mres'], mress['mres_fit'].p['mres'], pionfit['meson_fit'].p['E0'], kaonfit['meson_fit'].p['E0']])
    print "fpi:", fpion
    #print gv.evalcorr([fkaon,fpion])
    # chipt parameters
    priors = gv.BufferDict()
    priors['mpion2fpion2'] = (mpion/fpion)**2
    data = gv.BufferDict()
    data['fpi'] = fpion
    # write output
    result = {ensemble:{'data':data, 'priors':priors}}
    print result
    pickle.dump(result, open('./pickle_result/fpi_%s.pickle' %ensemble, 'wb'))
    g = pickle.load(open('./pickle_result/fpi_%s.pickle' %ensemble, 'rb'))
    print g
