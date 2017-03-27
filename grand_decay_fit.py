# Bayes the decay constants with all PQ points
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

if __name__=="__main__":
    # read yaml #
    # master
    f = open('./grand_decay_fit.yml' ,'r')
    params = yaml.load(f)
    f.close()
    ens = params['grand_ensemble'].keys()[0]
    stream = params['grand_ensemble'][ens]['stream']
    ml = params['grand_ensemble'][ens]['ml']
    ms = params['grand_ensemble'][ens]['ms']
    # decay parameter file
    user_flag = c51.user_list()
    f = open('./decay.yml','r')
    decay_params = yaml.load(f)
    f.close()
    decay_params['decay_ward_fit']['ens'] =  {"tag": ens, "stream": stream}
    # log in sql #
    psqlpwd = pwd.passwd()
    psql = sql.pysql('cchang5','cchang5',psqlpwd)
    ### read data ###
    x_list = dict()
    data = []
    # mres
    for m in ml:
        mp, pp = decay.read_mres_bs(psql,decay_params,m)
        if len(data) != 0:
            x_list['mp_%s' %str(m)] = len(data[0]) + np.arange(len(mp[0]))
            data = np.concatenate((data,mp), axis=1)
        else:
            x_list['mp_%s' %str(m)] = np.arange(len(mp[0]))
            data = mp
        x_list['pp_%s' %str(m)] = len(data[0]) + np.arange(len(pp[0]))
        data = np.concatenate((data,pp), axis=1)
    for s in ms:
        mp, pp = decay.read_mres_bs(psql,decay_params,s)
        x_list['mp_%s' %str(s)] = len(data[0]) + np.arange(len(mp[0]))
        data = np.concatenate((data,mp), axis=1)
        x_list['pp_%s' %str(s)] = len(data[0]) + np.arange(len(pp[0]))
        data = np.concatenate((data,pp), axis=1)
    # mesons
    for m in ml:
        decay_params['decay_ward_fit']['ml'] = m
        ss, ps = decay.read_decay_bs(psql,decay_params,'pion')
        x_list['ss_%s_%s' %(str(m),str(m))] = len(data[0]) + np.arange(len(ss[0]))
        data = np.concatenate((data,ss), axis=1)
        x_list['ps_%s_%s' %(str(m),str(m))] = len(data[0]) + np.arange(len(ps[0]))
        data = np.concatenate((data,ps), axis=1)
        for s in ms:
            decay_params['decay_ward_fit']['ms'] = s
            ss, ps = decay.read_decay_bs(psql,decay_params,'kaon')
            x_list['ss_%s_%s' %(str(m),str(s))] = len(data[0]) + np.arange(len(ss[0]))
            data = np.concatenate((data,ss), axis=1)
            x_list['ps_%s_%s' %(str(m),str(s))] = len(data[0]) + np.arange(len(ps[0]))
            data = np.concatenate((data,ps), axis=1)
    # make gvars
    data_gv = c51.make_gvars(data)
    ### fit ###
    result = dict()
    # mres
    for m in ml:
        gv_mp = data_gv[x_list['mp_%s' %str(m)]]
        gv_pp = data_gv[x_list['pp_%s' %str(m)]]
        fit = decay.fit_mres_bs(psql,decay_params,m,gv_mp,gv_pp)['mres_fit']
        result['mres_%s' %str(m)] = fit
    for s in ms:
        gv_mp = data_gv[x_list['mp_%s' %str(s)]]
        gv_pp = data_gv[x_list['pp_%s' %str(s)]]
        fit = decay.fit_mres_bs(psql,decay_params,s,gv_mp,gv_pp)['mres_fit']
        result['mres_%s' %str(s)] = fit
    # mesons
    for m in ml:
        decay_params['decay_ward_fit']['ml'] = m
        gv_SS = data_gv[x_list['ss_%s_%s' %(str(m),str(m))]]
        gv_PS = data_gv[x_list['ps_%s_%s' %(str(m),str(m))]]
        fit = decay.fit_decay_bs(psql,decay_params,'pion', gv_SS, gv_PS)['meson_fit']
        result['meson_%s_%s' %(str(m),str(m))] = fit
        for s in ms:
            decay_params['decay_ward_fit']['ms'] = s
            gv_SS = data_gv[x_list['ss_%s_%s' %(str(m),str(s))]]
            gv_PS = data_gv[x_list['ps_%s_%s' %(str(m),str(s))]]
            fit = decay.fit_decay_bs(psql,decay_params,'kaon', gv_SS, gv_PS)['meson_fit']
            result['meson_%s_%s' %(str(m),str(s))] = fit
    # make decay constants
    decay = dict()
    for m in ml:
        Z0_p = result['meson_%s_%s' %(str(m),str(m))].p['Z0_p']
        E0 = result['meson_%s_%s' %(str(m),str(m))].p['E0']
        mres = result['mres_%s' %str(m)].p['mres']
        decay['%s_%s' %(str(m),str(m))] = Z0_p*np.sqrt(2.)*(2.*float(m)+2.*mres)/E0**(3./2.)
        for s in ms:
            Z0_p = result['meson_%s_%s' %(str(m),str(s))].p['Z0_p']
            E0 = result['meson_%s_%s' %(str(m),str(s))].p['E0']
            mres_s = result['mres_%s' %str(s)].p['mres']
            decay['%s_%s' %(str(m),str(s))] = Z0_p*np.sqrt(2.)*(float(m)+float(s)+mres+mres_s)/E0**(3./2.)
            print m, s, decay['%s_%s' %(str(m),str(s))]/decay['%s_%s' %(str(m),str(m))]
