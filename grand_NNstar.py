#simultaneous N N* fit... sigh...
import sys
sys.path.append('$HOME/c51/scripts/')
import gvar as gv
import cPickle as pickle
import calsql as sql
import sqlc51lib as c51
import password_file as pwd
import numpy as np
import yaml
import sql_decay_ward as decay
import dMNNstar as gA

if __name__=="__main__":
    # read yaml #
    # master
    f = open('./grand_NNstar.yml' ,'r')
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
    f = open('./gANNstar.yml','r')
    gA_params = yaml.load(f)
    f.close()
    gA_params['gA_fit']['ens'] = params['grand_ensemble']['ens']
    gA_params['gA_fit']['ml'] = params['grand_ensemble']['ml']
    # log in sql #
    psqlpwd = pwd.passwd()
    psql = sql.pysql('cchang5','cchang5',psqlpwd)
    # read data #
    # pion correlator
    ss_pion, ps_pion = decay.read_decay_bs(psql,decay_params,'pion')
    # gA parameter file
    gAboot0 = gA.read_gA_bs(psql,gA_params)
    # nucleon correlator
    ss_N = gAboot0[:,:len(gAboot0[0])/4]
    ps_N = gAboot0[:,len(gAboot0[0])/4:len(gAboot0[0])/2]
    nbasak = len(gA_params['gA_fit']['basak'])
    cT = len(ss_N[0])
    ss_Nt = [np.roll(np.array(ss_N[:,i*cT/nbasak:(i+1)*cT/nbasak][:,::-1]), 1, axis=1) for i in range(nbasak)]
    ps_Nt = [np.roll(np.array(ps_N[:,i*cT/nbasak:(i+1)*cT/nbasak][:,::-1]), 1, axis=1) for i in range(nbasak)]
    ss_Ns = ss_Nt[0]
    ps_Ns = ps_Nt[0]
    for i in range(1,len(ss_Nt)):
        ss_Ns = np.concatenate((ss_Ns,ss_Nt[i]),axis=1)
        ps_Ns = np.concatenate((ps_Ns,ps_Nt[i]),axis=1)
    # gA correlator
    ss_gA = gAboot0[:,len(gAboot0[0])/2:3*len(gAboot0[0])/4]
    ps_gA = gAboot0[:,3*len(gAboot0[0])/4:]
    ss_gAt = [np.roll(np.array(ss_gA[:,i*cT/nbasak:(i+1)*cT/nbasak][:,::-1]), 1, axis=1) for i in range(nbasak)]
    ps_gAt = [np.roll(np.array(ps_gA[:,i*cT/nbasak:(i+1)*cT/nbasak][:,::-1]), 1, axis=1) for i in range(nbasak)]
    ss_gAs = ss_gAt[0]
    ps_gAs = ps_gAt[0]
    for i in range(1,len(ss_Nt)):
        ss_gAs = np.concatenate((ss_gAs,ss_gAt[i]),axis=1)
        ps_gAs = np.concatenate((ps_gAs,ps_gAt[i]),axis=1)
    # index
    # meson
    x_ss_pion = np.arange(len(ss_pion[0]))
    x_ps_pion = np.arange(len(ps_pion[0]))+len(x_ss_pion)
    # nucleon
    x_N = np.arange(len(ss_N[0])+len(ps_N[0])+len(ss_Ns[0])+len(ps_Ns[0]))+len(x_ss_pion)+len(x_ps_pion) 
    x_ss_N = np.arange(len(ss_N[0]))+len(x_ss_pion)+len(x_ps_pion)
    x_ps_N = np.arange(len(ps_N[0]))+len(x_ss_pion)+len(x_ps_pion)+len(x_ss_N)
    x_ss_Ns = np.arange(len(ss_Ns[0]))+len(x_ss_pion)+len(x_ps_pion)+len(x_ss_N)+len(x_ps_N)
    x_ps_Ns = np.arange(len(ps_Ns[0]))+len(x_ss_pion)+len(x_ps_pion)+len(x_ss_N)+len(x_ps_N)+len(x_ss_Ns)
    # concatenate data
    data = c51.make_gvars(np.concatenate((ss_pion, ps_pion, ss_N, ps_N, ss_Ns, ps_Ns), axis=1))
    # separate data
    gvss_pion = data[x_ss_pion]
    gvps_pion = data[x_ps_pion]
    gv_N = data[x_N]
    # meson
    pifit = decay.fit_decay_bs(psql,decay_params,'pion',gvss_pion,gvps_pion)
    print pifit['meson_fit']
    prior = {'Epi': pifit['meson_fit'].p['E0']}
    # crazy proton fit sighz
    print prior
