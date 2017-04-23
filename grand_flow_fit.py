# Fit all flow parameters
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
import sql_decay_axial as axial
import fh2 as gA

if __name__=="__main__":
    # read yaml #
    # master
    f = open('./grand_flow_fit.yml' ,'r')
    params = yaml.load(f)
    f.close()
    # decay parameter file
    f = open('./decay_flow.yml','r')
    decay_params = yaml.load(f)
    f.close()
    decay_params['decay_ward_fit']['ens'] = params['grand_ensemble']['ens']
    decay_params['decay_ward_fit']['ml'] = params['grand_ensemble']['ml']
    decay_params['decay_ward_fit']['ms'] = params['grand_ensemble']['ms']
    # axial parameter file
    f = open('./axial_flow.yml','r')
    axial_params = yaml.load(f)
    f.close()
    axial_params['axial_fit']['ens'] = params['grand_ensemble']['ens']
    axial_params['axial_fit']['ml'] = params['grand_ensemble']['ml']
    axial_params['axial_fit']['ms'] = params['grand_ensemble']['ms']
    # gA parameter file
    f = open('./fh.yml','r')
    gA_params = yaml.load(f)
    f.close()
    gA_params['gA_fit']['ens'] = params['grand_ensemble']['ens']
    gA_params['gA_fit']['ml'] = params['grand_ensemble']['ml']
    # log in sql #
    psqlpwd = pwd.passwd()
    psql = sql.pysql('cchang5','cchang5',psqlpwd)
    # read data #
    # mres for ml
    lmp, lpp = decay.read_mres_bs(psql,decay_params,decay_params['decay_ward_fit']['ml'])
    smp, spp = decay.read_mres_bs(psql,decay_params,decay_params['decay_ward_fit']['ms'])
    # pion correlator
    ss_pion, ps_pion = decay.read_decay_bs(psql,decay_params,'pion')
    ss_kaon, ps_kaon = decay.read_decay_bs(psql,decay_params,'kaon')
    # axial correlator
    ll = axial.read_axial(psql, axial_params, 'axial_ll')
    ls = axial.read_axial(psql, axial_params, 'axial_ls')
    # gA parameter file
    gAboot0 = gA.read_gA_bs(psql,gA_params,twopt=True)
    # concatenate and build grand covariance matrix #
    # index
    x_lmp = np.arange(len(lmp[0]))
    x_lpp = np.arange(len(lpp[0]))+len(x_lmp)
    x_smp = np.arange(len(smp[0]))+len(x_lmp)+len(x_lpp)
    x_spp = np.arange(len(spp[0]))+len(x_lmp)+len(x_lpp)+len(x_smp)
    x_ss_pion = np.arange(len(ss_pion[0]))+len(x_lmp)+len(x_lpp)+len(x_smp)+len(x_spp)
    x_ps_pion = np.arange(len(ps_pion[0]))+len(x_lmp)+len(x_lpp)+len(x_smp)+len(x_spp)+len(x_ss_pion)
    x_ss_kaon = np.arange(len(ss_kaon[0]))+len(x_lmp)+len(x_lpp)+len(x_smp)+len(x_spp)+len(x_ss_pion)+len(x_ps_pion)
    x_ps_kaon = np.arange(len(ps_kaon[0]))+len(x_lmp)+len(x_lpp)+len(x_smp)+len(x_spp)+len(x_ss_pion)+len(x_ps_pion)+len(x_ss_kaon)
    x_ll = np.arange(len(ll[0]))+len(x_lmp)+len(x_lpp)+len(x_smp)+len(x_spp)+len(x_ss_pion)+len(x_ps_pion)+len(x_ss_kaon)+len(x_ps_kaon)
    x_ls = np.arange(len(ls[0]))+len(x_lmp)+len(x_lpp)+len(x_smp)+len(x_spp)+len(x_ss_pion)+len(x_ps_pion)+len(x_ss_kaon)+len(x_ps_kaon)+len(x_ll)
    x_gAboot0 = np.arange(len(gAboot0[0]))+len(x_lmp)+len(x_lpp)+len(x_smp)+len(x_spp)+len(x_ss_pion)+len(x_ps_pion)+len(x_ss_kaon)+len(x_ps_kaon)+len(x_ll)+len(x_ls)
    print "len lmp:", np.shape(lmp)
    print "len lpp:", np.shape(lpp)
    print "len smp:", np.shape(smp)
    print "len spp:", np.shape(spp)
    print "len ss_pion:", np.shape(ss_pion)
    print "len ps_pion:", np.shape(ps_pion)
    print "len ss_kaon:", np.shape(ss_kaon)
    print "len ps_kaon:", np.shape(ps_kaon)
    print "len ll:", np.shape(ll)
    print "len ls:", np.shape(ls)
    print "len gAboot0:", np.shape(gAboot0)
    # concatenate
    boot0 = np.concatenate((lmp,lpp,smp,spp,ss_pion,ps_pion,ss_kaon,ps_kaon,ll,ls,gAboot0),axis=1)
    #if len(boot0) > 200:
    #    boot0 = boot0[0::5,:]
    # correlated covariance
    gvboot0 = c51.make_gvars(boot0)
    # split to different project subsets
    gvlmp = gvboot0[x_lmp]
    gvlpp = gvboot0[x_lpp]
    gvsmp = gvboot0[x_smp]
    gvspp = gvboot0[x_spp]
    gvss_pion = gvboot0[x_ss_pion]
    gvps_pion = gvboot0[x_ps_pion]
    gvss_kaon = gvboot0[x_ss_kaon]
    gvps_kaon = gvboot0[x_ps_kaon]
    gvll = gvboot0[x_ll]
    gvls = gvboot0[x_ls]
    gvgAboot0 = gvboot0[x_gAboot0]
    # fit #
    # mres
    mresl = decay.fit_mres_bs(psql,decay_params,decay_params['decay_ward_fit']['ml'],gvlmp,gvlpp)
    mress = decay.fit_mres_bs(psql,decay_params,decay_params['decay_ward_fit']['ml'],gvsmp,gvspp)
    # meson
    #pifit = decay.fit_decay_bs(psql,decay_params,'pion',gvss_pion,gvps_pion)
    #kafit = decay.fit_decay_bs(psql,decay_params,'kaon',gvss_kaon,gvps_kaon)
    # axial
    aldata = np.concatenate((gvss_pion, gvps_pion, gvll))
    alfit = axial.fit_axial(psql,axial_params,decay_params,'axial_ll',aldata,flow=True) #,pifit['meson_fit'].p)
    asdata = np.concatenate((gvss_kaon, gvps_kaon, gvls))
    asfit = axial.fit_axial(psql,axial_params,decay_params,'axial_ls',asdata,flow=True) #,kafit['meson_fit'].p)
    # mN and gA
    mNfit = gA.fit_proton(psql,gA_params,gvgAboot0,twopt=True)
    # chipt parameters
    result = gv.BufferDict()
    result['mpi'] = alfit['axial_fit'].p['E0']
    result['mka'] = asfit['axial_fit'].p['E0']
    result['fpi'] = decay.decay_constant(decay_params,alfit['axial_fit'].p['Z0_p'],alfit['axial_fit'].p['E0'],mresl['mres_fit'].p['mres'])/np.sqrt(2)
    result['fka'] = decay.decay_constant(decay_params,asfit['axial_fit'].p['Z0_p'],asfit['axial_fit'].p['E0'],mresl['mres_fit'].p['mres'],mress['mres_fit'].p['mres'])/np.sqrt(2)
    result['fk/fpi'] = result['fka']/result['fpi']
    result['fpi4D'] = axial.fphi(alfit['axial_fit'].p['F0'], alfit['axial_fit'].p['E0'])/np.sqrt(2)
    result['fka4D'] = axial.fphi(asfit['axial_fit'].p['F0'], asfit['axial_fit'].p['E0'])/np.sqrt(2)
    result['ZAll'] = result['fpi']/result['fpi4D']
    result['ZAls'] = result['fka']/result['fka4D']
    #result['mN'] = mNfit['nucleon_fit'].p['E0']
    #result['mN/fpi'] = result['mN']/result['fpi']
    print "ZAll  :", result['ZAll']
    print "ZAls  :", result['ZAls']
    print "fpi   :", result['fpi']
    print "fka   :", result['fka']
    #print "mN    :", result['mN']
    print "fk/fpi:", result['fk/fpi']
    #print "mN/fpi:", result['mN/fpi']
    # write output
    #pickle.dump(result, open('./pickle_result/flow%s_%s.pickle' %(params['grand_ensemble']['flow'], params['grand_ensemble']['ens']['tag']), 'wb'))
    #g = pickle.load(open('./pickle_result/flow%s_%s.pickle' %((params['grand_ensemble']['flow'],params['grand_ensemble']['ens']['tag']), 'rb'))
    #print g
