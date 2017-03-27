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
import sql_hisq_meson as hisq
import sql_mixed_meson as mix

def r(mixed, dw, hisq):
    num = mixed**2 - 0.5*(dw**2+hisq**2)
    den = 0.5*(dw**2+hisq**2)
    return num/den, num

if __name__=="__main__":
    # read yaml #
    # master
    f = open('./grand_mix_fit.yml' ,'r')
    params = yaml.load(f)
    f.close()
    # decay parameter file
    f = open('./decay.yml','r')
    decay_params = yaml.load(f)
    f.close()
    decay_params['decay_ward_fit']['ens'] = params['grand_ensemble']['ens']
    decay_params['decay_ward_fit']['ml'] = params['grand_ensemble']['ml']
    decay_params['decay_ward_fit']['ms'] = params['grand_ensemble']['ms']
    # hisq parameter file
    f = open('./hisq.yml','r')
    hisq_params = yaml.load(f)
    f.close()
    hisq_params['hisq_fit']['ens'] = params['grand_ensemble']['ens']
    hisq_params['hisq_fit']['ml'] = str(float("0.%s" %params['grand_ensemble']['ens']['tag'].split('m')[1]))
    hisq_params['hisq_fit']['ms'] = str(float("0.%s" %params['grand_ensemble']['ens']['tag'].split('m')[2]))
    # mixed parameter file
    f = open('./mixed.yml','r')
    mixed_params = yaml.load(f)
    f.close()
    mixed_params['mixed_fit']['ens'] = params['grand_ensemble']['ens']
    mixed_params['mixed_fit']['ml'] = params['grand_ensemble']['ml']
    mixed_params['mixed_fit']['ms'] = params['grand_ensemble']['ms']
    # log in sql #
    psqlpwd = pwd.passwd()
    psql = sql.pysql('cchang5','cchang5',psqlpwd)
    # read data #
    # domain wall mesons
    # pion correlator
    ss_pion, ps_pion = decay.read_decay_bs(psql,decay_params,'pion')
    ss_kaon, ps_kaon = decay.read_decay_bs(psql,decay_params,'kaon')
    ss_etas, ps_etas = decay.read_decay_bs(psql,decay_params,'etas')
    # hisq correlator
    phi_jj_5 = hisq.read_hisq_meson(psql,hisq_params,'phi_jj_5')
    phi_jr_5 = hisq.read_hisq_meson(psql,hisq_params,'phi_jr_5')
    phi_rr_5 = hisq.read_hisq_meson(psql,hisq_params,'phi_rr_5')
    # mixed correlator
    phi_ju = mix.read_mixed_meson(psql,mixed_params,'phi_ju')
    phi_js = mix.read_mixed_meson(psql,mixed_params,'phi_js')
    phi_ru = mix.read_mixed_meson(psql,mixed_params,'phi_ru')
    phi_rs = mix.read_mixed_meson(psql,mixed_params,'phi_rs')
    # concatenate and build grand covariance matrix #
    # index
    x_ss_pion = np.arange(len(ss_pion[0]))
    x_ps_pion = np.arange(len(ps_pion[0]))+len(x_ss_pion)
    x_ss_kaon = np.arange(len(ss_kaon[0]))+len(x_ss_pion)+len(x_ps_pion)
    x_ps_kaon = np.arange(len(ps_kaon[0]))+len(x_ss_pion)+len(x_ps_pion)+len(x_ss_kaon)
    x_ss_etas = np.arange(len(ss_etas[0]))+len(x_ss_pion)+len(x_ps_pion)+len(x_ss_kaon)+len(x_ps_kaon)
    x_ps_etas = np.arange(len(ps_etas[0]))+len(x_ss_pion)+len(x_ps_pion)+len(x_ss_kaon)+len(x_ps_kaon)+len(x_ss_etas)
    x_phi_jj_5 = np.arange(len(phi_jj_5[0]))+len(x_ss_pion)+len(x_ps_pion)+len(x_ss_kaon)+len(x_ps_kaon)+len(x_ss_etas)+len(x_ps_etas)
    x_phi_jr_5 = np.arange(len(phi_jr_5[0]))+len(x_ss_pion)+len(x_ps_pion)+len(x_ss_kaon)+len(x_ps_kaon)+len(x_ss_etas)+len(x_ps_etas)+len(x_phi_jj_5)
    x_phi_rr_5 = np.arange(len(phi_rr_5[0]))+len(x_ss_pion)+len(x_ps_pion)+len(x_ss_kaon)+len(x_ps_kaon)+len(x_ss_etas)+len(x_ps_etas)+len(x_phi_jj_5)+len(x_phi_jr_5)
    x_phi_ju = np.arange(len(phi_ju[0]))+len(x_ss_pion)+len(x_ps_pion)+len(x_ss_kaon)+len(x_ps_kaon)+len(x_ss_etas)+len(x_ps_etas)+len(x_phi_jj_5)+len(x_phi_jr_5)+len(x_phi_rr_5)
    x_phi_js = np.arange(len(phi_js[0]))+len(x_ss_pion)+len(x_ps_pion)+len(x_ss_kaon)+len(x_ps_kaon)+len(x_ss_etas)+len(x_ps_etas)+len(x_phi_jj_5)+len(x_phi_jr_5)+len(x_phi_rr_5)+len(x_phi_ju)
    x_phi_ru = np.arange(len(phi_ru[0]))+len(x_ss_pion)+len(x_ps_pion)+len(x_ss_kaon)+len(x_ps_kaon)+len(x_ss_etas)+len(x_ps_etas)+len(x_phi_jj_5)+len(x_phi_jr_5)+len(x_phi_rr_5)+len(x_phi_ju)+len(x_phi_js)
    x_phi_rs = np.arange(len(phi_rs[0]))+len(x_ss_pion)+len(x_ps_pion)+len(x_ss_kaon)+len(x_ps_kaon)+len(x_ss_etas)+len(x_ps_etas)+len(x_phi_jj_5)+len(x_phi_jr_5)+len(x_phi_rr_5)+len(x_phi_ju)+len(x_phi_js)+len(x_phi_ru)
    print "len ss_pion :", np.shape(ss_pion)
    print "len ps_pion :", np.shape(ps_pion)
    print "len ss_kaon :", np.shape(ss_kaon)
    print "len ps_kaon :", np.shape(ps_kaon)
    print "len ss_etas :", np.shape(ss_etas)
    print "len ps_etas :", np.shape(ps_etas)
    print "len phi_jj_5:", np.shape(phi_jj_5)
    print "len phi_jr_5:", np.shape(phi_jr_5)
    print "len phi_rr_5:", np.shape(phi_rr_5)
    print "len phi_ju  :", np.shape(phi_ju)
    print "len phi_js  :", np.shape(phi_js)
    print "len phi_ru  :", np.shape(phi_ru)
    print "len phi_rs  :", np.shape(phi_rs) 
    # concatenate
    boot0 = np.concatenate((ss_pion,ps_pion,ss_kaon,ps_kaon,ss_etas,ps_etas,phi_jj_5,phi_jr_5,phi_rr_5,phi_ju,phi_js,phi_ru,phi_rs),axis=1)
    if len(boot0) > 200:
        boot0 = boot0[:200,:]
    # correlated covariance
    gvboot0 = c51.make_gvars(boot0)
    # split to different project subsets
    gvss_pion = gvboot0[x_ss_pion]
    gvps_pion = gvboot0[x_ps_pion]
    gvss_kaon = gvboot0[x_ss_kaon]
    gvps_kaon = gvboot0[x_ps_kaon]
    gvss_etas = gvboot0[x_ss_etas]
    gvps_etas = gvboot0[x_ps_etas]
    gvphi_jj_5 = gvboot0[x_phi_jj_5]
    gvphi_jr_5 = gvboot0[x_phi_jr_5]
    gvphi_rr_5 = gvboot0[x_phi_rr_5]
    gvphi_ju = gvboot0[x_phi_ju]
    gvphi_js = gvboot0[x_phi_js]
    gvphi_ru = gvboot0[x_phi_ru]
    gvphi_rs = gvboot0[x_phi_rs]
    # fit #
    # meson
    pifit = decay.fit_decay_bs(psql,decay_params,'pion',gvss_pion,gvps_pion)
    kafit = decay.fit_decay_bs(psql,decay_params,'kaon',gvss_kaon,gvps_kaon)
    etfit = decay.fit_decay_bs(psql,decay_params,'etas',gvss_etas,gvps_etas)
    # hisq
    jjfit = hisq.fit_hisq_meson(psql,hisq_params,'phi_jj_5',gvphi_jj_5)
    jrfit = hisq.fit_hisq_meson(psql,hisq_params,'phi_jr_5',gvphi_jr_5)
    rrfit = hisq.fit_hisq_meson(psql,hisq_params,'phi_rr_5',gvphi_rr_5)
    # mixed
    jufit = mix.fit_mixed_meson(psql,mixed_params,'phi_ju',gvphi_ju)
    jsfit = mix.fit_mixed_meson(psql,mixed_params,'phi_js',gvphi_js)
    rufit = mix.fit_mixed_meson(psql,mixed_params,'phi_ru',gvphi_ru)
    rsfit = mix.fit_mixed_meson(psql,mixed_params,'phi_rs',gvphi_rs)
    # collect output
    result = gv.BufferDict()
    result['mpi'] = pifit['meson_fit'].p['E0']
    result['mka'] = kafit['meson_fit'].p['E0']
    result['met'] = etfit['meson_fit'].p['E0']
    result['mjj'] = jjfit['hisq_meson_fit'].p['E0']
    result['mjr'] = jrfit['hisq_meson_fit'].p['E0']
    result['mrr'] = rrfit['hisq_meson_fit'].p['E0']
    result['mju'] = jufit['mixed_meson_fit'].p['E0']
    result['mjs'] = jsfit['mixed_meson_fit'].p['E0']
    result['mru'] = rufit['mixed_meson_fit'].p['E0']
    result['mrs'] = rsfit['mixed_meson_fit'].p['E0']
    # ratios
    result['ju/mpi'], result['ju'] = r(result['mju'], result['mpi'], result['mjj'])
    result['js/mka'], result['js'] = r(result['mjs'], result['mka'], result['mjr'])
    result['ru/mka'], result['ru'] = r(result['mru'], result['mka'], result['mjr'])
    result['rs/met'], result['rs'] = r(result['mrs'], result['met'], result['mrr'])
    # w0a
    w0a15 = 1.1358515573227301
    w0a12 = 1.4212271973466004
    w0a09 = 1.9588571428571429
    print result
    # write output
    pickle.dump(result, open('./pickle_result/mix_%s.pickle' %(params['grand_ensemble']['ens']['tag']), 'wb'))
    g = pickle.load(open('./pickle_result/mix_%s.pickle' %(params['grand_ensemble']['ens']['tag']), 'rb'))
    print g
