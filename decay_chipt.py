import numpy as np
import calsql as sql
import sqlc51lib as c51
import password_file as pwd
import yaml
from collections import OrderedDict
import gvar as gv
import lsqfit

def ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    class OrderedLoader(Loader):
        pass
    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)

def decay_ratio(mpi, mk, fpi, fk):
    r = fk/fpi
    lpi = 5./4. * 1./(16.*np.pi**2) * (mpi/fpi)**2 * np.log((mpi/fpi)**2)
    lk = -1./2. * 1./(16.*np.pi**2) * (mk/fpi)**2 * np.log((mk/fpi)**2)
    leta = -1./(16.*np.pi**2) * ((mk/fpi)**2 - 1./4. * (mpi/fpi)**2) * np.log(4./3. * (mk/fpi)**2 - 1./3. * (mpi/fpi)**2)
    return fpi**2*(r - 1.0 - lpi - lk - leta)

def L5_ratio(mpi, mk, fpi, fk):
    r = fk/fpi
    lpi = 5./4. * 1./(16.*np.pi**2) * (mpi/fpi)**2 * np.log((mpi/fpi)**2)
    lk = -1./2. * 1./(16.*np.pi**2) * (mk/fpi)**2 * np.log((mk/fpi)**2)
    leta = -1./(16.*np.pi**2) * ((mk/fpi)**2 - 1./4. * (mpi/fpi)**2) * np.log(4./3. * (mk/fpi)**2 - 1./3. * (mpi/fpi)**2)
    D = fpi**2 / (8.*(mk**2 - mpi**2))
    return D*(r - 1.0 - lpi - lk - leta)

def fit_fcn(x, p):
    ens = x[0]
    ml = x[1]
    ms = x[2]
    alpha = x[3]
    y = []
    for i in range(len(ens)):
        fit = 8.0 * (p['E0_%s_%s' %(str(ml[i]),str(ms[i]))]**2 - p['E0_%s_%s' %(str(ml[i]),str(ml[i]))]**2) * p['L5']
        # NLO lattice spacing
        fit += p['cA'] * p['a_%s' %str(ens[i])]**2
        fit += p['cAa'] * alpha[i] * p['a_%s' %str(ens[i])]**2
        # NNLO
        #fit += p['E0_%s_%s' %(str(ml[i]),str(ml[i]))]**2 * (2.*p['E0_%s_%s' %(str(ml[i]),str(ms[i]))]**2 - p['E0_%s_%s' %(str(ml[i]),str(ml[i]))]**2) * p['d0']
        #fit += p['E0_%s_%s' %(str(ml[i]),str(ms[i]))]**2 * (2.*p['E0_%s_%s' %(str(ml[i]),str(ms[i]))]**2 - p['E0_%s_%s' %(str(ml[i]),str(ml[i]))]**2) * p['d1']
        #fit += (2.*p['E0_%s_%s' %(str(ml[i]),str(ms[i]))]**2 - p['E0_%s_%s' %(str(ml[i]),str(ml[i]))]**2)**2 * p['d2']
        #fit += p['a_%s' %str(ens[i])]**2 * (2.*p['E0_%s_%s' %(str(ml[i]),str(ms[i]))]**2 - p['E0_%s_%s' %(str(ml[i]),str(ml[i]))]**2) * p['d3']
        #fit += p['dA'] * p['a_%s' %str(ens[i])]**4 # saturated
        y.append(fit)
    return np.array(y)

if __name__=='__main__':
    # read master
    f = open('./decay_chipt.yml', 'r')
    dataset = ordered_load(f,yaml.SafeLoader)['decay_chipt']['dataset']
    f.close()
    # read correlator master
    f = open('./decay.yml', 'r')
    cmeta = ordered_load(f,yaml.SafeLoader)
    f.close()
    # log in sql
    psqlpwd = pwd.passwd()
    psql = sql.pysql('cchang5','cchang5',psqlpwd)
    # read data
    ens = dataset.keys()
    # read alphas
    alpha = dict()
    alpha['l1648f211b580m013m065m838'] =  0.58801
    alpha['l2448f211b580m0064m0640m828'] = 0.58801
    alpha['l3248f211b580m00235m0647m831'] = 0.58801
    alpha['l2464f211b600m0102m0509m635'] = 0.53796
    alpha['l4064f211b600m00507m0507m628'] = 0.53796
    alpha['l4864f211b600m00184m0507m628'] = 0.53796
    alpha['l3296f211b630m0074m037m440'] = 0.43356
    alpha['l4896f211b630m00363m0363m430'] = 0.43356
    # construct fk, fpi, mpi/fpi, mk/fpi
    prior = dict()
    ens_list = []
    ml_list = []
    ms_list = []
    alpha_list = []
    fk_list = []
    fk_bs_list = []
    fpi_list = []
    fpi_bs_list = []
    pion_list = []
    pion_bs_list = []
    kaon_list = []
    kaon_bs_list = []
    for e in ens:
        print e
        #print '%s_mpi2' %e, ',', '+-', ',', '%s_mkmpi2' %e, ',', '+-', ',', '%s_L5' %e, ',', '+-'
        # read a_fm
        sql_cmd = "SELECT a_fm FROM callat_corr.hisq_ensembles WHERE tag='%s';" %(e)
        psql.cur.execute(sql_cmd)
        a_fm = psql.cur.fetchone()[0]
        a_gv = gv.gvar(float(a_fm[:-4]), 10**(-1.0*int(len(a_fm[:-4].split('.')[1])))*float(a_fm[-3:-1]))
        # read mres
        # mres meta
        mmeta = cmeta[e]['mres']
        # pion meta
        pmeta = cmeta[e]['pion']
        # kaon meta
        kmeta = cmeta[e]['kaon']
        # write output
        prior['a_%s' %(e)] = a_gv
        for ml in dataset[e]['ml']:
            # get ml mres
            mmetal = mmeta[ml]
            sql_cmd = "SELECT id, result->>'mres' FROM callat_proj.jmu WHERE corr1_id=%s AND corr2_id=%s AND tmin=%s AND tmax=%s;" %(mmetal['meta_id']['mp'], mmetal['meta_id']['pp'], mmetal['trange']['tmin'][0], mmetal['trange']['tmax'][0])
            psql.cur.execute(sql_cmd)
            temp = psql.cur.fetchall()
            mresl_id = temp[0][0]
            mresl = float(temp[0][1])
            # get ml mres bootstrap
            sql_cmd = "SELECT result->>'mres' FROM callat_proj.jmu_bs WHERE jmu_id=%s ORDER BY bs_id;" %str(mresl_id)
            psql.cur.execute(sql_cmd)
            temp = psql.cur.fetchall()
            mresl_bs = np.array([float(i) for i in np.array(temp)[:,0]])
            # get pion
            pmetal = pmeta['%s_%s' %(ml,ml)]
            sql_cmd = "SELECT id, result->>'E0', result->>'Z0_p', result->>'dE0' FROM callat_proj.meson WHERE corr1_id=%s AND corr2_id=%s AND tmin=%s AND tmax=%s;" %(pmetal['meta_id']['SS'], pmetal['meta_id']['PS'], pmetal['trange']['tmin'][0], pmetal['trange']['tmax'][0])
            psql.cur.execute(sql_cmd)
            temp = psql.cur.fetchall()
            print temp
            pion_id = temp[0][0]
            pion_E0 = float(temp[0][1])
            pion_Z0p = float(temp[0][2])
            pion_dE0 = float(temp[0][3])
            # get pion bs
            sql_cmd = "SELECT result->>'E0', result->>'Z0_p' FROM callat_proj.meson_bs WHERE meson_id=%s;" %(pion_id)
            psql.cur.execute(sql_cmd)
            temp = psql.cur.fetchall()
            pion_E0_bs = np.array([float(i) for i in np.array(temp)[:,0]])
            pion_Z0p_bs = np.array([float(i) for i in np.array(temp)[:,1]])
            # make fpi
            fpi = pion_Z0p*np.sqrt(2.)*(2.*float(ml)+2.*mresl)/pion_E0**(3./2.)
            fpi_bs = pion_Z0p_bs*np.sqrt(2.)*(2.*float(ml)+2.*mresl_bs)/pion_E0_bs**(3./2.)
            # make prior
            prior['E0_%s_%s' %(ml, ml)] = gv.gvar(pion_E0, pion_dE0)
            #prior['f_%s' %(ml)] = gv.gvar(fpi, np.std(fpi_bs))
            for ms in dataset[e]['ms']:
                mmetas = mmeta[ms]
                sql_cmd = "SELECT id, result->>'mres' FROM callat_proj.jmu WHERE corr1_id=%s AND corr2_id=%s AND tmin=%s AND tmax=%s;" %(mmetas['meta_id']['mp'], mmetas['meta_id']['pp'], mmetas['trange']['tmin'][0], mmetas['trange']['tmax'][0])
                psql.cur.execute(sql_cmd)
                temp = psql.cur.fetchall()
                mress_id = temp[0][0]
                mress = float(temp[0][1])
                # get mres bootstrap
                sql_cmd = "SELECT result->>'mres' FROM callat_proj.jmu_bs WHERE jmu_id=%s ORDER BY bs_id;" %str(mresl_id)
                psql.cur.execute(sql_cmd)
                temp = psql.cur.fetchall()
                mress_bs = np.array([float(i) for i in np.array(temp)[:,0]])
                # get kaon
                kmetal = kmeta['%s_%s' %(ml,ms)]
                sql_cmd = "SELECT id, result->>'E0', result->>'Z0_p', result->>'dE0' FROM callat_proj.meson WHERE corr1_id=%s AND corr2_id=%s AND tmin=%s AND tmax=%s;" %(kmetal['meta_id']['SS'], kmetal['meta_id']['PS'], kmetal['trange']['tmin'][0], kmetal['trange']['tmax'][0])
                psql.cur.execute(sql_cmd)
                temp = psql.cur.fetchall()
                kaon_id = temp[0][0]
                kaon_E0 = float(temp[0][1])
                kaon_Z0p = float(temp[0][2])
                kaon_dE0 = float(temp[0][3])
                # get pion bs
                sql_cmd = "SELECT result->>'E0', result->>'Z0_p' FROM callat_proj.meson_bs WHERE meson_id=%s;" %(kaon_id)
                psql.cur.execute(sql_cmd)
                temp = psql.cur.fetchall()
                kaon_E0_bs = np.array([float(i) for i in np.array(temp)[:,0]])
                kaon_Z0p_bs = np.array([float(i) for i in np.array(temp)[:,1]])
                # make fpi
                fk = kaon_Z0p*np.sqrt(2.)*(float(ml)+float(ms)+mresl+mress)/kaon_E0**(3./2.) 
                fk_bs = kaon_Z0p_bs*np.sqrt(2.)*(float(ml)+float(ms)+mresl_bs+mress_bs)/kaon_E0_bs**(3./2.)
                # write prior
                prior['E0_%s_%s' %(ml, ms)] = gv.gvar(kaon_E0, kaon_dE0)
                # write results
                ens_list.append(e)
                alpha_list.append(alpha[e])
                ml_list.append(ml)
                ms_list.append(ms)
                fpi_list.append(fpi)
                fpi_bs_list.append(fpi_bs)
                pion_list.append(pion_E0)
                pion_bs_list.append(pion_E0_bs)
                fk_list.append(fk)
                fk_bs_list.append(fk_bs)
                kaon_list.append(kaon_E0)
                kaon_bs_list.append(kaon_E0_bs)
                # write L5
                #mpi_gv = gv.gvar(pion_E0, np.std(pion_E0_bs))
                #mk_gv = gv.gvar(kaon_E0, np.std(kaon_E0_bs))
                #fpi_gv = gv.gvar(fpi, np.std(fpi_bs))
                #fk_gv = gv.gvar(fk, np.std(fk_bs))
                #L5_param = L5_ratio(mpi_gv, mk_gv, fpi_gv, fk_gv)
                #print (mpi_gv**2).mean, ',', (mpi_gv**2).sdev, ',', (2.*mk_gv**2-mpi_gv**2).mean, ',', (2.*mk_gv**2-mpi_gv**2).sdev, ',', L5_param.mean, ',', L5_param.sdev
    #print prior
    #print ens_list
    #print ml_list
    #print ms_list
    #print np.shape(pion_list), np.shape(pion_bs_list), np.shape(kaon_list), np.shape(kaon_bs_list)
    #print np.shape(fpi_list), np.shape(fpi_bs_list), np.shape(fk_list), np.shape(fk_bs_list)
    # create fit data http://arxiv.org/pdf/hep-lat/0606023v1.pdf Eq. 8
    # set scale as fpi_latt to skip scale setting
    # move everything to left except meson masses
    # 8 (mk^2/fpi^2 -mpi^2/fpi^2) L5 + ac a^2 = [fk/fpi -1 -5/4 1/16pi^2 mpi^2/fpi^2 log(mpi^2/fpi^2) + 1/2 1/16pi^2 mk^2/fpi^2 log(mk^2/fpi^2) + 1/16pi^2(mk^2/fpi^2 -1/4 mpi^2/fpi^2) log(4/3 mpi^2/fpi^2 -1/3 mpi^2/fpi^2)]
    data = []
    data_bs = []
    for i in range(len(pion_list)):
        data.append(decay_ratio(pion_list[i], kaon_list[i], fpi_list[i], fk_list[i]))
        data_bs.append(decay_ratio(pion_bs_list[i], kaon_bs_list[i], fpi_bs_list[i], fk_bs_list[i]))
    x = [ens_list, ml_list, ms_list, alpha_list]
    y = np.array(data)
    err = np.cov(data_bs)
    # prior
    prior['L5'] = gv.gvar(0.0, 10.0)
    prior['cA'] = gv.gvar(0.0, 10.0)
    prior['cAa'] = gv.gvar(0.0, 10.0)
    #prior['d0'] = gv.gvar(0.0, 0.01)
    #prior['d1'] = gv.gvar(0.0, 0.01)
    #prior['d2'] = gv.gvar(0.0, 0.01)
    #prior['d3'] = gv.gvar(0.0, 0.01)
    #prior['dA'] = gv.gvar(0.0, 0.01)
    # fit
    print "Priors:", prior
    fit = lsqfit.nonlinear_fit(data=(x,y,err), fcn=fit_fcn, prior=prior)
    print fit
    # continuum fk/fpi
    phys_pion = gv.gvar(139.57018, 0.00035) #PDG 2016 pi+ MeV
    phys_kaon = gv.gvar(493.677 , 0.016) #PDG 2016 k+ MeV
    phys_fpi = gv.gvar(130.2, 1.7) #PDG fpi+ Table 1 http://pdg.lbl.gov/2015/reviews/rpp2015-rev-pseudoscalar-meson-decay-cons.pdf
    mpifpi = phys_pion**2/phys_fpi**2
    mkafpi = phys_kaon**2/phys_fpi**2
    pifac = 1./(16.*np.pi**2)
    ratio = 1. + 5./4. * pifac * mpifpi * np.log(mpifpi) - 1./2. * pifac * mkafpi * np.log(mkafpi) - pifac * (mkafpi - 1./4.*mpifpi) * np.log(4./3.*mkafpi - 1./3.*mpifpi) + 8.*(mkafpi - mpifpi)*fit.p['L5']
    # NNLO
    #ratio += phys_pion**2 * (2.*mkafpi - mpifpi) * fit.p['d0']
    #ratio += phys_kaon**2 * (2.*mkafpi - mpifpi) * fit.p['d1']
    #ratio += (2.*phys_kaon**2 - phys_pion**2) * (2.*mkafpi - mpifpi) * fit.p['d2']
    print "physical fk/fpi:", ratio
