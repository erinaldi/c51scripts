# reads from pickled gvar file
import numpy as np
import calsql as sql
import sqlc51lib as c51
import password_file as pwd
import yaml
from collections import OrderedDict
import gvar as gv
import lsqfit
import cPickle as pickle

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

def fit_fcn(x, p):
    fit = []
    for k in x:
        pipi = p['%s_mpion2fpion2' %k]
        kapi = p['%s_mkaon2fpion2' %k]
        R = 1
        R += 5./4. * (1./(16.*np.pi**2)) * pipi * np.log(pipi)
        R += -1./2. * (1./(16.*np.pi**2)) * kapi * np.log(kapi)
        R += -(1./(16.*np.pi**2)) * (kapi-1./4.*pipi) * np.log(4./3.*kapi-1./3.*pipi)
        R += 8. * (kapi-pipi) * p['L5']
        # a^2
        R += p['C'] * p['%s_a' %k]**2
        # alpha_s a^2
        R += p['C_alpha'] * p['%s_as' %k] * p['%s_a' %k]**2
        # a^4
        R += p['D'] * p['%s_a' %k]**4
        # (mk^2-mpi^2) a^2
        R += p['Da'] * (kapi-pipi) * p['%s_a' %k]**2
        # (mk^2-mpi^2) a^2 alpha_s
        R += p['Da'] * (kapi-pipi) * p['%s_as' %k] * p['%s_a' %k]**2
        fit.append(R)
    return np.array(fit)

def solve(x, y, p):
    # input fk/fpi, mk/fpi, mpi/fpi and solve for l5 from the NLO chipt expression per ensemble
    l5 = []
    for i, k in enumerate(x):
        pipi = p['%s_mpion2fpion2' %k]
        kapi = p['%s_mkaon2fpion2' %k]
        fkfpi = y[i]
        R = 1
        R += 5./4. * (1./(16.*np.pi**2)) * pipi * np.log(pipi)
        R += -1./2. * (1./(16.*np.pi**2)) * kapi * np.log(kapi)
        R += -(1./(16.*np.pi**2)) * (kapi-1./4.*pipi) * np.log(4./3.*kapi-1./3.*pipi)
        S = (fkfpi - R)/(8. * (kapi-pipi))
        l5.append(S)
    return l5

if __name__=='__main__':
    # read master
    f = open('./decay_chipt_gvar.yml', 'r')
    dataset = ordered_load(f,yaml.SafeLoader)['decay_chipt']['dataset']
    f.close()
    # log in sql
    psqlpwd = pwd.passwd()
    psql = sql.pysql('cchang5','cchang5',psqlpwd)
    # read alphas
    alpha = dict()
    alpha['l1648f211b580m013m065m838']    = gv.gvar(0.58801, 0.0000001)
    alpha['l2448f211b580m0064m0640m828']  = gv.gvar(0.58801, 0.0000001)
    alpha['l3248f211b580m00235m0647m831'] = gv.gvar(0.58801, 0.0000001)
    alpha['l2464f211b600m0102m0509m635']  = gv.gvar(0.53791, 0.0000001)
    alpha['l4064f211b600m00507m0507m628'] = gv.gvar(0.53791, 0.0000001)
    alpha['l4864f211b600m00184m0507m628'] = gv.gvar(0.53791, 0.0000001)
    alpha['l3296f211b630m0074m037m440']   = gv.gvar(0.43351, 0.0000001)
    alpha['l4896f211b630m00363m0363m430'] = gv.gvar(0.43351, 0.0000001)
    # read pickle
    prior = dict()
    x = dataset.keys()
    y = []
    for k in x:
        g = pickle.load(open('./pickle_result/fkfpi_%s.pickle' %k, 'rb'))
        y.append(g[k]['data']['fkfpi'])
        prior['%s_mpion2fpion2' %k] = g[k]['priors']['mpion2fpion2']
        prior['%s_mkaon2fpion2' %k] = g[k]['priors']['mkaon2fpion2']
        # get a
        sql_cmd = "SELECT a_fm FROM callat_corr.hisq_ensembles WHERE tag='%s';" %(k)
        psql.cur.execute(sql_cmd)
        a_fm = psql.cur.fetchone()[0]
        prior['%s_a' %k] = gv.gvar(float(a_fm[:-4]), 10**(-1.0*int(len(a_fm[:-4].split('.')[1])))*float(a_fm[-3:-1]))
        # get alpha_s
        prior['%s_as' %k] = alpha[k]
    prior['L5'] = gv.gvar(0.0, 1.0)
    prior['C'] = gv.gvar(0.0, 10.0) # a^2 coefficient
    prior['C_alpha'] = gv.gvar(0.0, 10.0) # alpha_s a^2 coefficient
    prior['D'] = gv.gvar(0.0, 10.0)
    prior['Da'] = gv.gvar(0.0, 10.0)
    prior['D_alpha'] = gv.gvar(0.0, 10.0)
    print prior
    print x
    print y
    # fit
    fit = lsqfit.nonlinear_fit(data=(x,y), fcn=fit_fcn, prior=prior)
    print fit

    # continuum fk/fpi
    phys_pion = gv.gvar(139.57018, 0.00035) #PDG 2016 pi+ MeV
    phys_kaon = gv.gvar(493.677 , 0.016) #PDG 2016 k+ MeV
    phys_fpi = gv.gvar(130.2, 1.7) #PDG fpi+ Table 1 http://pdg.lbl.gov/2015/reviews/rpp2015-rev-pseudoscalar-meson-decay-cons.pdf
    mpifpi = phys_pion**2/phys_fpi**2
    mkafpi = phys_kaon**2/phys_fpi**2
    pifac = 1./(16.*np.pi**2)
    ratio = 1. + 5./4. * pifac * mpifpi * np.log(mpifpi) - 1./2. * pifac * mkafpi * np.log(mkafpi) - pifac * (mkafpi - 1./4.*mpifpi) * np.log(4./3.*mkafpi - 1./3.*mpifpi) + 8.*(mkafpi - mpifpi)*fit.p['L5']
    print "physical fk/fpi: %s" %ratio
    #l5 = solve(x,y,prior)
    #print l5
