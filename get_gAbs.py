import sys
sys.path.append('$HOME/c51/scripts/')
import calsql as sql
import password_file as pwd
import numpy as np
import tqdm
import gvar as gv

def select_data():
    # Full ensembles
    #data_idx = {'l1648f211b580m013m065m838':    {'gA': 186, 'mpi': 47, 'mres': 26, 'mju': 13, 'mbs': 1960},
    #            'l2448f211b580m0064m0640m828':  {'gA': 162, 'mpi': 48, 'mres': 27, 'mju': 11, 'mbs': 1000},
    #            'l3248f211b580m00235m0647m831': {'gA': 9,   'mpi': 40, 'mres': 20, 'mju': 12, 'mbs': 1000},
    #            'l2464f211b600m0102m0509m635':  {'gA': 3,   'mpi': 27, 'mres': 29, 'mju': 5,  'mbs': 1053},
    #            'l2464f211b600m00507m0507m628': {'gA': 43,  'mpi': 46, 'mres': 25, 'mju': 6,  'mbs': 1000}, #mju from 3264
    #            'l3264f211b600m00507m0507m628': {'gA': 7,   'mpi': 26, 'mres': 13, 'mju': 6,  'mbs': 1000},
    #            'l4064f211b600m00507m0507m628': {'gA': 47,  'mpi': 45, 'mres': 24, 'mju': 6,  'mbs': 1000}, #mju from 3264
    #            'l4864f211b600m00184m0507m628': {'gA': 11,  'mpi': 24, 'mres': 14, 'mju': 10, 'mbs': 1000},
    #            'l3296f211b630m0074m037m440':   {'gA': 126, 'mpi': 49, 'mres': 28, 'mju': 1,  'mbs': 784},
    #            'l4896f211b630m00363m0363m430': {'gA': 8,   'mpi': 28, 'mres': 11, 'mju': 1,  'mbs': 1001}} #need mju
    
    # PRD
    #data_idx = {'l1648f211b580m013m065m838':    {'gA': 71, 'mpi': 43, 'mres': 21},
    #            'l2448f211b580m0064m0640m828':  {'gA': 10, 'mpi': 39, 'mres': 23},
    #            'l3248f211b580m00235m0647m831': {'gA': 9,  'mpi': 40, 'mres': 20},
    #            'l2464f211b600m0102m0509m635':  {'gA': 3,  'mpi': 27, 'mres': 12},
    #            'l2464f211b600m00507m0507m628': {'gA': 43, 'mpi': 46, 'mres': 25},
    #            'l3264f211b600m00507m0507m628': {'gA': 7,  'mpi': 26, 'mres': 13},
    #            'l4064f211b600m00507m0507m628': {'gA': 47, 'mpi': 45, 'mres': 24},
    #            'l3296f211b630m0074m037m440':   {'gA': 5,  'mpi': 29, 'mres': 1}}
    
    data_idx = {'l4864f211b600m00184m0507m628': {'gA0': 256, 'gA1': 265, 'gA2': 266, 'mbs': 1000}}
    return data_idx


def fpi(ml, E0, Z0_p, mres_pion):
    constant = Z0_p*(2.*ml+2.*mres_pion)/E0**(3./2.)
    return constant

def get_bootstrap():
    psqlpwd = pwd.passwd()
    psql = sql.pysql('cchang5','cchang5',psqlpwd)
    # get data
    data_idx = select_data()
    data = dict()
    data['gA'] = {}
    data['Fpi'] = {}
    data['mpi'] = {}
    data['mju'] = {}
    data['epi'] = {}
    data['eju'] = {}
    data['ml'] = {}
    data['Z0p'] = {}
    data['mres'] = {}
    data['aw0'] = {}
    data['a2DI'] = {}
    for e in data_idx.keys():
        # read gA and gV
        sqlcmd = "SELECT (result->>'gA00')::double precision / (result->>'gV00')::double precision FROM callat_proj.ga_v1_bs g JOIN callat_corr.hisq_bootstrap b ON g.bs_id=b.id WHERE g.ga_v1_id=%s AND mbs=%s ORDER BY nbs;" %(data_idx[e]['gA'],data_idx[e]['mbs'])
        psql.cur.execute(sqlcmd)
        gA = np.array(psql.cur.fetchall()).flatten()
        # read data from DB get E0 and Z0_p
        sqlcmd = "SELECT (result->>'E0')::double precision FROM callat_proj.meson_v1_bs bn JOIN callat_corr.hisq_bootstrap bs ON bn.bs_id = bs.id WHERE bn.meson_v1_id=%s AND mbs=%s ORDER BY nbs;" %(data_idx[e]['mpi'],data_idx[e]['mbs'])
        psql.cur.execute(sqlcmd)
        E = np.array(psql.cur.fetchall()).flatten()
        sqlcmd = "SELECT (result->>'Z0_p')::double precision FROM callat_proj.meson_v1_bs bn JOIN callat_corr.hisq_bootstrap bs ON bn.bs_id = bs.id WHERE bn.meson_v1_id=%s AND mbs=%s ORDER BY nbs;" %(data_idx[e]['mpi'],data_idx[e]['mbs'])
        psql.cur.execute(sqlcmd)
        Z = np.array(psql.cur.fetchall()).flatten()
        # read mres from DB
        sqlcmd = "SELECT (result->>'mres')::double precision FROM callat_proj.mres_v1_bs bn JOIN callat_corr.hisq_bootstrap bs ON bn.bs_id = bs.id WHERE bn.mres_v1_id=%s AND mbs=%s ORDER BY nbs;" %(data_idx[e]['mres'],data_idx[e]['mbs'])
        psql.cur.execute(sqlcmd)
        mres = np.array(psql.cur.fetchall()).flatten()
        # read ml
        sqlcmd = "SELECT p.mval::double precision FROM callat_proj.mres_v1 m JOIN callat_corr.dwhisq_corr_jmu j ON m.mp_id = j.id JOIN callat_corr.dwhisq_props p ON j.dwhisq_prop_id = p.id WHERE m.id = '%s';" %data_idx[e]['mres']
        psql.cur.execute(sqlcmd)
        ml = psql.cur.fetchone()[0]
        # read mju
        sqlcmd = "SELECT (result->>'E0')::double precision FROM callat_proj.mixed_v1_bs bn JOIN callat_corr.hisq_bootstrap bs ON bn.bs_id = bs.id WHERE bn.mixed_v1_id=%s AND mbs=%s ORDER BY nbs;" %(data_idx[e]['mju'],data_idx[e]['mbs'])
        psql.cur.execute(sqlcmd)
        Eju = np.array(psql.cur.fetchall()).flatten()
        # read aw0 and a2DI
        sqlcmd = "SELECT id FROM callat_corr.hisq_ensembles WHERE tag='%s';" %e
        psql.cur.execute(sqlcmd)
        hisq_id = psql.cur.fetchone()[0]
        sqlcmd = "SELECT aw0 FROM callat_corr.hisq_params_bootstrap WHERE hisq_ensembles_id=%s ORDER BY nbs;" %hisq_id
        psql.cur.execute(sqlcmd)
        aw0 = np.array(psql.cur.fetchall()).flatten()
        sql.cmd = "SELECT a2di FROM callat_corr.hisq_params_bootstrap WHERE hisq_ensembles_id=%s ORDER BY nbs;" %hisq_id
        psql.cur.execute(sqlcmd)
        a2di = np.array(psql.cur.fetchall()).flatten()
        # make data
        data['gA'][e] = gA
        data['Fpi'][e] = fpi(ml, E, Z, mres)
        data['epi'][e] = E/(4.*np.pi*data['Fpi'][e])
        data['eju'][e] = Eju/(4.*np.pi*data['Fpi'][e])
        data['mpi'][e] = E
        data['mju'][e] = Eju
        data['mres'][e] = mres
        data['ml'][e] = ml
        data['Z0p'][e] = Z
        data['aw0'][e] = aw0
        data['a2DI'][e] = a2di
    return data

def get_bootstrap_list():
    psqlpwd = pwd.passwd()
    psql = sql.pysql('cchang5','cchang5',psqlpwd)
    # get data
    data_idx = select_data()
    for ens in data_idx.keys():
        sqlcmd = "SELECT draws FROM callat_corr.hisq_bootstrap bs JOIN callat_corr.hisq_ensembles e ON bs.hisq_ensembles_id=e.id WHERE e.tag='%s' ORDER BY nbs LIMIT 500;" %ens
        psql.cur.execute(sqlcmd)
        nbs = psql.cur.fetchall()
        nbs = np.array([np.array(nbs[i][0])[:len(nbs[0][0])] for i in range(len(nbs))])
        s = ''
        for i in tqdm.tqdm(range(len(nbs))):
            for j in range(len(nbs[0])):
                if j == len(nbs[0])-1:
                    s += '%s' %nbs[i,j]
                else:
                    s += '%s,' %nbs[i,j]
            s += '\n'
        with open('./nbslist/nbs_%s.csv' %ens, 'w+') as f:
            f.write(s)
    return 

def get_systematic():
    psqlpwd = pwd.passwd()
    psql = sql.pysql('cchang5','cchang5',psqlpwd)
    # get data
    data_idx = select_data()
    ens = data_idx.keys()[0]
    l = []
    for n in range(len(data_idx[ens].keys())-1):
        # read gA and gV
        sqlcmd = "SELECT (result->>'gA00')::double precision / (result->>'gV00')::double precision FROM callat_proj.ga_v1_bs g JOIN callat_corr.hisq_bootstrap b ON g.bs_id=b.id WHERE g.ga_v1_id=%s AND mbs=%s ORDER BY nbs;" %(data_idx[ens]['gA%s' %n],data_idx[ens]['mbs'])
        psql.cur.execute(sqlcmd)
        gA = np.array(psql.cur.fetchall()).flatten()
        l.append(gA)
    l = np.array(l).T
    lgv = gv.dataset.avg_data(l,spread=True,median=False)
    #print lgv
    print np.mean(lgv)
    b0 = l[0,:]
    b0mean = np.mean(b0)
    #print b0mean
    cov = np.cov(l,rowvar=False)
    #print np.std(l,axis=0)
    SA = np.einsum('ij,j->i', cov,np.ones(len(b0)))
    #print SA
    ASA = np.einsum('i,i->', np.ones(len(b0)),SA)
    #print ASA
    print b0mean, np.sqrt(ASA)/len(b0)

    

if __name__=="__main__":
    #data = get_bootstrap()
    #print data['aw0']

    #nbs = get_bootstrap_list()

    sys = get_systematic()
