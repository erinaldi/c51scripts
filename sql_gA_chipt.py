import sys
sys.path.append('$HOME/c51/scripts/')
import sqlc51lib as c51
import calsql as sql
import password_file as pwd
import numpy as np
import matplotlib.pyplot as plt
import gvar as gv
import yaml
import tqdm
import lsqfit 
import get_gAbs as getga

def select_aw0():
    aw0 = {'l1648f211b580m013m065m838':     gv.gvar(0.8804, 0.5*(0.0082+0.0064)),
            'l2448f211b580m0064m0640m828':  gv.gvar(0.8804, 0.5*(0.0082+0.0064)),
            'l3248f211b580m00235m0647m831': gv.gvar(0.8804, 0.5*(0.0082+0.0064)),
            'l2464f211b600m0102m0509m635':  gv.gvar(0.7036, 0.5*(0.0064+0.0052)),
            'l2464f211b600m00507m0507m628': gv.gvar(0.7036, 0.5*(0.0064+0.0052)),
            'l3264f211b600m00507m0507m628': gv.gvar(0.7036, 0.5*(0.0064+0.0052)),
            'l4064f211b600m00507m0507m628': gv.gvar(0.7036, 0.5*(0.0064+0.0052)),
            'l4864f211b600m00184m0507m628': gv.gvar(0.7036, 0.5*(0.0064+0.0052)),
            'l3296f211b630m0074m037m440':   gv.gvar(0.5105, 0.5*(0.0047+0.0041)),
            'l4896f211b630m00363m0363m430': gv.gvar(0.5105, 0.5*(0.0047+0.0041))}
    return aw0

def select_a2DI():
    a2DI = {'l1648f211b580m013m065m838':  gv.gvar(0.3678,0.0441)/gv.gvar(2.059,0.023)**2,
            'l2448f211b580m0064m0640m828':  gv.gvar(0.3678,0.0441)/gv.gvar(2.073,0.013)**2,
            'l3248f211b580m00235m0647m831': gv.gvar(0.3678,0.0441)/gv.gvar(2.089,0.008)**2,
            'l2464f211b600m0102m0509m635':  gv.gvar(0.2068,0.0172)/gv.gvar(2.575,0.017)**2,
            'l2464f211b600m00507m0507m628': gv.gvar(0.2068,0.0172)/gv.gvar(2.585,0.019)**2,
            'l3264f211b600m00507m0507m628': gv.gvar(0.2068,0.0172)/gv.gvar(2.626,0.013)**2,
            'l4064f211b600m00507m0507m628': gv.gvar(0.2068,0.0172)/gv.gvar(2.614,0.009)**2,
            'l4864f211b600m00184m0507m628': gv.gvar(0.2068,0.0172)/gv.gvar(2.608,0.008)**2,
            'l3296f211b630m0074m037m440':   gv.gvar(0.0631,0.0051)/gv.gvar(3.499,0.024)**2,
            'l4896f211b630m00363m0363m430': gv.gvar(0.0631,0.0051)/gv.gvar(3.566,0.014)**2}
    return a2DI

print select_a2DI()

c=299792458 #[m/s]
hbar=6.58211928E-16 #[ev s]
dconv=1E15 #[fm/m]
econv=1E-6 #[Mev/eV]
chbar=c*hbar*dconv*econv #[fm MeV]
afp4s = {'l1648f211b580m013m065m838':   gv.gvar(0.1520,0.0008),
        'l2448f211b580m0064m0640m828':  gv.gvar(0.1528,0.0006),
        'l3248f211b580m00235m0647m831': gv.gvar(0.1531,0.0007),
        'l2464f211b600m0102m0509m635':  gv.gvar(0.1224,0.0006),
        'l2464f211b600m00507m0507m628': gv.gvar(0.1220,0.0005),
        'l3264f211b600m00507m0507m628': gv.gvar(0.1220,0.0005),
        'l4064f211b600m00507m0507m628': gv.gvar(0.1219,0.0005),
        'l4864f211b600m00184m0507m628': gv.gvar(0.1221,0.0006),
        'l3296f211b630m0074m037m440':   gv.gvar(0.0887,0.0005),
        'l4896f211b630m00363m0363m430': gv.gvar(0.0882,0.0004)}
m5 = {'l1648f211b580m013m065m838':    gv.gvar(306.9,0.5)/chbar*gv.gvar(0.1520,0.0008),
      'l2448f211b580m0064m0640m828':  gv.gvar(214.5,0.2)/chbar*gv.gvar(0.1528,0.0006),
      'l3248f211b580m00235m0647m831': gv.gvar(131.0,0.1)/chbar*gv.gvar(0.1531,0.0007),
      'l2464f211b600m0102m0509m635':  gv.gvar(305.3,0.4)/chbar*gv.gvar(0.1224,0.0006),
      'l2464f211b600m00507m0507m628': gv.gvar(218.1,0.4)/chbar*gv.gvar(0.1220,0.0005),
      'l3264f211b600m00507m0507m628': gv.gvar(216.9,0.2)/chbar*gv.gvar(0.1220,0.0005),
      'l4064f211b600m00507m0507m628': gv.gvar(217.0,0.2)/chbar*gv.gvar(0.1219,0.0005),
      'l4864f211b600m00184m0507m628': gv.gvar(131.7,0.1)/chbar*gv.gvar(0.1221,0.0006),
      'l3296f211b630m0074m037m440':   gv.gvar(312.7,0.6)/chbar*gv.gvar(0.0887,0.0005),
      'l4896f211b630m00363m0363m430': gv.gvar(220.3,0.2)/chbar*gv.gvar(0.0882,0.0004)}
a2Dmj = {'l1648f211b580m013m065m838':    gv.gvar(0.0439,0.0041)*gv.gvar(0.8804, 0.5*(0.0082+0.0064))**2,
       'l2448f211b580m0064m0640m828':  gv.gvar(0.0488,0.0038)*gv.gvar(0.8804, 0.5*(0.0082+0.0064))**2,
       'l3248f211b580m00235m0647m831': gv.gvar(0.0450,0.0040)*gv.gvar(0.8804, 0.5*(0.0082+0.0064))**2, # guessed
       'l2464f211b600m0102m0509m635':  gv.gvar(0.0214,0.0017)*gv.gvar(0.7036, 0.5*(0.0064+0.0052))**2,
       'l2464f211b600m00507m0507m628': gv.gvar(0.0279,0.0013)*gv.gvar(0.7036, 0.5*(0.0064+0.0052))**2, # 3264 number
       'l3264f211b600m00507m0507m628': gv.gvar(0.0279,0.0013)*gv.gvar(0.7036, 0.5*(0.0064+0.0052))**2,
       'l4064f211b600m00507m0507m628': gv.gvar(0.0279,0.0013)*gv.gvar(0.7036, 0.5*(0.0064+0.0052))**2,
       'l4864f211b600m00184m0507m628': gv.gvar(0.0250,0.0020)*gv.gvar(0.7036, 0.5*(0.0064+0.0052))**2, # guessed
       'l3296f211b630m0074m037m440':   gv.gvar(0.0102,0.0009)*gv.gvar(0.5105, 0.5*(0.0047+0.0041))**2,
       'l4896f211b630m00363m0363m430': gv.gvar(0.0102,0.0009)*gv.gvar(0.5105, 0.5*(0.0047+0.0041))**2} # guessed
# priors
priors = dict()
priors['g0'] = gv.gvar(1.0, 1.0)
priors['g0t'] = gv.gvar(0.55, 0.5)
priors['C1'] = gv.gvar(0.0, 1.0)
priors['C2'] = gv.gvar(0.0, 10.0)
priors['A1'] = gv.gvar(0.0, 10.0)
priors['CA1'] = gv.gvar(0.0, 10.0)

def fitfcn(x,p):
    #fit = p['gA'] + p['C2']*p['x']**2 + p['A1']*p['aw0']**2
    #fit = p['gA'] + p['C1']*p['x'] + p['C2']*p['x']**2 + p['A1']*p['aw0']**2
    # Taylor A + B x^2 + C a^2 + D x^2 a^2
    #fit =  p['gA'] + p['C2']*p['x']**2 + p['A1']*p['aw0']**2 + p['CA1']*p['x']**2*p['aw0']**2
    # Taylow A         + C a^2 + D x^2 a^2
    #fit =  p['gA'] + p['A1']*p['aw0']**2 + p['CA1']*p['x']**2*p['aw0']**2
    #fit =  p['gA'] + p['CA1']*p['x']*p['aw0']
    # NLO chipt
    #fit = p['gA'] - (p['gA'] + 2*p['gA']**3) * p['x']**2 * np.log(p['x']**2) + p['C2']*p['x']**2 + p['A1']*p['aw0']**2 + p['CA1']*p['x']**2*p['aw0']**2
    # NLO mixed chipt
    fit = p['g0'] - p['epi']**2*np.log(p['epi']**2)*(p['g0']+2*p['g0']**3) + (14.*p['g0']*p['g0t']**2-15.*p['g0']**2*p['g0t']+p['g0t']**3)/12.*(p['emix']**2*np.log(p['emix']**2)-p['epi']**2*np.log(p['epi']**2)) + p['g0']*p['g0t']**2*p['a2DI']*(np.log(p['epi']**2)/(4.*np.pi*p['Fpi']**2)+1./(4.*np.pi*p['Fpi'])) + p['C2']*p['epi']**2 + p['A1']*p['aw0']**2
    return fit

def fpi(ml, E0, Z0_p, mres_pion):
    constant = Z0_p*(2.*ml+2.*mres_pion)/E0**(3./2.)
    return constant

def reject_outliers(data,e):
    if True: #e in ['l3248f211b580m00235m0647m831','l4896f211b630m00363m0363m430']:
        for i in range(10):
            idx = np.argmax(data,axis=0)[0]
            data = np.delete(data, idx, axis=0)
            gvdat = gv.dataset.avg_data(data,bstrap=True)
            idx = np.argmin(data,axis=0)[0]
            data = np.delete(data, idx, axis=0)
        print max(data[:,0])
        print min(data[:,0])
    else:
        print "pass",e 
    return data

psqlpwd = pwd.passwd()
psql = sql.pysql('cchang5','cchang5',psqlpwd)

gAV = dict()
eps = dict()
Fps = dict()
Epi = dict()
for e in data_idx.keys():
    # read gA and gV
    sqlcmd = "SELECT (result->>'gA00')::double precision / (result->>'gV00')::double precision FROM callat_proj.ga_v1_bs g JOIN callat_corr.hisq_bootstrap b ON g.bs_id=b.id WHERE g.ga_v1_id=%s AND mbs=%s ORDER BY nbs;" %(data_idx[e]['gA'],data_idx[e]['mbs'])
    psql.cur.execute(sqlcmd)
    gAV[e] = np.array(psql.cur.fetchall()).flatten()
    #sqlcmd = "SELECT (result->>'gV00')::double precision FROM callat_proj.ga_v1_bs g JOIN callat_corr.hisq_bootstrap b ON g.bs_id=b.id WHERE g.ga_v1_id=%s ORDER BY nbs;" %data_idx[e]['gA']
    #psql.cur.execute(sqlcmd)
    #gV[e] = np.array(psql.cur.fetchall()).flatten()
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
    # make Fpi
    Fps[e] = fpi(ml, E, Z, mres)
    # make epsilon
    eps[e] = E/(4.*np.pi*Fps[e])
    # make mpi
    Epi[e] = E

# make data
output = dict()
x = []
y = []
a = []
f = []
emix = []
Dpq = []
for e in data_idx.keys():
    if e == 'l1648f211b580m013m065m838':
        x.append(gv.gvar(0.2495, 0.0004))
        y.append(gv.gvar(1.2240, 0.0086))
    elif e == 'l2448f211b580m0064m0640m828':
        x.append(gv.gvar(0.1814, 0.0004))
        y.append(gv.gvar(1.2339, 0.0101))
    else:
        dat = np.transpose([gAV[e],eps[e]])
        boot0 = dat[0] #{'y': dat[0,0], 'x': dat[0,1]}
        bootn = dat[1:] #{'y': dat[1:,0], 'x': dat[1:,1]}
        gvdat = reject_outliers(bootn,e) # reject outliers
        print len(gvdat)
        #gvdat = gv.dataset.avg_data(bootn,bstrap=True) # make array into gvars
        #sdev = gv.evalcov(gvdat) # gvars covariance
        sdev = np.cov(gvdat,rowvar=False) # numpy covariance
        #sdev = np.std(gvdat,axis=0) # numpy standard deviation
        #yCI = 0.5*(np.sort(bootn[:,0])[int(len(bootn)*0.841344746)] - np.sort(bootn[:,0])[int(len(bootn)*0.158655254)]) # CI
        #xCI = 0.5*(np.sort(bootn[:,1])[int(len(bootn)*0.841344746)] - np.sort(bootn[:,1])[int(len(bootn)*0.158655254)]) # CI
        #sdev = [yCI, xCI] # CI
        #print e, yCI
        #print "gA/gV:", boot0[0], np.sqrt(sdev[0,0])
        #print "epsil:", boot0[1], np.sqrt(sdev[1,1])
        output[e] = {}
        output[e]['y'], output[e]['x'] = gv.gvar(boot0, sdev)
        x.append(output[e]['x'])
        y.append(output[e]['y'])
    a.append(aw0[e])
    f.append(gv.dataset.avg_data(Fps[e]))
    emix.append((0.5*(gv.dataset.avg_data(Epi[e])**2+m5[e]**2)+a2Dmj[e])/(4.*np.pi*gv.dataset.avg_data(Fps[e])))
    Dpq.append(a2DI[e])

priors['epi'] = x
priors['aw0'] = a
priors['emix'] = emix
priors['Fpi'] = f
priors['a2DI'] = Dpq
x = data_idx.keys()
y = np.array(y)
fit = lsqfit.nonlinear_fit(data=(x,y),prior=priors,fcn=fitfcn,maxit=10000)
posterior = dict()
for k in fit.p.keys():
    if k == 'epi':
        posterior['epi'] = 0.12
    elif k == 'emix':
        posterior['emix'] = 0.12
    elif k == 'a2DI':
        posterior['a2DI'] = 0.0
    elif k == 'aw0':
        posterior['aw0'] = 0.0
    else:
        posterior[k] = fit.p[k]
print posterior
print fit
print "gA         :", fitfcn(0.12, posterior)
#print gvgAV
#print gveps

#print "epsilon    :", fit.p['gA'].partialsdev(fit.prior['x'][0])/fit.p['gA'].mean*100
#print "a/w0       :", fit.p['gA'].partialsdev(fit.prior['aw0'])/fit.p['gA'].mean*100
#print "disc       :", fit.p['gA'].partialsdev(fit.prior['A1'])/fit.p['gA'].mean*100
#print "chiral     :", fit.p['gA'].partialsdev(fit.prior['C2'],fit.prior['gA'])/fit.p['gA'].mean*100
#print "statistical:", fit.p['gA'].partialsdev(fit.y)/fit.p['gA'].mean*100
