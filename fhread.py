import h5py
import sqlc51lib as c51
import calsql as sql
import numpy as np
import yaml
import password_file as pwd

tau = 1

g = h5py.File('../data/l1648f211b580m013m065m838a_avg.h5-pe', 'r')
# pp
DDdn = g['/l1648f211b580m013m065m838a/wf1p0_m51p3_l512_a52p0_smrw4p5_n60/spectrum/ml0p0158_ms0p0902/proton/A3_DD/s_spin_dn_1pe'][()][:,:,0,0]
DDup = g['/l1648f211b580m013m065m838a/wf1p0_m51p3_l512_a52p0_smrw4p5_n60/spectrum/ml0p0158_ms0p0902/proton/A3_DD/s_spin_up_1pe'][()][:,:,0,0]
UUdn = g['/l1648f211b580m013m065m838a/wf1p0_m51p3_l512_a52p0_smrw4p5_n60/spectrum/ml0p0158_ms0p0902/proton/A3_UU/s_spin_dn_1pe'][()][:,:,0,0]
UUup = g['/l1648f211b580m013m065m838a/wf1p0_m51p3_l512_a52p0_smrw4p5_n60/spectrum/ml0p0158_ms0p0902/proton/A3_UU/s_spin_up_1pe'][()][:,:,0,0]
dn = g['/l1648f211b580m013m065m838a/wf1p0_m51p3_l512_a52p0_smrw4p5_n60/spectrum/ml0p0158_ms0p0902/proton/spin_dn_s_1pe'][()][:,:,0,0]
up = g['/l1648f211b580m013m065m838a/wf1p0_m51p3_l512_a52p0_smrw4p5_n60/spectrum/ml0p0158_ms0p0902/proton/spin_up_s_1pe'][()][:,:,0,0]
# np
npDDdn = g['/l1648f211b580m013m065m838a/wf1p0_m51p3_l512_a52p0_smrw4p5_n60/spectrum/ml0p0158_ms0p0902/proton_np/A3_DD/s_spin_dn_1pe'][()][:,:,0,0]
npDDup = g['/l1648f211b580m013m065m838a/wf1p0_m51p3_l512_a52p0_smrw4p5_n60/spectrum/ml0p0158_ms0p0902/proton_np/A3_DD/s_spin_up_1pe'][()][:,:,0,0]
npUUdn = g['/l1648f211b580m013m065m838a/wf1p0_m51p3_l512_a52p0_smrw4p5_n60/spectrum/ml0p0158_ms0p0902/proton_np/A3_UU/s_spin_dn_1pe'][()][:,:,0,0]
npUUup = g['/l1648f211b580m013m065m838a/wf1p0_m51p3_l512_a52p0_smrw4p5_n60/spectrum/ml0p0158_ms0p0902/proton_np/A3_UU/s_spin_up_1pe'][()][:,:,0,0]
npdn = g['/l1648f211b580m013m065m838a/wf1p0_m51p3_l512_a52p0_smrw4p5_n60/spectrum/ml0p0158_ms0p0902/proton_np/spin_dn_s_1pe'][()][:,:,0,0]
npup = g['/l1648f211b580m013m065m838a/wf1p0_m51p3_l512_a52p0_smrw4p5_n60/spectrum/ml0p0158_ms0p0902/proton_np/spin_up_s_1pe'][()][:,:,0,0]
# parity avg
DDdn = c51.parity_avg(DDdn,npDDdn,phase=1)
DDup = c51.parity_avg(DDup,npDDup,phase=1)
UUdn = c51.parity_avg(UUdn,npUUdn,phase=1)
UUup = c51.parity_avg(UUup,npUUup,phase=1)
dn = c51.parity_avg(dn,npdn,phase=-1)
up = c51.parity_avg(up,npup,phase=-1)

fh = np.mean(0.5*((UUup-UUdn) - (DDup - DDdn)),axis=0)
twopt = np.mean(0.5*(up + dn),axis=0)
r = fh/twopt
dm = (np.roll(r,-tau,axis=0)-r)/float(tau)
print dm 

# sql
# read master
user_flag = c51.user_list()
f = open('./fhprotonmaster.yml.%s' %(user_flag),'r')
params = yaml.load(f)
f.close()
# params
mq = params['gA_fit']['ml']
basak = params['gA_fit']['basak']
tag = params['gA_fit']['ens']['tag']
stream = params['gA_fit']['ens']['stream']
Nbs = params['gA_fit']['nbs']
Mbs = params['gA_fit']['mbs']
nstates = params['gA_fit']['nstates']
tau = params['gA_fit']['tau']
barp = params[tag]['proton'][mq]
fhbp = params[tag]['gA'][mq]
# log in sql
psqlpwd = pwd.passwd()
psql = sql.pysql('cchang5','cchang5',psqlpwd)
# read two point
SSl = np.array([psql.data('dwhisq_corr_baryon',idx) for idx in [barp['meta_id']['SS'][i] for i in basak]])[0]
gvSSl = c51.make_gvars(SSl)
T = len(gvSSl)
# read fh correlator
fhSSl = np.array([psql.data('dwhisq_fhcorr_baryon',idx) for idx in [fhbp['meta_id']['SS'][i] for i in basak]])[0]
gvfhSSl = c51.make_gvars(fhSSl)
concat = np.concatenate((SSl,fhSSl),axis=1)
gvcon = c51.make_gvars(concat)
L = len(gvcon)
# R(t)
RSSl = np.array(gvcon[L/2:]/gvcon[:L/2])
# dmeff [R(t+tau) - R(t)] / tau
dMSSl = (np.roll(RSSl,-tau)-RSSl)/float(tau)
print "psql"
print dMSSl
