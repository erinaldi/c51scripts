import h5py as h5
import numpy as np
import c51lib as c51
import matplotlib.pyplot as plt

#Read spin averaged positive parity proton
#filename = 'l1648f211b580m013m065m838a_avg.h5'
#path = 'l1648f211b580m013m065m838a/wf1p0_m51p3_l512_a52p0_smrw4p5_n60/spec_ml0p0158_ms0p0902/'
#filename = 'l2464f211b600m0102m0509m635a_avg.h5'
#path = 'l2464f211b600m0102m0509m635/wf1p0_m51p2_l58_a51p5_smrw4p45_n60/spectrum/ml0p0126_ms0p0696/'
#filename = 'l2464f211b600m0102m0509m635a_tune_avg.h5'
#path = 'l2464f211b600m0102m0509m635/wf1p0_m51p2_l58_a51p5_smrw4p0_n100/spectrum/ml0p0126_ms0p0696/'
#path = 'l2464f211b600m0102m0509m635/wf1p0_m51p2_l58_a51p5_smrw7p0_n150/spectrum/ml0p0126_ms0p0696/'
#path = 'l2464f211b600m0102m0509m635/wf1p0_m51p2_l58_a51p5_smrw5p0_n75/spectrum/ml0p0126_ms0p0696/'
#filename = 'l3296f211b630m0074m037m440e_tune_avg.h5'
#path = 'l3296f211b630m0074m037m440/wf1p0_m51p1_l56_a51p5_smrw7p5_n167/spectrum/ml0p00950_ms0p0490/'
#filename = 'l3264f211b600m00507m0507m628a_avg.h5'
#path = 'l3264f211b600m00507m0507m628/wf1p0_m51p2_l512_a52p0_smrw5p5_n75/spectrum/ml0p00600_ms0p0693/'
#path = 'l3264f211b600m00507m0507m628/wf1p0_m51p2_l512_a52p0_smrw6p0_n90/spectrum/ml0p00600_ms0p0693/'
filename = 'l3248f211b580m00235m0647m831a_avg.h5'
path = 'l3248f211b580m00235m0647m831/wf1p0_m51p3_l524_a53p5_smrw4p5_n60/spectrum/ml0p00211_ms0p0902/'

datapath = path+'proton/spin_up'
data_up = c51.read_data(filename, datapath, 0, 0)
datapath = path+'proton/spin_dn'
data_dn = c51.read_data(filename, datapath, 0, 0)
data_pos = c51.ispin_avg(data_up, data_dn)
#Read spin averaged negative parity proton
datapath = path+'proton_np/spin_up'
data_up = c51.read_data(filename, datapath, 0, 0)
datapath = path+'proton_np/spin_dn'
data_dn = c51.read_data(filename, datapath, 0, 0)
data_neg = c51.ispin_avg(data_up, data_dn)
#Parity average
data_ss = c51.parity_avg(data_pos, data_neg, -1)
datapath = path+'proton/spin_up'
data_up = c51.read_data(filename, datapath, 3, 0)
datapath = path+'proton/spin_dn'
data_dn = c51.read_data(filename, datapath, 3, 0)
data_pos = c51.ispin_avg(data_up, data_dn)
#Read spin averaged negative parity proton
datapath = path+'proton_np/spin_up'
data_up = c51.read_data(filename, datapath, 3, 0)
datapath = path+'proton_np/spin_dn'
data_dn = c51.read_data(filename, datapath, 3, 0)
data_neg = c51.ispin_avg(data_up, data_dn)
#Parity average
data_ps = c51.parity_avg(data_pos, data_neg, -1)

##READ SPIN AND PARITY AVG'D DATA
#filename = 'l2464f211b600m0102m0509m635_tune_avg.h5'
#datapath = 'prot_w6p1_n94'
#datapath = 'prot_w5p6_n94'
filename = 'l2464f211b600m0102m0509m635a_avg.h5'
datapath = 'l2464f211b600m0102m0509m635/wf1p0_m51p2_l58_a51p5_smrw5p0_n75/spectrum/ml0p0126_ms0p0693/pion/corr'
data_ss = c51.read_data(filename, datapath, 0, 0)
data_ps = c51.read_data(filename, datapath, 3, 0)

data = np.concatenate((data_ss, data_ps), axis=1)
data = c51.make_gvars(data)
#data = data_ps

#Plot effective mass
T = len(data)*0.5
meff = c51.effective_mass(data, 1)
x = np.arange(len(meff))
ylim = c51.find_yrange(meff, 1, 10)
#ylim = [0.47, 0.57]
xr = [1,15]
c51.scatter_plot(x, meff, 'effective mass', xlim=[xr[0],xr[1]], ylim=ylim)
#ylim = c51.find_yrange(meff, 65, 79)
c51.scatter_plot(x, meff, 'effective mass ps', xlim=[T+xr[0],T+xr[1]], ylim=ylim)

#Fit
inputs = c51.read_yaml('temp.yml')
prior = c51.dict_of_tuple_to_gvar(inputs['prior'])
trange = inputs['trange']
fitfcn = c51.fit_function(T, nstates=2)
fit = c51.fitscript(trange, data, prior, fitfcn.twopt_fitfcn_ss_ps, sets=2, result_flag='on')
c51.stability_plot(fit, 'E0')
c51.stability_plot(fit, 'Z0_s')
plt.show()
