import h5py as h5
import numpy as np
import c51lib as c51
import matplotlib.pyplot as plt

#Read meson data
filename = 'l2464f211b600m0102m0509m635a_avg.h5'
datapath = 'l2464f211b600m0102m0509m635/wf1p0_m51p2_l58_a51p5_smrw5p0_n75/spectrum/ml0p0126_ms0p0693/pion/corr'
data_ss = c51.read_data(filename, datapath, 0, 0)
data_ps = c51.read_data(filename, datapath, 1, 0)

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
c51.stability_plot(fit, 'Z0_p')
plt.show()
