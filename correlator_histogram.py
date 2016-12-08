# plot correlator histogram
# this is to show that we are in the region where the CLT is valid.
# The correlator should be Gaussian.
# Without enough data, the correlator will be log-normal due to having an non-zero imaginary part. [arXiv:1611.07643 Wagman & Savage]

import sys
sys.path.append('$HOME/c51/scripts/')
import sqlc51lib as c51
import calsql as sql
import password_file as pwd
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

def read_gA_bs(psql,params):
    # read data
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
    print "reading for gA mq %s, basak %s, ens %s%s, Nbs %s, Mbs %s" %(str(mq),str(basak),str(tag),str(stream),str(Nbs),str(Mbs))
    # read two point
    SSl = np.array([psql.data('dwhisq_corr_baryon',idx) for idx in [barp['meta_id']['SS'][i] for i in basak]])
    PSl = np.array([psql.data('dwhisq_corr_baryon',idx) for idx in [barp['meta_id']['PS'][i] for i in basak]])
    T = len(SSl[0,0])
    # read fh correlator
    fhSSl = np.array([psql.data('dwhisq_fhcorr_baryon',idx) for idx in [fhbp['meta_id']['SS'][i] for i in basak]])
    fhPSl = np.array([psql.data('dwhisq_fhcorr_baryon',idx) for idx in [fhbp['meta_id']['PS'][i] for i in basak]])
    # concatenate and make gvars to preserve correlations
    SS = SSl[0]
    PS = PSl[0]
    for i in range(len(SSl)-1): # loop over basak operators
        SS = np.concatenate((SS,SSl[i+1]),axis=1)
        PS = np.concatenate((PS,PSl[i+1]),axis=1)
    fhSS = fhSSl[0]
    fhPS = fhPSl[0]
    for i in range(len(fhSSl)-1):
        fhSS = np.concatenate((fhSS,fhSSl[i+1]),axis=1)
        fhPS = np.concatenate((fhPS,fhPSl[i+1]),axis=1)
    boot0 = np.concatenate((SS, PS, fhSS, fhPS), axis=1)
    return SS, PS, fhSS, fhPS

def plot_histogram(data, databs, t, xlabel='x label', filename=''):
    plt.figure()
    # plot histogram
    n, bins, patches = plt.hist(databs, 50, facecolor='green', normed=True, alpha=0.5, histtype='bar')
    mean = np.mean(data)
    sdev = np.std(data)/np.sqrt(len(data))
    print mean, sdev
    x = np.linspace(bins[0], bins[-1], 500)
    y = mlab.normpdf(x, mean, sdev)
    plt.plot(x,y,'r--')
    plt.xlabel(xlabel)
    plt.ylabel('counts')
    plt.draw()
    f = open('/Users/cchang5/Documents/Papers/FH/c51_p1/paper/figures/%scorr_hist_t%s.csv' %(filename,str(t)), 'w+')
    string = "bins, count\n"
    for i in range(len(bins)-1):
        string += "%s, %s\n" %(str(bins[i]), str(n[i]))
    string +=  "\n x, gauss\n"
    for i in range(len(x)):
        string += "%s, %s\n" %(str(x[i]), str(y[i]))
    f.write(string)
    f.flush()
    f.close()
    return 0

def bootstrap(data,nbs):
    ncfg = len(data)
    databs = []
    for n in range(nbs):
        a = []
        for m in range(ncfg):
            a.append(np.random.randint(ncfg))
        databs.append(np.mean(data[a,:],axis=0))
    return np.array(databs)

if __name__=='__main__':
    # options
    t = 10 # time slice plotted
    nbs = 10000 # bootstrap sampling for mean
    # read master
    f = open('./fh.yml','r')
    params = yaml.load(f)
    f.close()
    # log in sql
    psqlpwd = pwd.passwd()
    psql = sql.pysql('cchang5','cchang5',psqlpwd)
    # read gA
    SS, PS, fhSS, fhPS = read_gA_bs(psql,params)
    # pick
    corr = SS
    filename = 'SS'
    # bootstrap this
    corrbs = bootstrap(corr,nbs)
    # plot histogram
    plot_histogram(corr[:,t], corrbs[:,t], t, 't=%s' %str(t), filename)
    plt.show()
