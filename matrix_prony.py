# implement matrix prony for proton two point
import sys
sys.path.append('$HOME/c51/scripts/')
import c51lib as c51
import tuneproton as tp
import numpy as np
import matplotlib.pyplot as plt
import gvar as gv
import multiprocessing as multi
from tabulate import tabulate
import yaml
import collections


if __name__=='__main__':
    # read params
    pr = c51.process_params()
    tau = 4 # M(tau)
    tmin = 9
    tmax = 15
    # read data
    p_avg, T = tp.read_proton(pr)
    p_ss = p_avg[:,:,0,0]
    p_ps = p_avg[:,:,3,0]
    # matrix prony read 1301.1114 starting Eq. 12
    # V_inv and M_inv have shape 2 x 2 x Nconf x T
    V_inv = np.array([[p_ps*p_ps, p_ps*p_ss], [p_ss*p_ps, p_ss*p_ss]])
    p_ss_roll = np.roll(p_ss, tau)
    p_ps_roll = np.roll(p_ps, tau)
    M_inv = np.array([[p_ps_roll*p_ps, p_ps_roll*p_ss], [p_ss_roll*p_ps, p_ss_roll*p_ss]])
    # avg configs
    V_inv = np.average(V_inv, axis=2)
    M_inv = np.average(M_inv, axis=2)
    # sum over T
    V_inv = np.sum(V_inv[:,:,tmin:tmax], axis=2)
    M_inv = np.sum(M_inv[:,:,tmin:tmax], axis=2)
    # invert V and M
    #V = np.zeros(np.shape(V_inv), float)
    #for g in range(np.shape(V_inv)[2]):
        #V[:,:,g] = np.linalg.inv(V_inv[:,:,g])
    #M = np.zeros(np.shape(M_inv), float)
    #for g in range(np.shape(M_inv)[2]):
    #    M[:,:,g] = np.linalg.inv(M_inv[:,:,g])
    V = np.linalg.inv(V_inv)
    M = np.linalg.inv(M_inv)
    # construct transfer matrix
    #T = np.zeros(np.shape(M_inv), float)
    #for g in range(np.shape(M_inv)[2]):
    #    T[:,:,g] = np.dot(M_inv[:,:,g], V[:,:,g])
    T = np.dot(M_inv, V)
    # calculate eigenvalues
    #for g in range(np.shape(T)[2]):
    #    eigval, eigvec = np.linalg.eig(T[:,:,g])
    #    print eigval
    eigval, eigvec = np.linalg.eig(T)
    print np.log(eigval**(1./tau))
    print eigvec[:,0]
    print eigvec[:,1]
