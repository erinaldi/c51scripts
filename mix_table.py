# read pickled grand flow results and format in latex

import cPickle as pickle

ensemble = ['l1648f211b580m013m065m838', 'l2464f211b600m0102m0509m635', 'l3296f211b630m0074m037m440', 'l2448f211b580m0064m0640m828', 'l3264f211b600m00507m0507m628']
enstag = ['a15m310', 'a12m310', 'a09m310', 'a15m220', 'a12m220']

# w0a
w0a = dict()
w0a['a15'] = 1.1358515573227301
w0a['a12'] = 1.4212271973466004
w0a['a09'] = 1.9588571428571429

string = "ensemble& $m_\pi$& $m_\kappa$& $m_{\phi_{ss}}$& $m_{jj_{5}}$& $m_{jr_{5}}$& $m_{rr_{5}}$& $m_{ju}$& $m_{js}$& $m_{ru}$& $m_{rs}$ & $ju*w0a$ & $js*w0a$& $ru*w0a$& $rs*w0a$ \\\\ \n"
for i,e in enumerate(ensemble):
    a = enstag[i].split('m')[0]
    string += "\hline \n"
    g = pickle.load(open('./pickle_result/mix_%s.pickle' %e, 'rb'))
    string += "%s& %s& %s& %s& %s& %s& %s& %s& %s& %s& %s& %s& %s& %s& %s&  \\\\ \n" %(enstag[i], g['mpi'], g['mka'], g['met'], g['mjj'], g['mjr'], g['mrr'], g['mju'], g['mjs'], g['mru'], g['mrs'], g['ju']*w0a[a]**2, g['js']*w0a[a]**2, g['ru']*w0a[a]**2, g['rs']*w0a[a]**2)
print string
            
