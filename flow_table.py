# read pickled grand flow results and format in latex

import cPickle as pickle

ensemble = ['l1648f211b580m013m065m838', 'l2464f211b600m0102m0509m635', 'l3296f211b630m0074m037m440', 'l2448f211b580m0064m0640m828', 'l3264f211b600m00507m0507m628']
enstag = ['a15m310', 'a12m310', 'a09m310', 'a15m220', 'a12m220']
wflowtag = ['0p2', '0p4', '0p6','0p8', '1p0']
wflow = ['0.2', '0.4', '0.6', '0.8', '1.0']

string = "ensemble& $t_{gf}$ & $Z_A^{ll}$ & $Z_A^{ls}$ & $aF_\pi$ & $aF_K$ & $am_N$ & $F_K / F_\pi$ & $m_N / F_\pi$ \\\\ \n"
for i,e in enumerate(ensemble):
    string += "\hline \n"
    for j,f in enumerate(wflowtag):
        g = pickle.load(open('./pickle_result/flow%s_%s.pickle' %(f,e), 'rb'))
        if f == '0p2':
            string += "%s & %s & %s & %s & %s & %s & %s & %s & %s \\\\ \n" %(enstag[i], wflow[j], g['ZAll'], g['ZAls'], g['fpi'], g['fka'], g['mN'], g['fk/fpi'], g['mN/fpi'])
        else:
            string += " & %s & %s & %s & %s & %s & %s & %s & %s \\\\ \n" %(wflow[j], g['ZAll'], g['ZAls'], g['fpi'], g['fka'], g['mN'], g['fk/fpi'], g['mN/fpi'])
print string
            
