import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np

ensemble = 'l1648f211b580m013m065m838'

f = open('%s_gAgV.pickle' %ensemble,'r')
bs = pickle.load(f)
f.flush()
f.close()
bssort = np.sort(bs['gAgV'])
bstruncate = bssort[10:len(bssort)-10]

CI = [bssort[int(len(bssort)*0.158655254)], bssort[int(len(bssort)*0.841344746)], bssort[int(len(bssort)*0.5)]]
print "median: %s" %CI[2]
print "dCI: %s" %(0.5*(CI[1]-CI[0]))

plt.figure()
n, bins, patches = plt.hist(bstruncate, 50, facecolor='green')
x = np.delete(bins, -1)
plt.plot(x, n)
plt.xlabel('gA/gV')
plt.ylabel('counts')
plt.draw()
plt.show()
