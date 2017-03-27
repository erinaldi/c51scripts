import iminuit as m
import numpy as np

def f(a,**kwargs):
    y = a**2+2*a+3
    #for k in kwargs:
    #    if k == 'a':
    #        y += kwargs[k]**2
    #    elif k == 'b':
    #        y += kwargs[k]*2.
    return y
            
init = {'a':100}
r = m.Minuit(f,**init)
r.migrad()
print r.values
print f(a=1,b=2,c=3)
