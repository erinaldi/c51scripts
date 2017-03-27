# tests randomness of random.randint
import random
import numpy as np
import matplotlib.pyplot as plt

def histogram(lst,nx,direction):
    plt.figure()
    n, bins, patches = plt.hist(lst, nx, facecolor='green')
    x = np.delete(bins, -1)
    plt.plot(x, n)
    plt.xlabel('source')
    plt.ylabel('counts '+direction)
    plt.draw()

def scatter(lst,nx,direction):
    plt.figure()
    plt.scatter(np.arange(len(lst)),lst)
    plt.xlabel('cfg')
    plt.ylabel('src location '+direction)
    plt.draw()

seed = 'a'
no_list = np.arange(300,5300,5)
nl = 16
nt = 48

x_list = []
y_list = []
z_list = []
t_list = []
for no in no_list:
    random.seed(seed+'.'+str(no))
    x0_0 = random.randint(0,int(nl)-1)
    y0_0 = random.randint(0,int(nl)-1)
    z0_0 = random.randint(0,int(nl)-1)
    t0_0 = random.randint(0,int(nt)-1)
    x_list.append(x0_0)
    y_list.append(y0_0)
    z_list.append(z0_0)
    t_list.append(t0_0)

histogram(x_list,nl,'x')
#histogram(y_list,nl,'y')
#histogram(z_list,nl,'z')
#histogram(t_list,nt,'t')
scatter(x_list,nl,'x')
#scatter(y_list,nl,'y')
#scatter(z_list,nl,'z')
#scatter(t_list,nl,'t')

random.seed(0)
x_list = []
y_list = []
z_list = []
t_list = []
for no in no_list:
    x0_0 = random.randint(0,int(nl)-1)
    y0_0 = random.randint(0,int(nl)-1)
    z0_0 = random.randint(0,int(nl)-1)
    t0_0 = random.randint(0,int(nt)-1)
    x_list.append(x0_0)
    y_list.append(y0_0)
    z_list.append(z0_0)
    t_list.append(t0_0)

histogram(x_list,nl,'x one seed')
#histogram(y_list,nl,'y one seed')
#histogram(z_list,nl,'z one seed')
#histogram(t_list,nt,'t one seed')
scatter(x_list,nl,'x one seed')
#scatter(y_list,nl,'y one seed')
#scatter(z_list,nl,'z one seed')
#scatter(t_list,nl,'t one seed')

plt.show()

