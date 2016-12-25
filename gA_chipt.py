import matplotlib
matplotlib.use('QT4Agg')
from matplotlib import pyplot as plt
import numpy as np
import lsqfit
import gvar as gv
import yaml
import sqlc51lib as c51
import cPickle as pickle
import collections as col

def J(p,mpi,fpi):
    Del = 293. #MeV
    mu = 2.*np.sqrt(2.)*np.pi*fpi
    if mpi <= Del:
        J = 2.*Del*np.sqrt(Del**2-mpi**2) * np.log( (Del-np.sqrt(Del**2-mpi**2)) / (Del+np.sqrt(Del**2-mpi**2)) )
    else:
        x = mpi**2/Del**2
        J = 4*Del**2*np.sqrt(x-1)*np.arctan(np.sqrt(x-1))
    J += mpi**2*np.log(mpi**2/mu**2)
    J += 2*Del**2*np.log(4.*Del**2/mpi**2)
    return J/(4.*np.pi*fpi)**2

def K(p,mpi,fpi):
    Del = 293. #MeV
    mu = 2.*np.sqrt(2.)*np.pi*fpi
    if mpi <= Del:
        K = 2./3. * (Del**2-mpi**2)**(3./2.) / Del * np.log( (Del-np.sqrt(Del**2-mpi**2)) / (Del+np.sqrt(Del**2-mpi**2)) )
    else:
        x = mpi**2/Del**2
        K = -4.*Del**2/3.*(x-1)**(3./2.)*np.arctan(np.sqrt(x-1))
    K += 2.*np.pi/3. * mpi**3/Del 
    K += mpi**2*np.log(mpi**2/mu**2)
    K += 2./3. * Del**2 * np.log(4.*Del**2/mpi**2)
    return K/(4.*np.pi*fpi)**2

def gA_chipt(x,p):
    gA_array = []
    #gpiND = 6./5.*p['gpiNN']*(1+p['ND']) #float by 10%
    #gpiDD = -9./5.*p['gpiNN']*(1+p['DD']) #float by 10%
    for e in x:
        mpi = p['mpi_%s' %e]*197.326971780039/p['alat_%s' %e]
        fpi = p['fpi_%s' %e]*197.326971780039/p['alat_%s' %e]
        mu = 2.*np.sqrt(2.)*np.pi*p['fpi_%s' %e]
        gA = p['gpiNN']
        gA += -2. * (p['gpiNN']+2.*p['gpiNN']**3) * p['mpi_%s' %e]**2 / (4.*np.pi*p['fpi_%s' %e])**2 * np.log(p['mpi_%s' %e]**2 / mu**2)
        gA += p['cpiNN'] * p['mpi_%s' %e]**2 / (4.*np.pi*p['fpi_%s' %e])**2
        gA += p['ca'] * p['alat_%s' %e]**2
        # delta degrees of freedom
        #gA += -1.*p['gpiND']**2 * (4.*p['gpiNN'] + 100./81. * p['gpiDD']) * J(p,mpi,fpi)
        #gA += 32./9. * p['gpiNN']*p['gpiND']**2 * K(p,mpi,fpi)
        # fix delta degrees of freedom with gpiNN
        #gA += -1.*gpiND**2 * (4.*p['gpiNN'] + 100./81. * gpiDD) * J(p,mpi,fpi)
        #gA += 32./9. * p['gpiNN']*gpiND**2 * K(p,mpi,fpi)
        # alpha_s a^2
        #gA += p['cafs'] * p['alpha_%s' %e] * p['alat_%s' %e]**2 
        gA_array.append(gA)
    return gA_array

def gA_chiral(mpi,p, e='continuum'): #this uses physical fpi instead of lattice fpi
    fpi = gv.gvar(130.2, 1.7) #[http://pdg.lbl.gov/2015/reviews/rpp2015-rev-pseudoscalar-meson-decay-cons.pdf]
    mu = 2.*np.sqrt(2.)*np.pi*fpi
    gA = p['gpiNN']
    gA += -2. * (p['gpiNN']+2.*p['gpiNN']**3) * mpi**2 * np.log(mpi**2 / mu**2) / (4.*np.pi*fpi)**2
    gA += p['cpiNN'] * mpi**2 / (4.*np.pi*fpi)**2
    if e=='continuum':
        pass
    else:
        gA += p['ca'] * p['alat_%s' %e]**2
        #gA += p['cafs'] * p['alpha_%s' %e] * p['alat_%s' %e]**2
    return gA

def select_color(e):
    if e in ['l1648f211b580m013m065m838',   'l2448f211b580m0064m0640m828',  'l3248f211b580m00235m0647m831']:
        return 'red'
    if e in ['l2464f211b600m0102m0509m635', 'l3264f211b600m00507m0507m628', 'l4864f211b600m00184m0507m628']:
        return 'green'
    if e in ['l3296f211b630m0074m037m440', 'l4896f211b630m00363m0363m430']:
        return 'blue'


if __name__=="__main__":
    # set up
    params = dict()
    #params['ens'] = ['l1648f211b580m013m065m838',   'l2448f211b580m0064m0640m828',  'l3248f211b580m00235m0647m831', 'l2464f211b600m0102m0509m635', 'l3264f211b600m00507m0507m628', 'l4864f211b600m00184m0507m628', 'l3296f211b630m0074m037m440', 'l4896f211b630m00363m0363m430']
    params['ens'] = ['l1648f211b580m013m065m838', 'l3248f211b580m00235m0647m831', 'l2464f211b600m0102m0509m635', 'l4864f211b600m00184m0507m628', 'l3296f211b630m0074m037m440', 'l4896f211b630m00363m0363m430']

    # fit with fpi physical not lat
    fpi_phys = True
    fpi_phys = False

    # MILC gradient flow [https://arxiv.org/pdf/1503.02769.pdf]
    # priors just for coding consistency.  They are effectively fixed.
    alpha = dict()
    alpha['l1648f211b580m013m065m838']    = gv.gvar(0.58801, 0.0000001)
    alpha['l2448f211b580m0064m0640m828']  = gv.gvar(0.58801, 0.0000001)
    alpha['l3248f211b580m00235m0647m831'] = gv.gvar(0.58801, 0.0000001)
    alpha['l2464f211b600m0102m0509m635']  = gv.gvar(0.53796, 0.0000001)
    alpha['l3264f211b600m00507m0507m628'] = gv.gvar(0.53796, 0.0000001)
    alpha['l4864f211b600m00184m0507m628'] = gv.gvar(0.53796, 0.0000001)
    alpha['l3296f211b630m0074m037m440']   = gv.gvar(0.43356, 0.0000001)
    alpha['l4896f211b630m00363m0363m430'] = gv.gvar(0.43356, 0.0000001)

    # lattice spacing [https://arxiv.org/pdf/1407.3772.pdf]
    alat = dict()
    alat['l1648f211b580m013m065m838']    = gv.gvar(0.14985,0.00038)
    alat['l2448f211b580m0064m0640m828']  = gv.gvar(0.15303,0.00019)
    alat['l3248f211b580m00235m0647m831'] = gv.gvar(0.15089,0.00017)
    alat['l2464f211b600m0102m0509m635']  = gv.gvar(0.12520,0.00022)
    alat['l3264f211b600m00507m0507m628'] = gv.gvar(0.12307,0.00016)
    alat['l4864f211b600m00184m0507m628'] = gv.gvar(0.12121,0.00010)
    alat['l3296f211b630m0074m037m440']   = gv.gvar(0.09242,0.00021)
    alat['l4896f211b630m00363m0363m430'] = gv.gvar(0.09030,0.00013)

    # fit parameters
    lec = dict()
    lec['gpiNN'] = gv.gvar(1.0, 1.0)
    #lec['gpiDD'] = gv.gvar(0.0, 2.0)
    #lec['DD'] = gv.gvar(0.0, 0.1) #float gpiDD by 10% when gpiDD = -9/5 gpiNN
    #lec['gpiND'] = gv.gvar(0.0, 2.0)
    #lec['ND'] = gv.gvar(0.0, 0.1) #float gpiND by 10% used when gpiND = 6/5 gpiNN
    lec['cpiNN'] = gv.gvar(-10.0, 20.0)
    lec['ca'] = gv.gvar(0.0, 1.0)
    #lec['cafs'] = gv.gvar(0.0, 1.0)

    # inputs and data
    inputs = dict()
    data = []
    for e in params['ens']:
        f = open('./pickle_result/gA_%s.pickle' %e, 'rb')
        dat = gv.gvar(pickle.load(f))
        f.close()
        inputs[e] = gv.gvar(dat[e]['priors'])
        if False: #e == 'l4896f211b630m00363m0363m430':
            hack = dat[e]['data']['gA'].sdev * 0.2
            data.append(gv.gvar(dat[e]['data']['gA'].mean,hack))
        else:
            data.append(dat[e]['data']['gA'])
        print "%s: %s" %(e, str(dat[e]['data']['gA']))
    data = np.array(data)
    #print inputs['l1648f211b580m013m065m838']['mpi']
    #print gv.evalcov(data)

    # make priors
    priors = col.OrderedDict()
    for e in params['ens']:
        if fpi_phys:
            fpi = gv.gvar(130.2, 1.7)*alat[e]/197.326971780039
            inputs[e]['fpi'] = fpi
        priors.update({'alat_%s' %e: alat[e], 'alpha_%s' %e: alpha[e], 'mpi_%s' %e: inputs[e]['mpi'], 'fpi_%s' %e: inputs[e]['fpi']}) #, 'mN_%s' %e: inputs[e]['mN']})
    priors.update(lec)
   
    # it's time to fittttTTTT
    #print gA_chipt(params['ens'],priors)
    fit = lsqfit.nonlinear_fit(data=(params['ens'],data),prior=priors,fcn=gA_chipt,maxit=10000)
    print fit
    
    # make pretty plot #
    mpi = np.linspace(0.01, 350, num=10)
    y = gA_chiral(mpi,fit.p)
    ymean = [i.mean for i in y]
    ypos = [i.mean+i.sdev for i in y]
    yneg = [i.mean-i.sdev for i in y]
    answer = gA_chiral(gv.gvar(139.57018, 0.00035),fit.p)
    
    # physical point error breakdown
    statx = answer.partialsdev(fit.y)/answer.mean*100
    inpux = answer.partialsdev(priors['mpi_l1648f211b580m013m065m838'], priors['mpi_l3296f211b630m0074m037m440'], priors['mpi_l4896f211b630m00363m0363m430'], priors['fpi_l1648f211b580m013m065m838'], priors['fpi_l3296f211b630m0074m037m440'], priors['fpi_l4896f211b630m00363m0363m430'])/answer.mean*100
    #discx = answer.partialsdev(priors['ca'], priors['cafs'])/answer.mean*100
    discx = answer.partialsdev(priors['ca'])/answer.mean*100
    #chirx = answer.partialsdev(priors['cpiNN'], priors['gpiNN'], priors['gpiND'], priors['gpiDD'])/answer.mean*100
    chirx = answer.partialsdev(priors['cpiNN'], priors['gpiNN'])/answer.mean*100
    print "ERROR BREAKDOWN"
    print "Stat:", statx
    print "Inpu:", inpux
    print "Disc:", discx
    print "Chir:", chirx
    print "Total:", np.sqrt(statx**2+inpux**2+discx**2+chirx**2)

    # error breakdown as mpi
    stat = np.array([i.partialsdev(fit.y)/i.mean*100 for i in y])
    inpu = np.array([i.partialsdev(priors['mpi_l1648f211b580m013m065m838'], priors['mpi_l3296f211b630m0074m037m440'], priors['mpi_l4896f211b630m00363m0363m430'], priors['fpi_l1648f211b580m013m065m838'], priors['fpi_l3296f211b630m0074m037m440'], priors['fpi_l4896f211b630m00363m0363m430'])/i.mean*100 for i in y])
    #disc = np.array([i.partialsdev(priors['ca'], priors['cafs'])/i.mean*100 for i in y])
    disc = np.array([i.partialsdev(priors['ca'])/i.mean*100 for i in y])
    chir = np.array([i.partialsdev(priors['cpiNN'], priors['gpiNN'])/i.mean*100 for i in y])

    # matplotlib
    #plt.xkcd()

    # plotting gA chiral-continuum extrapolation    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['right'].set_color('black')
    ax.spines['top'].set_color('black')
    #plt.xticks([])
    #plt.yticks([])
    ax.set_ylim([0.95, 1.45])
    # notes for plot
    textstr = "WITH THE FOLLOWING:\n ONE-LOOP a^2 \n Delta Split\n NOT YET INCLUDED:\n NNLO STUFF"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.05, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='bottom', horizontalalignment='left',bbox=props)
    # misc annotation
    #plt.annotate(
    #    "I'M USING THE REAL\n PHYSICAL DECAY \n CONSTANT HERE SO \nTHIS IS OFFFFFFFF\n (i think that's why)",
    #    xy=(300, 1.225), arrowprops=dict(arrowstyle='->'), xytext=(200, 1.0))
    textstr = "ERROR BUDGET (PCT.):\n STAT: %s \n INPU: %s \n DISC: %s \n CHIR: %s" %(str(round(statx,4)), str(round(inpux,4)), str(round(discx,4)), str(round(chirx,4)))
    props = dict(boxstyle='round', facecolor='blue', alpha=0.15)
    ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='bottom', horizontalalignment='right', bbox=props)
    plt.annotate(
        "I TAKE BREAKS \n %s TIMES \n %s PCT. ERROR" %(str(answer),str(round(answer.sdev/answer.mean*100,2))), 
        xy=(135, 1.27), arrowprops=dict(arrowstyle='->'), xytext=(25, 1.35))
    plt.annotate(
        "NATURE TAKE BREAKS \n %s TIMES" %str(gv.gvar(1.2723,0.0023)),
        xy=(140, 1.27), arrowprops=dict(arrowstyle='->'), xytext=(200, 1.37))
    for i in range(len(params['ens'])):
        e = params['ens'][i]
        mpidat = inputs[e]['mpi']/alat[e]*197.326971780039
        gAdat = data[i]
        print gAdat
        color = select_color(e)
        # plot data
        ax.errorbar(mpidat.mean, gAdat.mean, yerr=gAdat.sdev, color=color, fmt='s', fillstyle='none')
        # plot chipt line
        gAdate = gA_chiral(mpi,fit.p,e)
        gAdate_mean = [i.mean for i in gAdate]
        plt.plot(mpi, gAdate_mean, color=color)
    ax.errorbar(139.57018, 1.2723, yerr=0.0023, color='black', fmt='*', fillstyle='none')
    ax.fill_between(mpi, ypos, yneg, facecolor='cyan', color='cyan', interpolate=True)
    plt.axvline(x=139.57018, linestyle='dashed')
  
    plt.xlabel('Whiskey drank before lunch (mL)')
    plt.ylabel('Bathroom breaks before afternoon tea')
    
    plt.title("I MADE THIS PLOT DRUNK *hICcuP*")

    # plotting error breakdown
    fig = plt.figure()
    
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['right'].set_color('black')
    ax.spines['top'].set_color('black')
    #plt.xticks([])
    #plt.yticks([])
    ax.set_ylim([0, 20])
    # statistical
    ax.fill_between(mpi, 0, stat**2, facecolor='red', color='red', interpolate=True)
    # discretization
    ax.fill_between(mpi, stat**2, stat**2+disc**2, facecolor='blue', color='blue', interpolate=True)
    # input
    ax.fill_between(mpi, stat**2+disc**2, stat**2+disc**2+inpu**2, facecolor='magenta', color='magenta', interpolate=True)
    # chiral
    ax.fill_between(mpi, stat**2+disc**2+inpu**2, stat**2+disc**2+inpu**2+chir**2, facecolor='yellow', color='yellow', interpolate=True)

    plt.axvline(x=139.57018, linestyle='dashed')
    
    textstr = "TOP TO BOTTOM:\n (YELLO) CHIRAL\n (MAGEN) INPUT\n (BLOOO) DISC.\n (REDDD) STAT"
    props = dict(boxstyle='round', facecolor='purple', alpha=0.2)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
    verticalalignment='top', horizontalalignment='left',bbox=props)

    plt.xlabel('Whiskey drank before lunch (mL)')
    plt.ylabel('Error breakdown in quadrature')

    plt.title("I MADE THIS PLOT DRUNK *hICcuP*")

    plt.show()
