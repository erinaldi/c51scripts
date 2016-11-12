# do contractions with python
# takes in propagator from h5

import numpy as np
import h5py as h5
import spin_stuff as ss
import time

def levi():
    eijk = np.zeros((3, 3, 3))
    eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
    eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1
    return eijk

def Gamma_Proj(basak):
    bterms = dict()
    # basak 1
    bterms['G1g1u'] = [1,2,1,1]
    bterms['G1g1d'] = [1,2,2,1]
    bterms['G1u1u'] = [3,4,3,1]
    bterms['G1u1d'] = [3,4,4,1]
    # basak 2
    bterms['G1g2u_1'] = [1,4,3,1]
    bterms['G1g2u_2'] = [3,2,3,1]
    bterms['G1g2u_3'] = [3,4,1,1]
    bterms['G1g2d_1'] = [1,4,4,1]
    bterms['G1g2d_2'] = [3,2,4,1]
    bterms['G1g2d_3'] = [3,4,2,1]
    bterms['G1u2u_1'] = [1,2,3,1]
    bterms['G1u2u_2'] = [1,4,1,1]
    bterms['G1u2u_3'] = [3,2,1,1]
    bterms['G1u2d_1'] = [1,2,4,1]
    bterms['G1u2d_2'] = [1,4,2,1]
    bterms['G1u2d_3'] = [3,2,2,1]
    # basak 3
    bterms['G1g3u_1'] = [1,3,4,1]
    bterms['G1g3u_2'] = [3,2,3,1]
    bterms['G1g3u_3'] = [3,4,1,-1]
    bterms['G1g3d_1'] = [1,4,4,1]
    bterms['G1g3d_2'] = [4,2,3,1]
    bterms['G1g3d_3'] = [3,4,2,-1]
    bterms['G1u3u_1'] = [1,4,1,-1]
    bterms['G1u3u_2'] = [3,1,2,-1]
    bterms['G1u3u_3'] = [1,2,3,1]
    bterms['G1u3d_1'] = [3,2,2,-1]
    bterms['G1u3d_2'] = [2,4,1,-1]
    bterms['G1u3d_3'] = [1,2,4,-1]
    g = Gamma(bterms[basak])
    p = Proj(bterms[basak])
    u, ud = ss.basis_transform()
    up = np.einsum('ij,j->i', u, p)
    gud = np.einsum('ij,jk->ik', g, ud)
    ugud = np.einsum('ij,jk->ik', u, gud)
    return ugud, up

def Proj(bterms): #, bterms_transition):
    p = np.zeros((4))
    i = bterms[2]-1
    #j = bterms_transition[2]-1
    phase = bterms[3]
    p[i] = phase
    return p

def Gamma(bterms):
    g = np.zeros((4,4))
    i = bterms[0]-1
    j = bterms[1]-1
    g[i,j] = 1
    g[j,i] = -1
    return g

def read_prop(filename,datapath):
    f = h5.File('/Users/cchang5/Physics/c51/data/'+filename,'r')
    data = f[datapath][()]
    #data = data[:,:,src,snk] #00 is particle particle smear smear
    #f.close()
    return data

def contract(prop,P,Q,G,H,origin):
    # 1/2 e_efg e_abc Q_s P_l H_qr G_ij [ U^ec_ql D^fb_rj U^ga_si - U^ea_qi D^fb_rj U^gc_sl ]
    # term 1
    # spin contract
    UP = np.einsum('txyzqlec,l->txyzqec', prop, P)
    UPQ = np.einsum('txyzqec,s->txyzqsec', UP, Q)
    UG = np.einsum('txyzsiga,ij->txyzsjga', prop, G)
    DH = np.einsum('txyzrjfb,qr->txyzqjfb', prop, H)
    UGDH = np.einsum('txyzsjga,txyzqjfb->txyzsqgafb', UG, DH)
    UGDHUPQ = np.einsum('txyzsqgafb,txyzqsec->txyzgafbec', UGDH, UPQ)
    # color contract
    UGDHUPQ = np.einsum('abc,txyzgafbec->txyzgfe', levi(), UGDHUPQ)
    UGDHUPQ = np.einsum('efg,txyzgfe->txyz', levi(), UGDHUPQ)
    UGDHUPQ = 0.5*UGDHUPQ
    # momentum projection
    term1 = UGDHUPQ.sum(axis=3).sum(axis=2).sum(axis=1)
    # term 2
    # spin contract
    UG = np.einsum('txyzqiea,ij->txyzqjea', prop, G)
    DH = np.einsum('txyzrjfb,qr->txyzqjfb', prop, H)
    UP = np.einsum('txyzslgc,l->txyzsgc', prop, P)
    UPQ = np.einsum('txyzsgc,s->txyzgc', UP, Q)
    UGDH = np.einsum('txyzqjea,txyzqjfb->txyzeafb', UG, DH)
    # color contract
    UPQ = np.einsum('abc,txyzgc->txyzabg', levi(), UPQ)
    UPQ = np.einsum('efg,txyzabg->txyzabef', levi(), UPQ)
    UPQUGDH = np.einsum('txyzabef,txyzeafb->txyz', UPQ, UGDH)
    UPQUGDH = 0.5*UPQUGDH
    # momentum projection
    term2 = UPQUGDH.sum(axis=3).sum(axis=2).sum(axis=1)
    # difference
    corr = term1-term2
    corr[:origin[-1]] = -1*corr[:origin[-1]]
    corr = np.roll(corr,-1*origin[-1])
    return corr

def fhcontractU(prop,fhprop,P,Q,G,H,origin):
    # F^ab_ij = U^ad_in J_nm U^db_mj
    # corr = 1/2 e_efg e_abc Q_s P_l * [
    #      + F^ec_ql U^ga_si G_ij D^fb_rj H_qr
    #      - F^ea_qi G_ij D^fb_rj H_qr U^gc_sl
    #      - U^ea_qi G_ij D^fb_rj H_qr F^gc_sl
    #      + U^ec_ql F^ga_si G_ij D^fb_rj H_qr
    #      + two disconnected diagrams]
    # term 1
    # spin contract
    FP = np.einsum('txyzqlec,l->txyzqec', fhprop, P)
    FPQ = np.einsum('txyzqec,s->txyzqsec', FP, Q)
    FPQH = np.einsum('txyzqsec,qr->txyzrsec', FPQ, H)
    UG = np.einsum('txyzsiga,ij->txyzsjga', prop, G)
    UGD = np.einsum('txyzsjga,txyzrjfb->txyzsrgafb', UG, prop)
    UGDFPQH = np.einsum('txyzsrgafb,txyzrsec->txyzgafbec', UGD, FPQH)
    # color contract
    UGDFPQH = np.einsum('abc,txyzgafbec->txyzgfe', levi(), UGDFPQH)
    UGDFPQH = np.einsum('efg,txyzgfe->txyz', levi(), UGDFPQH)
    UGDFPQH = 0.5*UGDFPQH
    # momentum projection
    term1 = UGDFPQH.sum(axis=3).sum(axis=2).sum(axis=1)
    # term 2
    # spin contract
    FG = np.einsum('txyzqiea,ij->txyzqjea', fhprop, G)
    FGH = np.einsum('txyzqjea,qr->txyzrjea', FG, H)
    FGHD = np.einsum('txyzrjea,txyzrjfb->txyzeafb', FGH, prop)
    UQ = np.einsum('txyzslgc,s->txyzlgc', prop, Q)
    UQP = np.einsum('txyzlgc,l->txyzgc', UQ, P)
    FGHDUQP = np.einsum('txyzeafb,txyzgc->txyzeafbgc', FGHD, UQP)
    # color contract
    FGHDUQP = np.einsum('abc,txyzeafbgc->txyzefg', levi(), FGHDUQP)
    FGHDUQP = np.einsum('efg,txyzefg->txyz', levi(), FGHDUQP)
    FGHDUQP = -0.5*FGHDUQP
    # momentum projection
    term2 = FGHDUQP.sum(axis=3).sum(axis=2).sum(axis=1)
    # term 3
    # spin contract
    UG = np.einsum('txyzqiea,ij->txyzqjea', prop, G)
    UGH = np.einsum('txyzqjea,qr->txyzrjea', UG, H)
    UGHD = np.einsum('txyzrjea,txyzrjfb->txyzeafb', UGH, prop)
    FQ = np.einsum('txyzslgc,s->txyzlgc', fhprop, Q)
    FQP = np.einsum('txyzlgc,l->txyzgc', FQ, P)
    UGHDFQP = np.einsum('txyzeafb,txyzgc->txyzeafbgc', UGHD, FQP)
    # color contract
    UGHDFQP = np.einsum('abc,txyzeafbgc->txyzefg', levi(), UGHDFQP)
    UGHDFQP = np.einsum('efg,txyzefg->txyz', levi(), UGHDFQP)
    UGHDFQP = -0.5*UGHDFQP
    # momentum projection
    term3 = UGHDFQP.sum(axis=3).sum(axis=2).sum(axis=1)
    # term 4
    # spin contract
    UH = np.einsum('txyzqlec,qr->txyzrlec', prop, H)
    UHP = np.einsum('txyzrlec,l->txyzrec', UH, P)
    UHPQ = np.einsum('txyzrec,s->txyzrsec', UHP, Q)
    DG = np.einsum('txyzrjfb,ij->txyzrifb', prop, G)
    DGF = np.einsum('txyzrifb,txyzsiga->txyzrsfbga', DG, fhprop)
    UHPQDGF = np.einsum('txyzrsec,txyzrsfbga->txyzecfbga', UHPQ, DGF)
    # color contract
    UHPQDGF = np.einsum('abc,txyzecfbga->txyzefg', levi(), UHPQDGF)
    UHPQDGF = np.einsum('efg,txyzefg->txyz', levi(), UHPQDGF)
    UHPQDGF = 0.5*UHPQDGF
    # momentum projection
    term4 = UHPQDGF.sum(axis=3).sum(axis=2).sum(axis=1)
    # combine
    fhcorr = term1+term2+term3+term4
    fhcorr[:origin[-1]] = -1*fhcorr[:origin[-1]]
    fhcorr = np.roll(fhcorr,-1*origin[-1])
    return fhcorr

def fhcontractD(prop,fhprop,P,Q,G,H,origin):
    # F^ab_ij = D^ad_in J_nm D^db_mj
    # corr = 1/2 e_efg e_abc H_qr Q_s G_ij P_l * [
    #      - U^ea_qi U^gc_sl F^fb_rj
    #      + U^ga_si U^ec_ql F^fb_rj
    #      + two disconnected diagrams]
    # term 1
    # spin contract
    UG = np.einsum('txyzqiea,ij->txyzqjea', prop, G)
    UGH = np.einsum('txyzqjea,qr->txyzrjea', UG, H)
    UGHF = np.einsum('txyzrjea,txyzrjfb->txyzeafb', UGH, fhprop)
    UQ = np.einsum('txyzslgc,s->txyzlgc', prop, Q)
    UQP = np.einsum('txyzlgc,l->txyzgc', UQ, P)
    UGFHUQP = np.einsum('txyzeafb,txyzgc->txyzeafbgc', UGHF, UQP)
    # color contract
    UGFHUQP = np.einsum('abc,txyzeafbgc->txyzefg', levi(), UGFHUQP)
    UGFHUQP = np.einsum('efg,txyzefg->txyz', levi(), UGFHUQP)
    UGFHUQP = -0.5*UGFHUQP
    # momentum projection
    term1 = UGFHUQP.sum(axis=3).sum(axis=2).sum(axis=1)
    # term 2
    # spin contract
    UG = np.einsum('txyzsiga,ij->txyzsjga', prop, G)
    UGQ = np.einsum('txyzsjga,s->txyzjga', UG, Q)
    UGQP = np.einsum('txyzjga,l->txyzljga', UGQ, P)
    UH = np.einsum('txyzqlec,qr->txyzrlec', prop, H)
    UHF = np.einsum('txyzrlec,txyzrjfb->txyzljecfb', UH, fhprop)
    UHFUGQP = np.einsum('txyzljecfb,txyzljga->txyzecfbga', UHF, UGQP)
    # color contract
    UHFUGQP = np.einsum('abc,txyzecfbga->txyzefg', levi(), UHFUGQP)
    UHFUGQP = np.einsum('efg,txyzefg->txyz', levi(), UHFUGQP)
    UHFUGQP = 0.5*UHFUGQP
    # momentum projection
    term2 = UHFUGQP.sum(axis=3).sum(axis=2).sum(axis=1)
    # combine
    fhcorr = term1+term2
    fhcorr[:origin[-1]] = -1*fhcorr[:origin[-1]]
    fhcorr = np.roll(fhcorr,-1*origin[-1])
    return fhcorr

if __name__=='__main__':
    # flags
    twopt = True
    threeptU = True
    threeptD = True
    # read propagator
    #filename = 'prop_mq0.0158_wflow1.0_M51.3_L512_a2.0_cfg_200_srcx0y13z5t27_wv_w4.5_n60.hdf5'
    filename = 'prop_mq0.0158_wflow1.0_M51.3_L512_a2.0_cfg_200_srcx1y13z4t19_wv_w4.5_n60.hdf5'
    datapath = '/propagator'
    prop = read_prop(filename,datapath)
    # read fh propagator
    filename = 'seqprop_0.0158_A3_0.0158_wflow1.0_M51.3_L512_a2.0_cfg_200_srcx1y13z4t19_wv_w4.5_n60.hdf5'
    datapath = '/propagator'
    fhprop = read_prop(filename,datapath)
    #origin = [0,12,5,27]
    origin = [1,13,4,19]
    print "len(prop)  :", np.shape(prop)
    print "len(fhprop):", np.shape(fhprop)
    src = ['G1u3u_1','G1u3u_2','G1u3u_3']
    snk = ['G1u3u_1','G1u3u_2','G1u3u_3']
    #src = ['G1u2u_1','G1u2u_2','G1u2u_3']
    #snk = ['G1u2u_1','G1u2u_2','G1u2u_3']
    #src = ['G1u1u']
    #snk = ['G1u1u']
    # tk read 
    tkname = 'G1u3u_G1u2u_SS'
    #tkname = 'G1u1u_G1u1u'
    # kno read
    knospin = 'up'
    if twopt:
        # make two point
        corr = []
        for i in src:
            for j in snk:
                G, P = Gamma_Proj(i)
                H, Q = Gamma_Proj(j)
                #corr.append(contract(prop,P,Q,G,H,origin).real)
        #corr = np.array(corr).sum(axis=0)/np.sqrt(len(src)*len(snk))
        # read KNO
        filename = 'l1648f211b580m013m065m838a_200.h5'
        #datapath = '/wf1p0_m51p3_l512_a52p0_smrw4p5_n60/spectrum/ml0p0158_ms0p0902/proton/spin_%s/x1y13z4t19' %knospin
        datapath = '/wf1p0_m51p3_l512_a52p0_smrw4p5_n60/spectrum/ml0p0158_ms0p0902/proton_np/spin_%s/x1y13z4t19' %knospin
        #corrkno = read_prop(filename,datapath)[:,4,1]
        corrkno = read_prop(filename,datapath)[:,2,1]
        #corrkno = read_prop(filename,datapath)[:,3,0]
        # read TK
        #datapath = '/two_pt/%s' %tkname
        # with t-shift and smearing
        filename = 'test.16c48_x1y13z4t19.h5'
        #filename = 'test.16c48_x1y13z4t19.manualgeom.h5'
        datapath = '/two_pt/%s' %tkname
        corrtk = read_prop(filename,datapath)
        #corrtk[:origin[-1]] = -1*corrtk[:origin[-1]]
        #corrtk = np.roll(corrtk,-1*origin[-1])
        print "two point"
        #print corr/corrkno
        #print corr/corrtk.real
        print corrtk.real/corrkno
    if threeptU:
        # make three point
        fhcorr = []
        for i in src:
            for j in snk:
                G, P = Gamma_Proj(i)
                H, Q = Gamma_Proj(j)
                #fhcorr.append(fhcontractU(prop,fhprop,P,Q,G,H,origin).imag)
        #fhcorr = np.array(fhcorr).sum(axis=0)/np.sqrt(len(src)*len(snk))
        # read KNO
        filename = 'l1648f211b580m013m065m838a_200.h5'
        #datapath = '/wf1p0_m51p3_l512_a52p0_smrw4p5_n60/spectrum/ml0p0158_ms0p0902/proton/A3_UU_spin_%s/x1y13z4t19' %knospin
        datapath = '/wf1p0_m51p3_l512_a52p0_smrw4p5_n60/spectrum/ml0p0158_ms0p0902/proton_np/A3_UU_spin_%s/x1y13z4t19' %knospin
        #fhcorrkno = read_prop(filename,datapath)[:,4,1]
        fhcorrkno = read_prop(filename,datapath)[:,2,1]
        #fhcorrkno = read_prop(filename,datapath)[:,3,0]
        # read TK
        filename = 'test.16c48_x1y13z4t19.h5'
        #filename = 'test.16c48_x1y13z4t19.manualgeom.h5'
        datapath = '/fh_A3_UU/%s' %tkname
        fhcorrtk = read_prop(filename,datapath)
        #fhcorrtk[:origin[-1]] = -1*fhcorrtk[:origin[-1]]
        #fhcorrtk = np.roll(fhcorrtk,-1*origin[-1])
        filename = 'test.16c48_x1y13z4t19.manualgeom.h5'
        datapath = '/fh_A3_UU/%s' %tkname
        fhcorrtk2 = read_prop(filename,datapath)
        filename = 'test.16c48_x1y13z4t19.nogeom.h5'
        datapath = '/fh_A3_UU/%s' %tkname
        fhcorrtk3 = read_prop(filename,datapath)
        print "FU"
        #print fhcorr/fhcorrkno
        #print fhcorr/fhcorrtk.imag
        print fhcorrtk.imag/fhcorrkno
        #print fhcorrtk2/fhcorrtk
        #print fhcorrtk3/fhcorrtk
    if threeptD:
        # make three point
        fhcorr = []
        for i in src:
            for j in snk:
                G, P = Gamma_Proj(i)
                H, Q = Gamma_Proj(j)
                #fhcorr.append(fhcontractD(prop,fhprop,P,Q,G,H,origin).imag)
        #fhcorr = np.array(fhcorr).sum(axis=0)/np.sqrt(len(src)*len(snk))
        # read KNO
        filename = 'l1648f211b580m013m065m838a_200.h5'
        #datapath = '/wf1p0_m51p3_l512_a52p0_smrw4p5_n60/spectrum/ml0p0158_ms0p0902/proton/A3_DD_spin_%s/x1y13z4t19' %knospin
        datapath = '/wf1p0_m51p3_l512_a52p0_smrw4p5_n60/spectrum/ml0p0158_ms0p0902/proton_np/A3_DD_spin_%s/x1y13z4t19' %knospin
        #fhcorrkno = read_prop(filename,datapath)[:,4,1]
        fhcorrkno = read_prop(filename,datapath)[:,2,1]
        #fhcorrkno = read_prop(filename,datapath)[:,3,0]
        # read TK
        filename = 'test.16c48_x1y13z4t19.h5'
        #filename = 'test.16c48_x1y13z4t19.manualgeom.h5'
        datapath = '/fh_A3_DD/%s' %tkname
        fhcorrtk = read_prop(filename,datapath)
        #fhcorrtk[:origin[-1]] = -1*fhcorrtk[:origin[-1]]
        #fhcorrtk = np.roll(fhcorrtk,-1*origin[-1])
        print "FD"
        #print fhcorr/fhcorrkno
        #print fhcorr/fhcorrtk.imag
        print fhcorrtk.imag/fhcorrkno
