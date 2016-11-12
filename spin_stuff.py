import numpy as np
np.set_printoptions(linewidth=180)

def basis_transform(verbose=False):
    ns = 4
    Gdp_1 = np.zeros([ns,ns],dtype=np.complex128)
    Gdp_2 = np.zeros([ns,ns],dtype=np.complex128)
    Gdp_3 = np.zeros([ns,ns],dtype=np.complex128)
    Gdp_4 = np.zeros([ns,ns],dtype=np.complex128)
    Gdp_5 = np.zeros([ns,ns],dtype=np.complex128)
    
    # fill in non zero values of gamma matrices in DiracPauli convention
    # g_i = ((0, -i pauli_i), (i pauli_i, 0))
    # g_4 = ((1, 0), (0, -1))
    # g_5 = ((0, 1), (1, 0))
    # g_1
    Gdp_1[0,3] = -1.j
    Gdp_1[1,2] = -1.j
    Gdp_1[2,1] = 1.j
    Gdp_1[3,0] = 1.j
    # g_2
    Gdp_2[0,3] = -1.
    Gdp_2[1,2] = 1.
    Gdp_2[2,1] = 1.
    Gdp_2[3,0] = -1.
    # g_3
    Gdp_3[0,2] = -1.j
    Gdp_3[1,3] = 1.j
    Gdp_3[2,0] = 1.j
    Gdp_3[3,1] = -1.j
    # g_4
    Gdp_4[0,0] = 1.
    Gdp_4[1,1] = 1.
    Gdp_4[2,2] = -1.
    Gdp_4[3,3] = -1.
    # g_5
    Gdp_5[0,2] = 1.
    Gdp_5[1,3] = 1.
    Gdp_5[2,0] = 1.
    Gdp_5[3,1] = 1.
    
    U = 1./np.sqrt(2) * ( -1.j*Gdp_2 + 1.j*np.dot(Gdp_1,Gdp_3))
    Ud = np.conj(np.swapaxes(U,0,1))
    if verbose:
        for i in range(ns):
            print(U[i])
        for i in range(ns):
            print(Ud[i])
        for i in range(ns):
            print(np.matrix(U).getH()[i])
        for i in range(ns):
            print(np.dot(U,np.linalg.inv(U))[i])
    return U, Ud

if __name__=='__main__':
    basis_transform(True) 
