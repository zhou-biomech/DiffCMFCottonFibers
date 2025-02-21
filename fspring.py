import numpy as np
from scipy.optimize import minimize

def fspring(p, ln, k, l0):
    '''
    FSPRING energy and forces on vertexes of the spring.
       The potential energy is defined as
 
                        E = k/2 * l0 * (l/l0 - 1)^2

       The above potential energy yield the following forces

                 k
          fi = ------ * (l - l0) * ( xj-xi,  yj-yi ),     fj =  -fi,
                l*l0

       where j = i+1. l and l0 are the actual and rest length of the spring. k
       is the spring constant per unit length.

    Zhou Lvwen: zhoulvwen@nbu.edu.cn, November 28, 2023
    '''
    
    i, j= ln[:,0], ln[:,1]
    eji = p[j,:] - p[i,:]

    l0 = l0.reshape(-1,1) if isinstance(l0,np.ndarray) else l0
    k  =  k.reshape(-1,1) if isinstance( k,np.ndarray) else k

    l = np.hypot(eji[:,0], eji[:,1]).reshape(-1,1)

    E = 1/2*k/l0*(l-l0)**2

    fji = k/l0 * (l-l0) * eji/l
    
    f = np.zeros_like(p,dtype=float)

    for k in range(len(ln)):
        f[i[k]] +=  fji[k]
        f[j[k]] += -fji[k]
    
    return E, f

# -----------------------------------------------------------------------------

def fspring2(p, ni, nj, ri, rj, k, l0):
    f = np.zeros_like(p,dtype=float)
    E = 0

    if len(nj)==0:
        return E, f
    
    pi = p[ni[:,0]]*(1-ri) + p[ni[:,1]]*ri
    pj = p[nj[:,0]]*(1-rj) + p[nj[:,1]]*rj

    eji = pj - pi
    l = np.hypot(eji[:,0], eji[:,1]).reshape(-1,1)

    E = 1/2*k/l0*(l-l0)**2
    fji = k/l0 * (l-l0) * eji/l

    f = np.zeros_like(p,dtype=float)
    
    for k in range(len(l0)):
        f[ni[k,0]] +=  fji[k]*(1-ri[k])
        f[ni[k,1]] +=  fji[k]*ri[k]
    
        f[nj[k,0]] += -fji[k]*(1-rj[k])
        f[nj[k,1]] += -fji[k]*rj[k]
        
    return E, f

# -----------------------------------------------------------------------------

def fun(x, ln, k, l0):

    p = x.reshape(-1,2)
    E, f = fspring(p, ln, k, l0)
    f = -f.flatten()
    
    return np.sum(E), f

# -----------------------------------------------------------------------------
if __name__=='__main__':
    import matplotlib.pyplot as plt
    # np.random.seed(0)
    k, l0 = 1, 1
    p0 = 2*np.random.rand(4,2)-1
    ln = np.array([[0,1],[1,2],[2,3], [3,0], [3,1]])
    plt.plot(p0[ln.T,0], p0[ln.T,1], 'o--c')

    # -------- minimize -------------------------------------------------------
    x0 = p0.flatten()
    res = minimize(fun, x0, args=(ln, k, l0), jac=True, method='SLSQP')
    p1 = res.x.reshape(-1,2)
    plt.plot(p1[ln.T,0], p1[ln.T,1], 'o-r')
    plt.axis('scaled')
    plt.axis([-2,2,-2,2])
    
    # -------- p = p + f*dt ---------------------------------------------------
    dt = 1e-2
    p2 = p0.copy()
    x = np.r_[p2[ln.T,0], np.nan*np.ones((1,5))].T.flatten()
    y = np.r_[p2[ln.T,1], np.nan*np.ones((1,5))].T.flatten()
    l, = plt.plot(x, y, 'o-b')
    E, f = fspring(p2, ln, k, l0)
    while np.sum(E)>1e-6:
        p2 = p2 + f*dt
        x = np.r_[p2[ln.T,0], np.nan*np.ones((1,5))].T.flatten()
        y = np.r_[p2[ln.T,1], np.nan*np.ones((1,5))].T.flatten()
        l.set_data(x, y)
        plt.pause(1e-3)
        E, f = fspring(p2, ln, k, l0)
