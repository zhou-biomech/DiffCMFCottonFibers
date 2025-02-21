import numpy as np
from matplotlib import pyplot as plt
from Wall import Wall

xmax = 5
dx = 0.5
W = 1
km = 1*W

pdf = lambda xmax,x: 1.25-x/xmax
mf = 0.35
kf = 1/10
rho = mf/1.65/(1-mf)*700/kf
deg = 20

m = int(xmax/dx)
nsim = 10
L = np.zeros((nsim,m))
xshift = 7 # (7-5)/5 = 40%

F = np.zeros(nsim)
for i in range(nsim):
    print(i)
    np.random.seed(i)
    w = Wall(xmax, dx, W, km, rho, pdf, deg, kf)
    w.shiftxmax(xshift)
    L[i] = np.diff(w.x)
    F[i] = w.f[-1]
    
r = np.linspace(dx/2, xmax-dx/2, m)/xmax
plt.figure()

e = (L/dx-1)/(xshift/xmax-1)
em = np.mean(e,axis=0)
es = np.std(e,axis=0)
for i in range(len(r)):
    print('%4.2f %8.6f %8.6f'%(r[i],em[i],es[i]))
    
plt.errorbar(r, em, es)
plt.xlabel('x/L')
plt.ylabel('ei/e')
print('F = ', np.mean(F))
plt.show()
