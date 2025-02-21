import numpy as np

# 求多条线段 xv, yv 与多条线段 xs, ys 的交点及相交的位置

def intersect(xv, yv, xs, ys):

    xv, yv = np.array(xv), np.array(yv)
    xs, ys = np.array(xs), np.array(ys)
    
    # 将 xv, yv 拆解成多条线段 (x1, y1)-(x2,y2) 
    x1, y1 = xv[:,0].reshape(1,-1), yv[:,0].reshape(1,-1)
    x2, y2 = xv[:,1].reshape(1,-1), yv[:,1].reshape(1,-1)
    
    # 将 xs, ys 拆解成多条线段 (x3, y3)-(x4, y4)
    x3, y3 = xs[:,0].reshape(-1,1), ys[:,0].reshape(-1,1)
    x4, y4 = xs[:,1].reshape(-1,1), ys[:,1].reshape(-1,1)

    
    de =  (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
    
    dez = de.copy()
    dez[de==0] = dez[de==0] + 1
    
    ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / dez
    ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / dez
    
    flag = (de != 0) & ( (0<ua)&(ua<1) ) & ( (0<ub)&(ub<1) )

    # 找到相交的线段 inds 和 indv 及交点 x, y
    inds, indv = np.where(flag)
    
    x = x1 + ua * (x2-x1)
    y = y1 + ua * (y2-y1)

    return indv, inds, x[flag], y[flag]

# -----------------------------------------------------------------------------

if __name__=='__main__':
    import matplotlib.pyplot as plt
    import time
    
    np.random.seed(2)
    xmax = 5
    dx = 0.5
    dy = 2

    plt.figure(figsize=(6,5))
    plt.axis('scaled')
    plt.axis([-0.5,xmax+0.5,-1.5*dy/2,1.5*dy/2])
    
    n = int(np.ceil(xmax/dx))
    x1 = np.linspace(0, xmax, n+1)
    y1 = dy/2 * np.ones_like(x1)

    xv1 = np.c_[x1[0:-1], x1[1:]]
    yv1 = np.c_[y1[0:-1], y1[1:]]

    xv = np.r_[xv1,  np.flip(xv1)]
    yv = np.r_[yv1, -np.flip(yv1)]
    
    plt.plot(xv.T, yv.T, 'o-b')

    n = 10
    
    xs = xmax*np.random.rand(n,2)
    ys = np.array([-2,1]).reshape((1,2))+ np.random.rand(n,2)

    
    start = time.process_time()
    [indv, inds, x, y] = intersect(xv, yv, xs, ys)
    print("求交点用时：%5.4e s"%(time.process_time() - start))

    plt.plot(xs.T, ys.T, '-')
    plt.plot(x, y, '.r')
    plt.pause(1)
