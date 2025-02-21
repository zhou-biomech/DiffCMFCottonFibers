import numpy as np

def vecrndrot(v, sigma, n=[]):
    # 对向量 v，随机旋转角度，角度服从均值为0,方差为sigma的正态分布
    if not n: n = np.shape(v)[0]

    #rnd = np.random.normal(0,sigma,n)* np.pi/180
    rnd = (2*np.random.rand(n)-1) * sigma * np.pi/180
    
    vx = v[:,0]*np.cos(rnd) - v[:,1]*np.sin(rnd)
    vy = v[:,0]*np.sin(rnd) + v[:,1]*np.cos(rnd)
    
    return np.c_[vx, vy]

# -----------------------------------------------------------------------------

if __name__=='__main__':
    from matplotlib import pyplot as plt

    n = 5
    x = np.arange(n)
    y = np.zeros_like(x)

    ''' # case 1: 一个向量，n 次旋转 10 度以内的随机角度生成 n 个向量
    v = np.array([[0,1]])
    vr = vecrndrot(v, 10, n)
    '''
    
    # case 2: 多个向量，各自旋转 10 度以内的随机角度
    v = np.array([[0,1],[0,2],[2,1],[0,1],[0,1]])
    vr = vecrndrot(v, 10)
    
    '''
    # case 3: 一个向量，各自旋转不同度以内的随机角度
    v = np.array([[0,1]])
    ang = np.array([0,20,40,60,80])
    vr = vecrndrot(v, ang, 5)
    '''
    
    dr = 1
    plt.axis('scaled')
    plt.axis([-1,5,-1,2])
    for i in range(n):
        plt.plot([x[i], x[i]+dr*vr[i,0]],  [y[i], y[i]+dr*vr[i,1]], '-')

    plt.show()
