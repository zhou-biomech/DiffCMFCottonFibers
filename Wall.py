import numpy as np
from matplotlib import pyplot as plt
from intersect import intersect
from vecrndrot import vecrndrot
from rndfrompdf import rndfrompdf
from scipy.optimize import minimize
from fspring import fspring, fspring2
from scipy.interpolate import interp1d

# -----------------------------------------------------------------------------
# 边界上的顶点
class Node:
    def __init__(self, x, y, xpin=False, ypin=False):
        self.x, self.y = x, y
        self.ispin = (xpin, ypin)
        self.bounds = [(x, x) if xpin else (None, None), (y,y) if ypin else (None, None)]
        self.edges = []
        
    def plot(self, s='ob'):
        try:
            self.h.set_data(self.x, self.y)
        except:
            self.h, = plt.plot(self.x, self.y, s)

    def __repr__(self):
        return f"Node({self.x:.4f}, {self.y:.4f})"

# -----------------------------------------------------------------------------
# 主边
class Edge:
    def __init__(self, nodes, k, strain=0):
        self.nodes = nodes
        self.k = k
        self.l0 = self.l()/(1+strain)

    def x(self): # 返回 Link 两端点的坐标
        return [self.nodes[0].x, self.nodes[1].x]
    def y(self):
        return [self.nodes[0].y, self.nodes[1].y]

    def mx(self): # 返回 Link 两端点的中心 (mx, my)
        return (self.nodes[1].x + self.nodes[0].x)/2
    def my(self):
        return (self.nodes[1].y + self.nodes[0].y)/2
    
    def dx(self): # 返回 Link 两端点的 dx, dy
        return self.nodes[1].x - self.nodes[0].x
    def dy(self):
        return self.nodes[1].y - self.nodes[0].y

    def l(self): # 返回 Link 的长度
        return (self.dx()**2 + self.dy()**2)**0.5

    def plot(self, s='.-'):
        try:
            self.h.set_data(self.x(), self.y())
        except:
            self.h, = plt.plot(self.x(), self.y(), s, alpha=0.2)
            
    def __repr__(self):
        return f"Edge({self.nodes[0]}, {self.nodes[1]})"

# -----------------------------------------------------------------------------
# 由edges[0]上两个节点按比列p[0]构成点，
# 由edges[1]上两个节点按比列p[1]构成点，组成一个Link
class Link:
    def __init__(self, edges, prop, k, strain=0):
        self.edges = edges
        self.prop = prop
        self.k = k
        self.l0 = self.l()/(1+strain)

    def x(self):
        p = self.prop
        x0 = self.edges[0].nodes[0].x * p[0] + \
             self.edges[0].nodes[1].x * (1-p[0])
        
        x1 = self.edges[1].nodes[0].x * p[1] + \
             self.edges[1].nodes[1].x * (1-p[1])

        return [x0, x1]

    def y(self):
        p = self.prop
        y0 = self.edges[0].nodes[0].y * p[0] + \
             self.edges[0].nodes[1].y * (1-p[0])

        y1 = self.edges[1].nodes[0].y * p[1] + \
             self.edges[1].nodes[1].y * (1-p[1])
        return [y0, y1]

    def dx(self):
        x = self.x()
        return x[1] - x[0]
    
    def dy(self):
        y = self.y()
        return y[1] - y[0]

    def l(self): # 返回 Link 的长度
        return (self.dx()**2 + self.dy()**2)**0.5

    def plot(self, s='.-'):
        try:
            self.h.set_data(self.x(), self.y())
        except:
            self.h, = plt.plot(self.x(), self.y(), s)
            
    def __repr__(self):
        return f"Link({self.edges[0]}, {self.edges[1]})"

# -----------------------------------------------------------------------------

class Wall:
    def __init__(self, xmax, dx, dy, kx, rho, pdf, deg, ky):
        self.xmax = xmax
        self.dx = dx
        self.dy = dy
        self.kx = kx

        self.rho = rho
        self.pdf = pdf
        self.deg = deg
        self.ky = ky
        
        n = int(np.ceil(xmax/dx))
        x = np.linspace(0, xmax, n+1)
        
        nodes = [Node(x[i],    dy/2, i==0, True) for i in range(n+1)] + \
                [Node(x[n-i], -dy/2, i==n, True) for i in range(n+1)]
        
        edges = [Edge(nodes[i:i+2], k = kx/2) for i in range(n)] + \
                [Edge(nodes[i:i+2], k = kx/2) for i in range(n+1,2*n+1)] + \
                [Edge([nodes[i], nodes[2*n+1-i]], k = 0) for i in [0, n]]

        self.x = x
        self.nodes = nodes
        self.edges = edges
        self.addlnks()
    
    def addlnks(self):
        edges, xmax, rho, pdf = self.edges, self.xmax, self.rho, self.pdf
        
        xe = [e.x() for e in edges]
        ye = [e.y() for e in edges]
        
        ns = int(rho*(xmax+2*self.dy))
        xc = np.sort(rndfrompdf(lambda x: pdf(self.xmax,x), xmax,ns, self.dy))
        yc = self.dy * (np.random.rand(ns)-0.5)

          
        vr = vecrndrot(np.array([[0,1]]), self.deg, ns)
        
        ''' # 角度线性递减的情况

        ddeg = self.deg
        deg = interp1d([0-self.dy,0,xmax,xmax+self.dy],[20+ddeg/2,20+ddeg/2, 20-ddeg/2,20-ddeg/2])
        vr = vecrndrot(np.array([[0,1]]), deg(xc), ns)
        '''
        
        dr = 5*self.dy
        xs = np.c_[xc+dr*vr[:,0], xc-dr*vr[:,0]]
        ys = np.c_[yc+dr*vr[:,1], yc-dr*vr[:,1]]

        [inde, inds, x, y] = intersect(xe, ye, xs, ys)
        
        links, lnkid = [], []
        for i in range(ns):
            I = np.where(inds==i)[0]
            I = I[np.argsort(x[I])]
            xi, yi, ei= x[I], y[I], inde[I] 
            for j in range(len(xi)-1):
                x0, x1 = xi[j], xi[j+1]
                y0, y1 = yi[j], yi[j+1]
                e0, e1 = edges[ei[j]], edges[ei[j+1]]

                d0 = np.sqrt((x0-e0.nodes[0].x)**2 + (y0-e0.nodes[0].y)**2)
                p0 = 1-d0/e0.l()
        
                d1 = np.sqrt((x1-e1.nodes[0].x)**2 + (y1-e1.nodes[0].y)**2)
                p1 = 1-d1/e1.l()
                links.append(Link([e0, e1], [p0, p1], self.ky))
                lnkid.append(i)
                
        self.links = links
        self.lnkid = lnkid

    def shiftxmax(self, xright):
        self.x[-1] = xright
        self.minEnergy()
        
    def E(self, x, ln, l0, kl, dy,  ni, nj, ri, rj, ls0, ks):
        
        m = len(x)
        p = [(x[i    ],  dy/2) for i in range(m)] +  \
            [(x[m-1-i], -dy/2) for i in range(m)]
        p = np.array(p)
        
        El, fl = fspring(p, ln, kl, l0)
        Es, fs = fspring2(p, ni, nj, ri, rj, ks, ls0)
        
        f = (fl[:m,0]+fl[2*m-1:m-1:-1,0])/2 + (fs[:m,0]+fs[2*m-1:m-1:-1,0])/2

        
        return np.sum(El)+np.sum(Es), -f#-f.flatten()

    def minEnergy(self):
        x = np.array(self.x)
        nodes, edges, links = self.nodes, self.edges, self.links
        
        m = len(x)
        bds = [(None,None) if i>0 and i<m-1 else (x[i],x[i]) for i in range(m)]

        ln = np.array([[nodes.index(n) for n in e.nodes] for e in edges if e.k>0])
        l0 = np.array([e.l0 for e in edges if e.k>0])
        kl = np.array([e.k  for e in edges if e.k>0])

        # links 两端点所在 edges 的两点节点编号，每一行对应一个 link
        ni = np.array( [[nodes.index( l.edges[0].nodes[i] ) for i in range(2)] for l in links] )
        nj = np.array( [[nodes.index( l.edges[1].nodes[i] ) for i in range(2)] for l in links] )

        # links 两端点所在 edges 上的位置，第一行对应一个 link
        ri = np.array( [ 1-l.prop[0] for l in links] ).reshape(-1,1)
        rj = np.array( [ 1-l.prop[1] for l in links] ).reshape(-1,1)
        
        ls0 = np.array( [l.l0  for l in links] ).reshape(-1,1)
        ks  = np.array( [l.k   for l in links] ).reshape(-1,1)

        res = minimize(self.E, x, args=(ln, l0, kl, self.dy,  ni, nj, ri, rj, ls0, ks), jac=True, method='L-BFGS-B', bounds=bds)

        self.x = res.x
        self.f = -res.jac
        for i in range(m):
            self.nodes[i      ].x = self.x[i]
            self.nodes[2*m-1-i].x = self.x[i]
        
    def plot(self, every=1):
        for e in self.edges:
            e.plot('-k.')
            
        for i, l in enumerate(self.links):
            if self.lnkid[i]%every==0: l.plot('-r')
            
    def __repr__(self):
        return f"Wall(xmax={self.xmax}, dx={self.dx}, dy={self.dy})"
    
# -----------------------------------------------------------------------------

if __name__=='__main__':
    np.random.seed(1)
    xmax = 5
    dx = 0.5
    W = 1
    km = 1*W
    pdf = lambda xmax,x: 1.25-x/xmax
    mf = 0.35
    rho = mf/1.65/(1-mf)*700*10
    kf = 1/10
    deg = 20

    plt.figure(figsize=(12,6))
    plt.axis('scaled')
    plt.axis([-1,xmax+3,-2*W/2,2*W/2])

    w = Wall(xmax, dx, W, km, rho, pdf, deg, kf)
    l0 = np.diff(w.x)
    w.plot(50)
    plt.pause(1)
    
    for i in range(1,11):
        w.shiftxmax(xmax+i/10)
        w.plot(50)
        plt.pause(0.1)
        
    l = np.diff(w.x)
    e = l/l0 - 1
    print(e)
    
