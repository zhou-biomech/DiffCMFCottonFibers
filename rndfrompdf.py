import numpy as np
from scipy.stats.sampling import NumericalInversePolynomial

# -----------------------------------------------------------------------------
# https://stackoverflow.com/questions/41657704
class rdist:
    def __init__(self, f, xmax):
        self.xmax = xmax
        self.f = f
        
    def support(self):
        return (0, self.xmax)

    def pdf(self, x):
        return self.f(x)

# -----------------------------------------------------------------------------
# 根据概率密度函数生成随机数：pdf概率密度函数，[0,xmax] 随机数的范围，n 随机数的数量
def rndfrompdf(pdf, xmax, n, xshift=0):
    
    xpdf = lambda x: pdfshift(pdf, xmax, x, xshift)
    
    dist = rdist(xpdf, xmax+2*xshift)
    gen = NumericalInversePolynomial(dist)
    x = gen.rvs(size=n)
    return x - xshift


# -----------------------------------------------------------------------------

def pdfshift(pdf, xmax, x, xshift=0):
    xshift = abs(xshift)
    if x<xshift:
        y = pdf(0)
    elif x>xmax+xshift:
        y = pdf(xmax)
    else:
        y = pdf(x-xshift)
    return y

# -----------------------------------------------------------------------------
if __name__=='__main__':
    from matplotlib import pyplot as plt
    from scipy.interpolate import interp1d

    xmax = 5
    xshift = 1
    
    pdf = lambda x: 1.5 - x/xmax

    xi = rndfrompdf(pdf, xmax, 100000, xshift)

    plt.hist(xi, density=True, bins=20)
    plt.show()
