import numpy as np

def numdiff(x, y, n=1):
    
    dydx = np.diff(y, n)/np.diff(x[(n-1):], 1)**n 
    x    = 0.5*(x[n:] + x[:-n])
    return x, dydx