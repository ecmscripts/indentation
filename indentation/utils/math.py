import numpy as np

import numpy as np

def numdiff(x, y, n=1):
    """
    Calculate the nth numerical derivative of y with respect to x using central differences.
    
    Parameters
    ----------
    x : array-like
        Independent variable values, must be monotonically increasing or decreasing
    y : array-like
        Dependent variable values, must have same length as x
    n : int, optional (default=1)
        Order of the derivative to compute (1 for first derivative, 2 for second, etc.)
        
    Returns
    -------
    x_mid : ndarray
        Midpoint x values where the derivative is calculated
    dydx : ndarray
        nth derivative values
        
    Raises
    ------
    ValueError
        If inputs have invalid shapes or values
    """
    # Convert inputs to numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Input validation
    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    if n < 1:
        raise ValueError("Order of derivative must be positive")
    if x.size < n + 1:
        raise ValueError(f"Input arrays must have length > {n} for {n}th derivative")
    
    # Check if x is monotonic
    dx = np.diff(x)
    if not (np.all(dx > 0) or np.all(dx < 0)):
        raise ValueError("x values must be monotonically increasing or decreasing")
    
    # Calculate the nth derivative
    dydx = np.diff(y, n)
    
    # Calculate the appropriate denominator based on nth derivative
    dx_product = 1
    for i in range(n):
        dx_i = np.diff(x[i:-(n-i-1) if n-i-1 > 0 else None])
        dx_product *= dx_i
    
    dydx = dydx / dx_product
    
    # Calculate midpoints for the x values
    x_mid = np.mean([x[i:-(n-i) if n-i > 0 else None] for i in range(n)], axis=0)
    
    return x_mid, dydx
