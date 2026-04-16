import numpy as np

def f_1(x):
  return -x*np.sin(np.sqrt(np.abs(x)))

def f_2(val):
    X = val[0]
    Y = val[1]
    r2 = X**2 + Y**2
    z = (r2)**0.25 * (np.sin(50 * (r2)**0.1)**2 + 1)
    return z