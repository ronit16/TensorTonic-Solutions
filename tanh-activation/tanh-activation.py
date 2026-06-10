import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    # Write code here
    x = np.array(x, dtype=float)
    exp = np.exp(2*x)
    return (exp - 1) / (exp + 1)