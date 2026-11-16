import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(A, x0, h, steps):
    path = np.empty((steps+1, len(x0)))
    path[0] = x0
    for i in range(steps):
        path[i+1] = path[i] - h * (A @ path[i]) 
    return path

if __name__ == "__main__":
    A = np.array([[2.0]])
    x0 = np.array([1.0])
    h = 0.1
    path = gradient_descent(A, x0, h, 10)
    print(np.allclose(path[:, 0], (1 - h * 2) ** np.arange(11)))