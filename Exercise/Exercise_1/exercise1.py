import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def f_sq(x, C):
    y = np.array(x)
    fx = y.T @ C @ y  
    grad_fx = 2 * C @ y 
    hess_fx = 2 * C  
    return fx, grad_fx, hess_fx

def f_hole(x, C, a):
    y = np.array(x)
    numerator = y.T @ C @ y
    denominator = a**2 + numerator
    fx = numerator / denominator 
    grad_fx = (2 * a**2 / (denominator**2)) * (C @ y)  
    hess_fx = (2 * a**2 / (denominator**2)) * C - (8 * a**2 / (denominator**3)) * (C @ y)[:, None] @ (y.T @ C)[None, :]
    return fx, grad_fx, hess_fx

def f_exp(x, C):
    y = np.array(x)
    exponent = -0.5 * (y.T @ C @ y)
    fx = -np.exp(exponent)  
    grad_fx = np.exp(exponent) * (C @ y)  
    term1 = -np.exp(exponent) * (C @ y)[:, None] @ (y.T @ C)[None, :]
    term2 = np.exp(exponent) * C
    hess_fx = term1 + term2  
    return fx, grad_fx, hess_fx

def plotFunc(f, C, title, trace=None, a=None, bounds=[-1, 1]):
    x = np.linspace(bounds[0], bounds[1], 50)
    y = np.linspace(bounds[0], bounds[1], 50)
    xMesh, yMesh = np.meshgrid(x, y, indexing='ij')
    zMesh = np.zeros_like(xMesh)
    
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            if a is not None:
                zMesh[i, j], _, _ = f([xMesh[i, j], yMesh[i, j]], C, a)
            else:
                zMesh[i, j], _, _ = f([xMesh[i, j], yMesh[i, j]], C)

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121, projection="3d")
    surf = ax1.plot_surface(xMesh, yMesh, zMesh, cmap=cm.coolwarm, alpha=0.8)
    fig.colorbar(surf, ax=ax1)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f')
    ax1.set_title(title)

    if trace is not None:
        z_trace = [f([x, y], C, a)[0] if a is not None else f([x, y], C)[0] for x, y in trace]
        ax1.plot(trace[:, 0], trace[:, 1], z_trace, 'ro-', label='Trace', linewidth=2)

    ax2 = fig.add_subplot(122)
    cont = ax2.contourf(xMesh, yMesh, zMesh, cmap=cm.coolwarm)
    fig.colorbar(cont, ax=ax2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title(f"{title} (Contour)")
    
    plt.show()

def gradient_descent(f, grad_f, x0, C, alpha=0.01, tol=1e-6, max_iters=1000):
    x = np.array(x0)
    trace = [x.copy()]
    costs = []
    for i in range(max_iters):
        fx, grad_fx, _ = f(x, C)
        costs.append(fx)
        #print(f"Iteration {i+1}: Cost = {fx}")
        x_new = x - alpha * grad_fx
        if np.linalg.norm(x_new - x) < tol:
            print("Convergence reached!")
            break
        x = x_new
        trace.append(x.copy())
    trace = np.array(trace)
    return trace, costs

def create_matrix_C(c, n):
    C = np.zeros((n, n))
    for i in range(1, n + 1):  
        C[i - 1, i - 1] = c ** ((i - 1) / (n - 1))  
    return C

if __name__ == "__main__":
    c = 10
    n = 2
    C = create_matrix_C(c, n)
    print('C: ', C)
    x0 = [1, 1]



    print('f_sq: ', f_sq)
    trace_sq, costs_sq = gradient_descent(f_sq, lambda x, C: f_sq(x, C)[1], x0, C)
    a = 0.1
    print('f_hole: ', f_hole)
    trace_hole, costs_hole = gradient_descent(lambda x, C: f_hole(x, C, a), lambda x, C: f_hole(x, C, a)[1], x0, C)
    print('f_exp: ', f_exp)  
    trace_exp, costs_exp = gradient_descent(f_exp, lambda x, C: f_exp(x, C)[1], x0, C)

    plotFunc(f_sq, C, title="f_sq(x)", trace=trace_sq)
    plotFunc(f_hole, C, title="f_hole(x)", trace=trace_hole, a=a)
    plotFunc(f_exp, C, title="f_exp(x)", trace=trace_exp)
