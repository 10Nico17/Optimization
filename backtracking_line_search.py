import numpy as np
 
def backtracking_line_search(f, grad_f, x, p, alpha=0.5, beta=0.8, max_iter=100):
    """
    Backtracking line search algorithm for unconstrained optimization.
 
    Parameters:
        f (function): Objective function to minimize.
        grad_f (function): Gradient of the objective function.
        x (ndarray): Current point (numpy array).
        p (ndarray): Search direction (numpy array).
        alpha (float): Scaling factor for step size (default: 0.5).
        beta (float): Contraction factor for step size (default: 0.8).
        max_iter (int): Maximum number of iterations (default: 100).
 
    Returns:
        float: Optimal step size.
    """
    # Initialize step size
    t = 0.5
 
    # Armijo-Goldstein condition parameters
    c = 0.1
 
    # Iteratively adjust step size
    for _ in range(max_iter):
        # Evaluate objective function at new point
        f_new = f(x + t * p)
 
        # Evaluate Armijo-Goldstein condition
        if f_new <= f(x) + c * t * np.dot(grad_f(x), p):
            return t  # Optimal step size found
 
        # Reduce step size
        t *= beta
 
    # If no optimal step size found, return the final step size
    return t
 
# Example usage:
# Define objective function and its gradient
def f(x):
    return x[0]**2 +  x[1]**2
 
def grad_f(x):
    return np.array([2*x[0], 2*x[1]])
 
# Initial point and search direction
x = np.array([1.0, 1.0])
p = np.array([-1.0, -1.0])
 
# Perform backtracking line search
optimal_step_size = backtracking_line_search(f, grad_f, x, p)
 
print("Optimal Step Size:", optimal_step_size)