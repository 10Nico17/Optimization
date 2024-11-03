import sys
sys.path.append("../..")
from optalg.interface.objective_type import OT
from optalg.interface.nlp import NLP
import numpy as np


def solve(nlp: NLP):
    """
    Gradient descent with backtracking Line search
    Arguments:
    ---
        nlp: object of class NLP that only contains one feature of type OT.f.

    Returns:
        x: local optimal solution (1-D np.ndarray)
    Task:
    ---

    Implement a solver that does iterations of gradient descent
    with a backtracking line search
    x = x - k * Df(x),
    where Df(x) is the gradient of f(x)
    and the step size k is computed adaptively with backtracking line search
    Notes:
    ---
    Get the starting point with:
    x = nlp.getInitializationSample()
    Use the following to query the problem:
    
    phi, J = nlp.evaluate(x)
    phi is a vector (1D np.ndarray); use phi[0] to access the cost value
    (a float number).
    J is a Jacobian matrix (2D np.ndarray). Use J[0] to access the gradient
    (1D np.array) of phi[0].

    """
    # sanity check on the input nlp
    assert len(nlp.getFeatureTypes()) == 1
    assert nlp.getFeatureTypes()[0] == OT.f 

    alpha = 0.5     
    rho_plus = 1.2   
    rho_minus = 0.5  
    delta_max = float('inf')  
    rho_ls = 0.01    
    theta = 1e-4     

    x = np.copy(nlp.getInitializationSample())
    iteration = 0
    max_iters = 1000  

    while iteration < max_iters:
        phi, J = nlp.evaluate(x)
        grad = J[0] 
        phi_val = phi[0]  

        descent_dir = -grad / np.linalg.norm(grad)

        step_size = alpha
        while True:
            new_x = x + step_size * descent_dir
            new_phi, _ = nlp.evaluate(new_x)
            new_phi_val = new_phi[0]

            if new_phi_val <= phi_val + rho_ls * np.dot(grad, step_size * descent_dir):
                break  
            step_size *= rho_minus  

        x = x + step_size * descent_dir
        alpha = min(rho_plus * step_size, delta_max)  

        if np.linalg.norm(step_size * descent_dir) < theta:
            print(f"Convergence reached in {iteration} iterations.")
            break
        iteration += 1

    return x