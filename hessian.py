import autograd.numpy as au
from autograd import grad, jacobian 
import numpy as np

# Hessian
p = np.array([1, 2, 3], dtype=float)

def f(x): # Objective function
    return 2*x[0]*x[1]**3+3*x[1]**2*x[2]+x[2]**3*x[0]
grad_f = grad(f) # gradient of the objective function
hessian_f = jacobian(grad_f) # Hessian of the objective function
print("gradient vector:",grad_f(p))
print("Hessian matrix:\n",hessian_f(p))

