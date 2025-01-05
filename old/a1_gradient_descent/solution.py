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

    # get start point
    x = nlp.getInitializationSample()
    def fx(x: float) ->float :
        a,b = nlp.evaluate(x)
        return a[0]
    def dfx(x:float) ->float :
        a,b = nlp.evaluate(x)
        return b[0]
    #
    ro = 0.5
    c = 0.1
    alpha = 1
    co_al =1.2
    # Write your code here
    while True:
        
        dfx_1 = dfx(x)
        delta = -dfx_1
        delta_t = np.transpose(delta)
        fx_1 = fx(x)
        while True:
            if fx(x + alpha*delta) <= (fx_1 + c*alpha*(dfx_1@delta_t)) :
                break
            alpha = ro*alpha
        x = x + alpha * (delta)
        alpha = co_al*alpha
        if np.linalg.norm(alpha*dfx_1, np.shape(x)[0]) < 10e-6 :
            break
    return x
