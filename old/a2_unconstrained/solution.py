import sys
sys.path.append("../..")
from optalg.interface.objective_type import OT
from optalg.interface.nlp import NLP
import numpy as np


def solve(nlp: NLP):
    """
    Solver for unconstrained optimization


    Arguments:
    ---
        nlp: object of class NLP that only contains one feature of type OT.f.

    Returns:
        x: local optimal solution (1-D np.ndarray)


    Task:
    ---

    See instructions and requirements in the coding assignment PDF in ISIS.

    Notes:
    ---

    Get the starting point with:
    x = nlp.getInitializationSample()

    Use the following to query the function and gradient of the problem:
    phi, J = nlp.evaluate(x)

    phi is a vector (1D np.ndarray); use phi[0] to access the cost value 
    (a float number). 

    J is a Jacobian matrix (2D np.ndarray). Use J[0] to access the gradient 
    (1D np.array) of phi[0].

    Use getFHessian to query the Hessian.

    H = nlp.getFHessian(x)

    H is a matrix (2D np.ndarray) of shape n x n.


    """

    # sanity check on the input nlp
    assert len(nlp.getFeatureTypes()) == 1
    assert nlp.getFeatureTypes()[0] == OT.f

    # get start point
    x = nlp.getInitializationSample()
    #x = np.array([0.5,0.5,0.5])
    # f,df = nlp.evaluate(x)
    # df = df[0]
    # f =f[0]
    H = nlp.getFHessian(x)
    I = np.identity(np.shape(H)[0])
    lamda = 1e-2
    alpha = 1
    qls = 0.01
    qlmp = 1.2
    qlmm = 0.5
    qam = 0.5
    qap = 1.2
    while True:
        f,df = nlp.evaluate(x)
        df = df[0]
        f =f[0]
        H = nlp.getFHessian(x)
        delta = -np.linalg.inv(H + lamda*I)@(df)
        if (df.T)@delta > 0:
            delta = -(df/(np.linalg.norm(df,2)))
        left_term = nlp.evaluate(x + alpha * delta)[0][0]
        right_term = f + qls*(df.T)@(alpha*delta)
        while True:
            if left_term <= right_term:
                break
            alpha *= qam
            lamda *= qlmp
            delta = -np.linalg.inv(H + lamda*I)@(df)
            if (df.T)@delta > 0:
                delta = -(df/(np.linalg.norm(df,2)))
            left_term = nlp.evaluate(x + alpha * delta)[0][0]
            right_term = f + qls*(df.T)@(alpha*delta)
        x += alpha*delta
        alpha = min(qap*alpha, 1)
        lamda *= qlmm
        if np.linalg.norm(alpha*delta, np.inf) < 1e-5:
            break
    # return found solution
    return x
