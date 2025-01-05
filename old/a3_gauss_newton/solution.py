import sys
sys.path.append("../..")
from optalg.interface.objective_type import OT
from optalg.interface.nlp import NLP
import numpy as np


def prob_solve():
    pass



def solve(nlp: NLP):
    """
    solver for unconstrained optimization, including least squares terms

    Arguments:
    ---
        nlp: object of class NLP that contains one feature of type OT.f,
            and m features of type OT.r

    Returns:
        x: local optimal solution (1-D np.ndarray)


    Task:
    ---
    See the coding assignment PDF in ISIS.


    Notes:
    ---

    Get the starting point with:

    x = nlp.getInitializationSample()

    You can query the problem with:

    y,J = nlp.evaluate(x)
    H = npl.getFHessian(x)

    To know which type (normal cost or least squares) are the entries in
    the feature vector, use:

    types = nlp.getFeatureTypes()

    id_f = [i for i, t in enumerate(types) if t == OT.f]
    id_r = [i for i, t in enumerate(types) if t == OT.r]

    The total cost is:

    y[self.id_f[0]] + np.dot(y[self.id_r], y[self.id_r])

    Note that getFHessian(x) only returns the Hessian of the term of type OT.f.

    For example, you can get the value and Jacobian of the least squares terms with y[id_r] (1-D np.array), and J[id_r] (2-D np.array).

    The input NLP contains one feature of type OT.f (len(id_f) is 1) (but sometimes f = 0 for all x).
    If there are no least squares terms, the lists of indexes id_r will be empty (e.g. id_r = []).

    """
    x = nlp.getInitializationSample()
    def calculator(x):
        types = nlp.getFeatureTypes()
        y,J = nlp.evaluate(x)
        H = nlp.getFHessian(x)
        id_f = [i for i, t in enumerate(types) if t == OT.f]
        id_r = [i for i, t in enumerate(types) if t == OT.r]
        y_r = y[id_r]
        y_f = y[id_f]
        J_r = J[id_r]
        J_f = J[id_f]
        F = y[id_f[0]] + np.dot(y[id_r], y[id_r])
        J_F = J_f + 2*np.transpose(J_r)@y_r
        H_F = H + 2*(np.transpose(J_r)@J_r)
        return F, J_F, H_F

    lamda = 0.1
    qap = 1.2
    qam = 0.5
    qls = 0.01
    alpha = 1
    

    while True:
        F, JF, HF = calculator(x)
        I = np.identity(np.shape(HF)[0])
        delta = np.linalg.inv(HF + lamda*I)@(np.reshape(-JF, (-1,1)))
        delta = np.squeeze(np.reshape(delta, (1,-1)))
        while True:
            TTT = x + delta*alpha
            left_t = calculator(x + delta*alpha)[0]
            right_t = F + qls*alpha*(np.squeeze(JF.T))@(delta)
            if left_t <= right_t:
                break
            alpha *= qam
        x += alpha*delta
        alpha = min(qap*alpha, 1)
        if np.linalg.norm(alpha*delta,np.inf) < 1e-5:
            break

        



    #
    # Write your code Here
    #


    return x
