import sys
sys.path.append("/home/yuxiang/Documents/optalg22/optimization_algorithms_w22")
from optalg.interface.objective_type import OT
from optalg.example_nlps.quadratic_program import QuadraticProgram
from optalg.example_nlps.linear_program_ineq import LinearProgramIneq
from optalg.example_nlps.halfcircle import HalfCircle
from optalg.interface.nlp_traced import NLPTraced
from optalg.interface.nlp import NLP
import numpy as np
import copy

def value_log(x,mu,nlp, id_ineq, id_f):
    y,J = nlp.evaluate(x)
    H = nlp.getFHessian(x)
    cost_ineq = y[id_ineq]
    J_ineq = J[id_ineq]
    cost_ineq = np.where(cost_ineq < 0, -cost_ineq, 1e-20)
    cost_eq = y[id_f[0]]
    J_eq = J[id_f[0]]
    Cost = cost_eq - mu*np.sum(np.log(cost_ineq))
    H_ineq = np.zeros((np.shape(x)[0],np.shape(x)[0]))
    J_ineq_1 = np.zeros(np.shape(x)[0])
    for i in range(len(id_ineq)):
        H_ineq += (1/(cost_ineq[i]**2))*np.outer(J_ineq[i],J_ineq[i])
        J_ineq_1 += (1/(cost_ineq[i]))*J_ineq[i]
    Jacob = J_eq - mu*J_ineq_1
    Hess = H + mu*H_ineq
    return Cost, Jacob, Hess

def newton(x_inner_p, alpha, q_alpha_plus, q_alpha_minus, qls, lamda, mu, nlp, id_ineq, id_f):
    I = np.identity(np.shape(x_inner_p)[0])
    #print(delta)
    while True:
        cost, jacob, hess = value_log(x_inner_p,mu, nlp, id_ineq, id_f)
        delta = np.linalg.inv(hess + lamda*I)@(-jacob)
        while True:
            left_term = value_log(x_inner_p+alpha*delta,mu, nlp, id_ineq, id_f)[0]
            right_term = cost + qls*alpha*((jacob.T)@delta)
            if (left_term <= right_term):
                break
            alpha *= q_alpha_minus
        print("Linesearch complete")
        x_inner_p += alpha*delta
        if np.linalg.norm(alpha*delta, np.inf) < 1e-3:
            break
        alpha = min(q_alpha_plus*alpha, 1)
    return x_inner_p

def solve(nlp: NLP):
    """
    solver for constrained optimization (cost term and inequalities)


    Arguments:
    ---
        nlp: object of class NLP that contains one feature of type OT.f,
            and m features of type OT.ineq.

    Returns:
        x: local optimal solution (1-D np.ndarray)


    Task:
    ---
    See the coding assignment PDF in ISIS.


    Notes:
    ---

    Get the starting point with:

    x = nlp.getInitializationSample()

    To know which type (cost term or inequalities) are the entries in
    the feature vector, use:
    types = nlp.getFeatureTypes()

    Index of cost term
    id_f = [ i for i,t in enumerate(types) if t == OT.f ]
    There is only one term of type OT.f ( len(id_f) == 1 )

    Index of inequality constraints:
    id_ineq = [ i for i,t in enumerate(types) if t == OT.ineq ]

    Get all features (cost and constraints) with:

    y,J = nlp.evaluate(x)
    H = npl.getFHessian(x)

    The value, gradient and Hessian of the cost are:

    y[id_f[0]] (scalar), J[id_f[0]], H

    The value and Jacobian of inequalities are:
    y[id_ineq] (1-D np.array), J[id_ineq]


    """
    x = nlp.getInitializationSample()
    types = nlp.getFeatureTypes()
    miu = 1
    id_f = [i for i, t in enumerate(types) if t == OT.f]
    id_ineq = [i for i, t in enumerate(types) if t == OT.ineq]
    #----------function and its derivative-------
    # alpha = 1
    # q_alpha_plus = 1.2
    # q_alpha_minus = 0.5
    # qls = 0.01
    # lamda = 1e-2
    mu_minus = 0.5
    temp = copy.deepcopy(x)
    while True:
        x = newton(x,
                alpha = 1,
                q_alpha_minus = 0.5,
                q_alpha_plus = 1.2,
                qls = 0.01,
                lamda = 1e-2,
                mu = miu,
                nlp = nlp,
                id_ineq = id_ineq,
                id_f = id_f)
        print("log barrier loop")
        if np.linalg.norm(temp-x, 1) < 1e-3:
            break
        temp = copy.deepcopy(x)
        miu *= mu_minus
    return x

if __name__ == "__main__":
    MAX_EVALUATE = 10000
    # problem = NLPTraced(LinearProgramIneq(2), max_evaluate=MAX_EVALUATE)
    # x = solve(problem)
    problem = NLPTraced(HalfCircle(), max_evaluate=MAX_EVALUATE)
    x = solve(problem)
    print(x)
