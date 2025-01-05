import sys
sys.path.append("../..")
from optalg.interface.objective_type import OT
from optalg.interface.nlp import NLP
import numpy as np
import copy


def bs_cal(x, nlp, id_r, id_f, id_ineq, id_eq):
    y,J = nlp.evaluate(x)
    H = nlp.getFHessian(x)
    y_r = y[id_r]
    y_f = y[id_f]
    J_r = J[id_r]
    J_f = J[id_f]
    y_g = y[id_ineq]
    J_g = J[id_ineq]
    y_h = y[id_eq]
    J_h = J[id_eq]
    F = (y_f + np.dot(y[id_r], y[id_r]))[0]
    J_F = (J_f + 2*np.transpose(J_r)@y_r)[0]
    H_F = H + 2*(np.transpose(J_r)@J_r)
    return y_g, J_g, y_h, J_h, F, J_F, H_F

def al_cal(x:np.array,
                mu:float,
                nu:float,
                lamda: np.array,
                kappa: np.array,
                nlp:NLP,
                id_r:np.array,
                id_f:np.array,
                id_ineq:np.array,
                id_eq:np.array):
    y_g, J_g, y_h, J_h, F, J_F, H_F = bs_cal(x,nlp, id_r, id_f,id_ineq,id_eq)
#################AL terms################
    ind_lamda = np.where(lamda > 0)[0]
    ind_g = np.where(y_g > 0)[0]
    ind_gx2 = np.union1d(ind_lamda, ind_g) # indices of the third term
    if np.shape(ind_gx2)[0] == 0:
        sum_sel_g = 0
        #J_sel_g = np.zeros_like(y_g)
        J_sel_g = 0
        H_sel_g = 0
    else:
        sel_g = np.array([(y_g[i])**2 for i in ind_gx2])
        sum_sel_g = np.sum(sel_g)
        J_sel_g = np.zeros_like(x)
        H_sel_g = np.zeros_like(H_F)
        for i in ind_gx2:
            J_sel_g += 2*J_g[i]*y_g[i] # gradient of the third term
            H_sel_g += 2*np.outer(J_g[i],J_g[i])

    if np.shape(y_h)[0] == 0:
        #y_h = np.zeros_like(y_g)
        y_h = 0
        #J_h = np.zeros_like(J_g)
        J_h = 0
        #kappa = np.zeros_like(y_h)
        kappa = 0
        J_sum_h = 0
        J_sum_h_s = 0
        sum_h = 0
        sum_k_h = 0
        H_sum_h_s = 0
    else:
        J_sum_h = np.zeros_like(x)
        J_sum_h_s = np.zeros_like(x)
        H_sum_h_s = np.zeros_like(H_F)
        for i in range(len(J_h)):
            J_sum_h += J_h[i]*kappa[i]
            J_sum_h_s += 2*J_h[i]*y_h[i]
            H_sum_h_s += 2*np.outer(J_h[i],J_h[i])
        sum_h = np.sum(y_h**2)
        sum_k_h = np.sum(kappa*y_h)
    sum_l_g = np.sum(lamda*y_g)
    J_sum_g = np.zeros_like(x)
    for i in range(len(J_g)):
        J_sum_g += J_g[i]*lamda[i]
    A = F + sum_l_g + mu*sum_sel_g + sum_k_h + nu*sum_h
    dxA = J_F + J_sum_g + mu*(J_sel_g) + J_sum_h + nu*(J_sum_h_s)
    ddxA = H_F + mu*(H_sel_g) + nu*H_sum_h_s
    # print('gradient:\n', dxA)
    # print('value:\n', A)
    # print('x:\n', x)
    # print('F value:\n', F)
    # print('init x:\n', x)
    # print('Jacob of F:\n', J_F)

    return A, dxA,ddxA,y_g,y_h

def inner_p(x:np.array,
                lamda:np.array,
                kappa:np.array,
                nlp:NLP,
                id_r:np.array,
                id_f:np.array,
                id_ineq:np.array,
                id_eq:np.array,
                alpha = 1,
                mu = 10,
                nu = 10,
                qls = 0.01,
                qap = 1.2,
                qam = 0.1,
                n_l = 1e-2
                ):
    I = np.identity(np.shape(x)[0])
    while True:
        f, df, hf,gx,hx = al_cal(x,mu,nu, lamda, kappa,nlp,id_r,id_f,id_ineq,id_eq)
        #update_pack = bs_cal(x,nlp,id_r,id_f,id_ineq,id_eq)
        #gx = update_pack[0]
        #hx = update_pack[2]
        #delta = -df
        delta = np.linalg.inv(hf + n_l*I)@(np.transpose(-df))
        while True:
            left_t = al_cal(x + alpha*delta,mu,nu, lamda, kappa,nlp,id_r,id_f,id_ineq,id_eq)[0]
            right_t = f + qls*alpha*(df)@(np.transpose(delta))
            if left_t<= right_t:
                break
            alpha = qam*alpha
        x += alpha*delta
        #trace_x.append(copy.deepcopy(x))
        alpha = min(qap*alpha, 1)

        if (np.linalg.norm(alpha*delta,1) < 1e-3):
            break
    return x, gx, hx

def solve(nlp: NLP):
    """
    solver for constrained optimization


    Arguments:
    ---
        nlp: object of class NLP that contains features of type OT.f, OT.r, OT.eq, OT.ineq

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

    To know which type (normal cost, least squares, equalities or inequalities) are the entries in
    the feature vector, use:

    types = nlp.getFeatureTypes()

    id_f = [i for i, t in enumerate(types) if t == OT.f]
    id_r = [i for i, t in enumerate(types) if t == OT.r]
    id_ineq = [i for i, t in enumerate(types) if t == OT.ineq]
    id_eq = [i for i, t in enumerate(types) if t == OT.eq]

    Note that getFHessian(x) only returns the Hessian of the term of type OT.f.

    For example, you can get the value and Jacobian of the equality constraints with y[id_eq] (1-D np.array), and J[id_eq] (2-D np.array).

    All input NLPs contain one feature of type OT.f (len(id_f) is 1). In some problems,
    there no equality constraints, inequality constraints or residual terms.
    In those cases, some of the lists of indexes will be empty (e.g. id_eq = [] if there are not equality constraints).

    """

    x = nlp.getInitializationSample()
    types = nlp.getFeatureTypes()
    id_f = [i for i, t in enumerate(types) if t == OT.f]
    id_r = [i for i, t in enumerate(types) if t == OT.r]
    id_ineq = [i for i, t in enumerate(types) if t == OT.ineq]
    id_eq = [i for i, t in enumerate(types) if t == OT.eq]
    lamda = np.zeros(len(id_ineq))
    kappa = np.zeros(len(id_eq))
###########values################
    trace_x = [x]
    trace_lambda = [lamda]
    mu = 10
    nu = 10
    while True:
        x_last = copy.deepcopy(x)
        x, gx, hx = inner_p(x=x,lamda=lamda,kappa=kappa,nlp=nlp,id_r=id_r,id_f=id_f,id_eq=id_eq,id_ineq=id_ineq)
        lamda_update = np.copy(lamda + 2*mu*gx)
        lamda_update = np.where(lamda_update > 0, lamda_update, lamda_update*0)
        kappa_update = kappa + 2*nu*hx
        lamda = lamda_update
        kappa = kappa_update
        trace_x.append(x_last)
        trace_lambda.append(lamda)
        if (np.linalg.norm(x-x_last,1) < 1e-4) and (gx < 1e-2).all() and (np.absolute(hx) < 1e-2).all():
            break


    return x
