import sys
sys.path.append("../..")
from optalg.interface.objective_type import OT
from optalg.interface.nlp import NLP
import numpy as np
import copy


def compute_base_terms(var_x, var_nlp, var_id_r, var_id_f, var_id_ineq, var_id_eq):
    var_y, var_J = var_nlp.evaluate(var_x)
    var_H = var_nlp.getFHessian(var_x)
    var_y_r = var_y[var_id_r]
    var_y_f = var_y[var_id_f]
    var_J_r = var_J[var_id_r]
    var_J_f = var_J[var_id_f]
    var_y_g = var_y[var_id_ineq]
    var_J_g = var_J[var_id_ineq]
    var_y_h = var_y[var_id_eq]
    var_J_h = var_J[var_id_eq]
    var_F = (var_y_f + np.dot(var_y[var_id_r], var_y[var_id_r]))[0]
    var_J_F = (var_J_f + 2 * np.transpose(var_J_r) @ var_y_r)[0]
    var_H_F = var_H + 2 * (np.transpose(var_J_r) @ var_J_r)
    return var_y_g, var_J_g, var_y_h, var_J_h, var_F, var_J_F, var_H_F

def compute_augmented_lagrangian(var_x: np.array,
                                 var_mu: float,
                                 var_nu: float,
                                 var_lambda: np.array,
                                 var_kappa: np.array,
                                 var_nlp: NLP,
                                 var_id_r: np.array,
                                 var_id_f: np.array,
                                 var_id_ineq: np.array,
                                 var_id_eq: np.array):
    var_y_g, var_J_g, var_y_h, var_J_h, var_F, var_J_F, var_H_F = compute_base_terms(var_x, var_nlp, var_id_r, var_id_f, var_id_ineq, var_id_eq)
    var_ind_lambda = np.where(var_lambda > 0)[0]
    var_ind_g = np.where(var_y_g > 0)[0]
    var_ind_gx2 = np.union1d(var_ind_lambda, var_ind_g)
    if np.shape(var_ind_gx2)[0] == 0:
        var_sum_sel_g = 0
        var_J_sel_g = 0
        var_H_sel_g = 0
    else:
        var_sel_g = np.array([(var_y_g[i])**2 for i in var_ind_gx2])
        var_sum_sel_g = np.sum(var_sel_g)
        var_J_sel_g = np.zeros_like(var_x)
        var_H_sel_g = np.zeros_like(var_H_F)
        for i in var_ind_gx2:
            var_J_sel_g += 2 * var_J_g[i] * var_y_g[i]
            var_H_sel_g += 2 * np.outer(var_J_g[i], var_J_g[i])

    if np.shape(var_y_h)[0] == 0:
        var_y_h = 0
        var_J_h = 0
        var_kappa = 0
        var_J_sum_h = 0
        var_J_sum_h_s = 0
        var_sum_h = 0
        var_sum_k_h = 0
        var_H_sum_h_s = 0
    else:
        var_J_sum_h = np.zeros_like(var_x)
        var_J_sum_h_s = np.zeros_like(var_x)
        var_H_sum_h_s = np.zeros_like(var_H_F)
        for i in range(len(var_J_h)):
            var_J_sum_h += var_J_h[i] * var_kappa[i]
            var_J_sum_h_s += 2 * var_J_h[i] * var_y_h[i]
            var_H_sum_h_s += 2 * np.outer(var_J_h[i], var_J_h[i])
        var_sum_h = np.sum(var_y_h**2)
        var_sum_k_h = np.sum(var_kappa * var_y_h)
    var_sum_l_g = np.sum(var_lambda * var_y_g)
    var_J_sum_g = np.zeros_like(var_x)
    for i in range(len(var_J_g)):
        var_J_sum_g += var_J_g[i] * var_lambda[i]
    var_A = var_F + var_sum_l_g + var_mu * var_sum_sel_g + var_sum_k_h + var_nu * var_sum_h
    var_dxA = var_J_F + var_J_sum_g + var_mu * var_J_sel_g + var_J_sum_h + var_nu * var_J_sum_h_s
    var_ddxA = var_H_F + var_mu * var_H_sel_g + var_nu * var_H_sum_h_s
    return var_A, var_dxA, var_ddxA, var_y_g, var_y_h

def optimize_inner_loop(x: np.array,
                        lamda: np.array,
                        kappa: np.array,
                        nlp: NLP,
                        id_r: np.array,
                        id_f: np.array,
                        id_ineq: np.array,
                        id_eq: np.array,
                        alpha=1,
                        mu=10,
                        nu=10,
                        qls=0.01,
                        qap=1.2,
                        qam=0.1,
                        n_l=1e-2):
    I = np.identity(np.shape(x)[0])
    while True:
        f, df, hf, gx, hx = compute_augmented_lagrangian(x, mu, nu, lamda, kappa, nlp, id_r, id_f, id_ineq, id_eq)
        delta = np.linalg.inv(hf + n_l * I) @ (np.transpose(-df))
        while True:
            left_t = compute_augmented_lagrangian(x + alpha * delta, mu, nu, lamda, kappa, nlp, id_r, id_f, id_ineq, id_eq)[0]
            right_t = f + qls * alpha * (df) @ (np.transpose(delta))
            if left_t <= right_t:
                break
            alpha = qam * alpha
        x += alpha * delta
        alpha = min(qap * alpha, 1)
        if (np.linalg.norm(alpha * delta, 1) < 1e-3):
            break
    return x, gx, hx


def solve(nlp: NLP, Dout={}):
    x = nlp.getInitializationSample()
    types = nlp.getFeatureTypes()


    id_f = [i for i, t in enumerate(types) if t == OT.f]
    id_r = [i for i, t in enumerate(types) if t == OT.r]
    id_ineq = [i for i, t in enumerate(types) if t == OT.ineq]
    id_eq = [i for i, t in enumerate(types) if t == OT.eq]
    lamda = np.zeros(len(id_ineq))
    kappa = np.zeros(len(id_eq))
    trace_x = [x]
    trace_lambda = [lamda]
    mu = 10
    nu = 10
    
    
    while True:
        x_last = copy.deepcopy(x)
        x, gx, hx = optimize_inner_loop(x=x,lamda=lamda,kappa=kappa,nlp=nlp,id_r=id_r,id_f=id_f,id_eq=id_eq,id_ineq=id_ineq)
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
