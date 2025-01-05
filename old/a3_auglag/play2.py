import sys
sys.path.append("/home/yuxiang/Documents/optalg22/optimization_algorithms_w22")
import numpy as np
from optalg.example_nlps.linear_program_ineq import LinearProgramIneq
from optalg.example_nlps.quadratic_program import QuadraticProgram
from optalg.example_nlps.quarter_circle import QuaterCircle
from optalg.example_nlps.halfcircle import HalfCircle
from optalg.example_nlps.logistic_bounds import LogisticWithBounds
from optalg.example_nlps.nonlinearA import NonlinearA
from optalg.interface.nlp_traced import NLPTraced
from optalg.example_nlps.f_r_eq import F_R_Eq
from solution import al_cal, bs_cal
from optalg.utils.finite_diff import *
from optalg.interface.objective_type import OT
import matplotlib.pyplot as plt
mu = 10
nu = 10
problem = LinearProgramIneq(2)
x = problem.getInitializationSample()
types = problem.getFeatureTypes()
id_f = [i for i, t in enumerate(types) if t == OT.f]
id_r = [i for i, t in enumerate(types) if t == OT.r]
id_ineq = [i for i, t in enumerate(types) if t == OT.ineq]
id_eq = [i for i, t in enumerate(types) if t == OT.eq]
lamda = np.zeros(len(id_ineq))
kappa = np.ones(len(id_eq))



test_x_num = np.linspace(-2,2,99)
test_x = np.transpose([test_x_num, test_x_num])
value = [al_cal(i,mu,nu,lamda, kappa, problem, id_r, id_f, id_ineq, id_eq)[0] for i in test_x]
value_bs = [bs_cal(i,problem,id_r,id_f,id_ineq,id_eq)[4] for i in test_x]
grad = np.array([al_cal(i,mu,nu,lamda, kappa, problem, id_r, id_f, id_ineq, id_eq)[1] for i in test_x])
plt.scatter(test_x_num,value, s=2, marker='^', c="b")
plt.scatter(test_x_num,grad[:,0], s=2, marker='^', c="r")
# #plt.scatter(test_x_num,value_bs, s=2, marker='^', c="g")
plt.show()
# print(grad)