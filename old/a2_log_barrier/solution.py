import sys
sys.path.append("../..")
from optalg.interface.objective_type import OT
from optalg.interface.nlp import NLP
import numpy as np
import copy



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
    def logprob(x,miu):
        y,J = nlp.evaluate(x)
        value_c = y[id_f[0]]
        value_e_log = np.log([-y[i] for i in id_ineq])
        value_e = miu * np.sum(value_e_log)
        # if np.isnan(value_e) == True:
        #     value_e = -1e6
        value = value_c - value_e
        df_c = J[id_f[0]]
        df_e_log = [((1/(-y[i]))*(-J[i])) for i in id_ineq]
        df_e_log = np.array(df_e_log)
        df_e = np.sum(df_e_log, axis = 0)
        df_e = miu*df_e
        df = df_c - df_e
        #print("df:",df)
        return value, df
    
    def hes(x,miu):
        H = 0
        y,J = nlp.evaluate(x)
        for i in id_ineq:
            term_1 = 1/((y[i])**2)
            term_2 = J[i]
            term_3 = np.reshape(J[i],(1,-1))
            mult = term_1*(term_3@term_2)
            H += miu*mult
        return H

    mium = 0.5
    I = np.identity(np.shape(x)[0])
    #-------------iterative part-------------
    xtemp = copy.deepcopy(nlp.getInitializationSample())
    while True:
        lamda = 1e2 # 1e2
        alpha = 1
        qls = 0.5
        qam = 0.4
        qap = 1.2
        qlmp = 1.2
        qlmm = 0.8
        while True:
            #H = nlp.getFHessian(x)
            H = hes(x,miu) + nlp.getFHessian(x)
            val, df = logprob(x,miu)
            delta = np.linalg.inv(H+lamda*I)@(-df)
            # if (df.T)@delta > 0:
            #     print("non-descent delta!")
            #     delta = df/(np.linalg.norm(df, 1))
            left_term = logprob(x+alpha*delta,miu)[0]
            linesch = qls*alpha*((df.T)@(delta))
            right_term = val + linesch
            while True:
                if left_term <= right_term:
                    break
                alpha *= qam
                # lamda *= qlmp
                # H = hes(x,miu) + nlp.getFHessian(x)
                # val, df = logprob(x,miu)
                # delta = np.linalg.inv(H+lamda*I)@(-df)
                left_term = logprob(x+alpha*delta,miu)[0]
                linesch = qls*alpha*((df.T)@(delta))
                right_term = val + linesch
            x += alpha*delta
            # alpha = min(qap*alpha, 1)
            # lamda *= qlmm
            if np.linalg.norm(alpha*delta, np.inf) < 1e-4:
                break
            
        dx = np.array(x)-np.array(xtemp)
        if np.linalg.norm(dx,1) < 1e-3:
           break
        miu *= mium
        xtemp = copy.deepcopy(x)
    return x
