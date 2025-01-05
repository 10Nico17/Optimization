import sys
sys.path.append("../..")
from optalg.interface.nlp_stochastic import NLP_stochastic
import numpy as np
import copy

def solve(nlp: NLP_stochastic):
    """
    stochastic gradient descent


    Arguments:
    ---
        nlp: object of class NLP_stochastic that contains one feature of type OT.f.

    Returns:
    ---
        x: local optimal solution (1-D np.ndarray)

    Task:
    ---
    See the coding assignment PDF in ISIS.

    Notes:
    ---

    Get the starting point with:
    x = nlp.getInitializationSample()

    Get the number of samples with:
    N = nlp.getNumSamples()

    You can query the problem with any index i=0,...,N (N not included)

    y, J = nlp.evaluate_i(x, i)

    As usual, get the cost function (scalar) and gradient (1-D np.array) with y[0] and J[0]

    The output (y,J) is different for different values of i and x.

    The expected value (over i) of y,J at a given x is SUM_i [ nlp.evaluate_i(x, i) ]  / N

    """

    x = nlp.getInitializationSample()
    N = nlp.getNumSamples()
    alpha = 1.2
    lamda = 0.5
    

    #
    # Write your code Here
    #
    seq = range(1000)
    perm = range(N)
    perm1 = np.array(perm)
    qls = 0.01
    a_ls = 1
    qap = 1.2
    qam = 0.5
    i_list = []
    k = 1

    def linesearch(f: callable,
                    x: np.array,
                    a_ls: float,
                    delta: np.array,
                    qam: float,
                    qap: float,
                    i: int,
                    y: float,
                    J):
        while True:
            left_t = f(x + a_ls*delta, i)[0][0]
            right_t = qls*a_ls*(J[0])@(delta) + y[0]
            if left_t <= right_t:
                break
            a_ls *= qam
        x = x + a_ls*delta
        a_ls = min(a_ls*qap, 1)
        return x, a_ls*delta, J[0]

    while True:
        i = np.random.randint(len(perm1))
        ii = perm1[i]
        y, J = nlp.evaluate_i(x, ii)
        # method = 2
        # match method:
            # case 1:
            #     delta = -J[0]
            #     x, incr, JJ = linesearch(nlp.evaluate_i,x,a_ls, delta, qam, qap,ii,y, J)
            #     if np.linalg.norm(JJ, np.inf) < 1e-2:
            #         ind = np.where(perm1 == ii)[0]
            #         perm1 = np.delete(perm1, ind)
            #         i_list.append(copy.copy(ii))
                
            #     #cost = (1 / N) * np.sum(np.array([nlp.evaluate_i(x, i)[0][0] for i in range(N)]))
            #     #cost_J = (1 / N) * np.sum(np.array([nlp.evaluate_i(x, i)[1][0] for i in range(N)]), axis = 0)
            #     # if np.linalg.norm(cost_J, np.inf) < 1e-4:
            #     #     break
            #     if len(perm1) == 0:
            #         break
            # case 2:
        ak = alpha / (1 + alpha*lamda*k)
        x = x - ak*J[0]
        if np.linalg.norm(J[0], np.inf) < 1e-2:
            ind = np.where(perm1 == ii)[0]
            perm1 = np.delete(perm1, ind)
            i_list.append(copy.copy(ii))
        k += 1
        if (len(perm1) == 0) or(k > 10000):
            break
            # case 3:
            #     ak = alpha / (1 + alpha*lamda*k)
            #     x = x - ak*J[0]
            #     # cost_J = (1 / N) * np.sum(np.array([nlp.evaluate_i(x, i)[1][0] for i in range(N)]), axis = 0)
            #     # if np.linalg.norm(cost_J, np.inf) < 1e-3:
            #     #     break
            #     k += 1
            #     if k > 100000:
            #         break


    #return x
    return x
