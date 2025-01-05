import sys
sys.path.append("../..")
from optalg.interface.nlp_stochastic import NLP_stochastic
import numpy as np
import copy

def solve(nlp: NLP_stochastic):
    """
    stochastic gradient descent -- ADAM


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
    alpha = 1e-3
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    m0 = 0
    v0 = 0
    t = 0
    perm = range(N)
    perm1 = np.array(perm)
    i_list = []
    speedup = 0
    while t <= 9999:
        if speedup == 1:
            if len(perm1) == 0:
                break
            i = np.random.randint(len(perm1))
            ii = perm1[i]
            t += 1
            gt = nlp.evaluate_i(x, ii)[1][0]
            
            if np.linalg.norm(gt, np.inf) < 1e-5:
                ind = np.where(perm1 == ii)[0]
                perm1 = np.delete(perm1, ind)
                i_list.append(copy.copy(ii))
        else:
            i = np.random.randint(len(perm))
            gt = nlp.evaluate_i(x, i)[1][0]
            t += 1
        mt = beta1*m0 + (1 - beta1)*gt
        vt = beta2*v0 + (1 - beta2)*(gt**2)
        mtr = mt / (1 - (beta1**t))
        vtr = vt / (1 - (beta2**t))
        x = x - alpha * mtr / (np.sqrt(vtr) + epsilon)
        m0 = mt
        v0 = vt

    #
    # Write your code Here
    #



    #return x, i_list, t
    return x