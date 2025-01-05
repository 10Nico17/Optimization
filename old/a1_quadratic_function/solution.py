import sys
sys.path.append("")


from optalg.interface.nlp import NLP
import numpy as np
import optalg.utils.finite_diff as fd


class NLP_xCCx(NLP):
    """
    Nonlinear program with quadratic cost  x^T C^T C x
    x in R^n
    C in R^(m x n)
    ^T means transpose
    feature types: [ OT.f ]

    """

    def __init__(self, C):
        """
        """
        self.C = C

    def evaluate(self, x):
        """
        Returns the features and the Jacobians
        of a nonlinear program.
        In this case, we have a single feature (the cost function)
        because there are no constraints or residual terms.
        Therefore, the output should be:
            y: the feature (1-D np.ndarray of shape (1,)) 
            J: the jacobian (2-D np.ndarray of shape (1,n))
        See also:
        ----
        NLP.evaluate
        """
        dm_1 = np.shape(self.C)[0]
        dm_2 = np.shape(self.C)[1]
        dm_x = np.shape(x)[0]
        C_0 = self.C
        if dm_2 != dm_x:
            C_0 = np.transpose(self.C)
        x_t = x
        x = np.transpose(x)
        C_t = np.transpose(C_0)
        y = x_t@C_t@C_0@x
        J = 2*C_t@C_0@x
        # y =
        # J =

        # return  y , J
        return np.array([y]), np.array([J])

    def getDimension(self):
        """
        Return the dimensionality of the variable x

        See Also
        ------
        NLP.getDimension
        """
        # n =
        # return n

    def getFHessian(self, x):
        """
        Returns the hessian of the cost term.
        The output should be: 
            H: the hessian (2-D np.ndarray of shape (n,n))

        See Also
        ------
        NLP.getFHessian
        """
        #x_t = np.transpose(x)
        C_t = np.transpose(self.C)
        dm = np.shape(x)[0]
        comatrix = 2*C_t@self.C
        Hess = np.zeros((dm, dm))
        for i in range(dm):
            for j in range(dm):
                Hess[i, j] = comatrix[i, j]
        H = Hess
        return H

    def getInitializationSample(self):
        """
        See Also
        ------
        NLP.getInitializationSample
        """
        return np.ones(self.getDimension())

    def report(self, verbose):
        """
        See Also
        ------
        NLP.report
        """
        return "Quadratic function x^T C^T C x "
