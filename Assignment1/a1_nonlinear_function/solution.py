

import sys
sys.path.append("../..")
from optalg.interface.nlp import NLP
import numpy as np


class NLP_nonlinear(NLP):
    """
    Nonlinear program with cost  1 / || C x ||
    x in R^n
    C in R^(m x n)
    || . || is the 2-norm
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

        Cx = np.dot(self.C, x)
        norm_Cx = np.linalg.norm(Cx)
        if norm_Cx != 0:
            y = np.array([1 / norm_Cx])  
        else:
            y = np.array([np.inf])

        if norm_Cx != 0:
            J = -np.dot(self.C.T, Cx) / (norm_Cx ** 3)
            J = np.array(J)  
        else:
            J = np.zeros_like(x)  


        return y, J

    def getDimension(self):
        """
        Returns the dimensionality of the variable x

        See Also
        ------
        NLP.getDimension
        """

        n = self.C.size[1]
        return n

    def getFHessian(self, x):
        """
        Returns the hessian of the cost term.
        The output should be:
            H: the hessian (2-D np.ndarray of shape (n,n))

        See Also
        ------
        NLP.getFHessian
        """
        # add code to compute the Hessian matrix
        Cx = np.dot(self.C, x)
        norm_Cx = np.linalg.norm(Cx)

        if norm_Cx != 0:
            term1 = -np.dot(self.C.T, self.C) / (norm_Cx ** 3)
            term2 = 3 * np.outer(np.dot(self.C.T, Cx), np.dot(self.C.T, Cx)) / (norm_Cx ** 5)
            H = term1 + term2
        else:
            H = np.zeros((len(x), len(x)))  

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
        return "Nonlinear function  1 / || C x ||"
