import sys
sys.path.append("../..")
from optalg.interface.objective_type import OT
from optalg.interface.nlp import NLP
import numpy as np


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

    def compute_metrics(point):
        feature_types = nlp.getFeatureTypes()
        values, jacobian = nlp.evaluate(point)
        hessian = nlp.getFHessian(point)
        index_f = [i for i, t in enumerate(feature_types) if t == OT.f]
        index_r = [i for i, t in enumerate(feature_types) if t == OT.r]
        residual_values = values[index_r]
        cost_value = values[index_f]
        residual_jacobian = jacobian[index_r]
        cost_jacobian = jacobian[index_f]
        total_cost = values[index_f[0]] + np.dot(values[index_r], values[index_r])
        gradient = cost_jacobian + 2 * np.transpose(residual_jacobian) @ residual_values
        total_hessian = hessian + 2 * (np.transpose(residual_jacobian) @ residual_jacobian)
        return total_cost, gradient, total_hessian

    damping = 0.1
    step_increase = 1.2
    step_decrease = 0.5
    step_quality = 0.01
    step_size = 1

    while True:
        total_cost, gradient, hessian = compute_metrics(x)
        identity_matrix = np.identity(np.shape(hessian)[0])
        step = np.linalg.inv(hessian + damping * identity_matrix) @ (np.reshape(-gradient, (-1, 1)))
        step = np.squeeze(np.reshape(step, (1, -1)))
        while True:
            candidate = x + step * step_size
            candidate_cost = compute_metrics(x + step * step_size)[0]
            expected_cost = total_cost + step_quality * step_size * (np.squeeze(gradient.T)) @ (step)
            if candidate_cost <= expected_cost:
                break
            step_size *= step_decrease
        x += step_size * step
        step_size = min(step_increase * step_size, 1)
        if np.linalg.norm(step_size * step, np.inf) < 1e-5:
            break

    return x