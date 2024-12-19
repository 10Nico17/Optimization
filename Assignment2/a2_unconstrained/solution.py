import sys  # noqa
sys.path.append("../..")  # noqa

import numpy as np
from optalg.interface.nlp import NLP
from optalg.interface.objective_type import OT


def solve(nlp: NLP, Dout={}):
    """
    Solver for unconstrained optimization


    Arguments:
    ---
        nlp: object of class NLP that only contains one feature of type OT.f.

    Returns:
        x: local optimal solution (1-D np.ndarray)


    Task:
    ---

    See instructions and requirements in the coding assignment PDF in ISIS.

    Notes:
    ---

    Get the starting point with:
    x = nlp.getInitializationSample()

    Use the following to query the function and gradient of the problem:
    phi, J = nlp.evaluate(x)

    phi is a vector (1D np.ndarray); use phi[0] to access the cost value
    (a float number).

    J is a Jacobian matrix (2D np.ndarray). Use J[0] to access the gradient
    (1D np.array) of phi[0].

    Use getFHessian to query the Hessian.

    H = nlp.getFHessian(x)

    H is a matrix (2D np.ndarray) of shape n x n.


    You can use Dout to store any information you want during the computation,
    to analyze and debug your code.
    For instance, you can store the value of the cost function at each
    iteration, the variable x,...


    Dout["xs"] = []
    Dout["f"] = []

    ...

    Dout["x"].append(np.copy(x))
    Dout["f"].append(f)


    Do not use the assignment operator Dout = { ... }  in your code,
    just use Dout[...] = ...,
    otherwise you will not be able to access the information
    outside the solver.

    In test file, we call solve only with one argument, but it is fine
    if your code actually stores information in Dout.

    """

    # sanity check on the input nlp
    assert len(nlp.getFeatureTypes()) == 1
    assert nlp.getFeatureTypes()[0] == OT.f

    # get start point
    x = nlp.getInitializationSample()

    # Comment/Uncomment if you want to store information in Dout
    #Dout["xs"] = []
    #Dout["x0"] = np.copy(x)
    #Dout["xs"].append(np.copy(x))

    tolerance = 0.01
    step_decay = 0.5
    max_line_search_iters = 20
    step_size = 1.0  

    x_prev = np.copy(x)

    for iteration in range(100):
        cost, gradient_matrix = nlp.evaluate(x)
        hessian_matrix = nlp.getFHessian(x)
        initial_cost = cost[0]  
        gradient = gradient_matrix[0]  # gradient vector
        line_search_success = False
        line_search_iter = 0
        step_size = 1.0  # reset step size

        search_direction = - np.dot(np.linalg.inv(hessian_matrix), gradient)  # Newton step

        if(np.all((gradient.T @ search_direction) > 0)):
            search_direction = - gradient  # fallback to gradient descent
        else:
            search_direction = - np.dot(np.linalg.inv(hessian_matrix), gradient)

        while not line_search_success and line_search_iter < max_line_search_iters:
            # newton step
            trial_x = x + step_size * search_direction
            cost_trial, _ = nlp.evaluate(trial_x)
            new_cost = cost_trial[0]
            if (new_cost - initial_cost < tolerance * np.dot(gradient, step_size * search_direction)):
                line_search_success = True
                x = np.copy(trial_x)

            line_search_iter += 1
            step_size *= step_decay

        if np.linalg.norm(x - x_prev) < 0.0001:
            break

        x_prev = np.copy(x)

    return x
