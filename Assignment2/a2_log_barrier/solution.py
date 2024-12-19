import sys  # noqa
sys.path.append("../..")  # noqa

import numpy as np
from optalg.interface.objective_type import OT
from optalg.interface.nlp import NLP


def solve(nlp: NLP, Dout={}):
    """
    Solver für constrained optimization (Kostenfunktion und Ungleichungen).
    """

    def compute_barrier(x, mu, compute_gradient_and_hessian=False):
        """
        Berechnet den Wert der Barrierefunktion und optional Gradienten und Hesse-Matrix.
        """
        values, gradients = nlp.evaluate(x)
        cost_value = values[cost_index[0]]
        constraint_values = values[constraint_indices]

        barrier_value = cost_value - mu * np.sum(np.log(-constraint_values))

        if not compute_gradient_and_hessian:
            return barrier_value

        cost_gradient = gradients[cost_index[0]]
        constraint_gradients = gradients[constraint_indices]
        barrier_gradient = cost_gradient - mu * np.sum(
            constraint_gradients / constraint_values[:, np.newaxis], axis=0
        )

        hessian_cost = nlp.getFHessian(x)
        barrier_hessian = hessian_cost + mu * sum(
            np.outer(constraint_gradients[i], constraint_gradients[i]) / constraint_values[i]**2
            for i in range(len(constraint_indices))
        )

        return barrier_value, barrier_gradient, barrier_hessian

    def compute_constraints(x):
        """
        Gibt die Werte der Ungleichungs-Nebenbedingungen zurück.
        """
        values, _ = nlp.evaluate(x)
        return values[constraint_indices]

    def solve_inner_problem(x, mu):
        """
        Minimiert die Barrierefunktion für eine feste Mu.
        """
        alpha = 1
        max_iterations = 1000
        convergence_threshold = 1e-6
        alpha_decrease_factor = 0.5
        alpha_increase_factor = 1.2
        line_search_tolerance = 0.01

        for _ in range(max_iterations):
            barrier_val, barrier_grad, barrier_hessian = compute_barrier(x, mu, True)
            try:
                search_direction = np.linalg.solve(barrier_hessian, -barrier_grad)
                if barrier_grad.T @ search_direction > 0:
                    raise np.linalg.LinAlgError
            except np.linalg.LinAlgError:
                search_direction = -barrier_grad

            # Line search for feasibility and sufficient decrease
            while np.any(compute_constraints(x + alpha * search_direction) >= 0):
                alpha *= alpha_decrease_factor

            while (
                compute_barrier(x + alpha * search_direction, mu) > 
                barrier_val + line_search_tolerance * alpha * barrier_grad.T @ search_direction
            ):
                alpha *= alpha_decrease_factor

            Dout["dx"].append(search_direction)
            x += alpha * search_direction
            alpha = min(1, alpha_increase_factor * alpha)

            if np.linalg.norm(alpha * search_direction) < convergence_threshold:
                return x
        return x

    # Initialisierung
    x = nlp.getInitializationSample()
    feature_types = nlp.getFeatureTypes()
    cost_index = [i for i, t in enumerate(feature_types) if t == OT.f]
    constraint_indices = [i for i, t in enumerate(feature_types) if t == OT.ineq]

    Dout["x_steps"] = []
    Dout["dx"] = []
    penalty_parameter = 1
    penalty_reduction_factor = 0.5
    termination_tolerance = 1e-6
    max_outer_iterations = 100

    # Äußere Schleife für die Barriere-Methode
    for _ in range(max_outer_iterations):
        previous_x = x.copy()
        x = solve_inner_problem(x, penalty_parameter)
        penalty_parameter *= penalty_reduction_factor

        if np.linalg.norm(x - previous_x) < termination_tolerance:
            return x

    return x