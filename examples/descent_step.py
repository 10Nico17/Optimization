import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Die Rosenbrock-Funktion
def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

# Der Gradient der Rosenbrock-Funktion
def rosenbrock_grad(x):
    grad = np.zeros_like(x)
    grad[0] = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    grad[1] = 200 * (x[1] - x[0]**2)
    return grad

# Backtracking Line Search
def backtracking_line_search(f, grad_f, x, alpha=1.0, rho=0.8, c=1e-4):
    while f(x - alpha * grad_f(x)) > f(x) - c * alpha * np.dot(grad_f(x), grad_f(x)):
        alpha *= rho
    return alpha

# Gradient Descent Solver
def gradient_descent_solver(f, grad_f, x0, tol=1e-5, max_iters=1000):
    x = x0
    path = [x0]  # Speichert die Schritte für die Visualisierung
    for i in range(max_iters):
        grad = grad_f(x)
        
        if np.linalg.norm(grad) < tol:
            print(f'Konvergiert nach {i} Iterationen')
            break
        
        alpha = backtracking_line_search(f, grad_f, x)
        
        # Update: Gehe einen Schritt in Richtung des Gradienten
        x = x - alpha * grad
        path.append(x)  # Füge den neuen Punkt zum Pfad hinzu
        
    return np.array(path)

# Visualisiere die Rosenbrock-Funktion und die Schritte des Gradient Descent
def plot_rosenbrock_with_steps(path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Erstelle ein Gitter von x und y Werten
    x = np.linspace(-2, 2, 400)
    y = np.linspace(-1, 3, 400)
    X, Y = np.meshgrid(x, y)
    Z = (1 - X)**2 + 100 * (Y - X**2)**2

    # Plotte die Rosenbrock-Funktion
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)

    # Pfad des Gradient Descent
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], rosenbrock(path.T), 'r', marker='o', markersize=5, label="Gradientenschritte")

    # Beschriftungen
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    ax.set_title("Gradient Descent auf der Rosenbrock-Funktion")
    plt.legend()
    plt.show()

# Startpunkt
x0 = np.array([-1.5, 2.0])

# Führe Gradient Descent Solver aus
path = gradient_descent_solver(rosenbrock, rosenbrock_grad, x0)

# Visualisiere die Rosenbrock-Funktion und die Gradientenschritte
plot_rosenbrock_with_steps(path)
