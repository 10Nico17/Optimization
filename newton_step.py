# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definieren der mehrdimensionalen Funktion
def f(x):
    return x[0]**2 + x[1]**2 + 3 * x[0] * x[1] + 4 * x[0] + 5 * x[1] + 7

# Berechnung des Gradienten von f
def gradient_f(x):
    grad = np.zeros_like(x)
    grad[0] = 2 * x[0] + 3 * x[1] + 4
    grad[1] = 2 * x[1] + 3 * x[0] + 5
    return grad

# Berechnung der Hessischen Matrix von f
def hessian_f(x):
    hess = np.zeros((2, 2))
    hess[0, 0] = 2  # zweite Ableitung nach x[0]
    hess[1, 1] = 2  # zweite Ableitung nach x[1]
    hess[0, 1] = 3  # gemischte Ableitung nach x[0] und x[1]
    hess[1, 0] = 3  # gemischte Ableitung nach x[1] und x[0]
    return hess

# Wähle einen Punkt x0
x0 = np.array([1.0, 2.0])

# Berechne die Hessische Matrix und den Gradient bei x0
hess_matrix = hessian_f(x0)
grad = gradient_f(x0)

# Berechne die Eigenwerte der Hessischen Matrix
eigenvalues = np.linalg.eigvals(hess_matrix)

# Berechne den Newton-Schritt: (Hessian)^(-1) * (-Gradient)
newton_step = -np.linalg.inv(hess_matrix).dot(grad)

# Definieren der Funktion f für das Plotten
def f_plot(x0, x1):
    return x0**2 + x1**2 + 3 * x0 * x1 + 4 * x0 + 5 * x1 + 7

# Erstelle ein Gitter zum Plotten der Funktion
x0_vals = np.linspace(-5, 5, 100)
x1_vals = np.linspace(-5, 5, 100)
X0, X1 = np.meshgrid(x0_vals, x1_vals)
Z = f_plot(X0, X1)

# Plot der Funktion
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X0, X1, Z, cmap='viridis', alpha=0.7)

# Markiere den Startpunkt
ax.scatter(x0[0], x0[1], f_plot(x0[0], x0[1]), color='r', s=50, label="Startpunkt (x0)")

# Berechne den neuen Punkt
next_point = x0 + newton_step

# Zeichne eine Linie zwischen Startpunkt und neuem Punkt
ax.plot([x0[0], next_point[0]], [x0[1], next_point[1]],
        [f_plot(x0[0], x0[1]), f_plot(next_point[0], next_point[1])],
        color='r', label="Newton-Schritt")

# Markiere den neuen Punkt
ax.scatter(next_point[0], next_point[1], f_plot(next_point[0], next_point[1]), color='b', s=50, label="Neuer Punkt")

# Achsen und Labels
ax.set_xlabel('x0')
ax.set_ylabel('x1')
ax.set_zlabel('f(x0, x1)')
ax.set_title('Newton-Schritt auf der Funktion f(x0, x1)')

plt.legend()
plt.show()

# Print out results for Hessian, eigenvalues, and newton step
print("Hessische Matrix:\n", hess_matrix)
print("\nEigenwerte der Hessischen Matrix:\n", eigenvalues)
print("\nNewton-Schritt:\n", newton_step)
