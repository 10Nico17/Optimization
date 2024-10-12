import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definiere die Funktion f(x1, x2) = x1^2 + x2^2
def func(x1, x2):
    return x1**2 + x2**2

# Definiere den Gradient der Funktion f(x1, x2)
def gradient(x1, x2):
    grad_x1 = 2 * x1
    grad_x2 = 2 * x2
    return np.array([grad_x1, grad_x2])

# Berechne die Richtungsableitung in einer bestimmten Richtung delta
def directional_derivative(f_grad, delta):
    # Normalisiere den Richtungsvektor delta
    delta_normalized = delta / np.linalg.norm(delta)
    # Berechne das Skalarprodukt von f_grad und delta
    return np.dot(f_grad, delta_normalized)

# Beispielpunkt, an dem wir den Gradienten berechnen
x_point = np.array([1.0, 2.0])

# Richtung delta, in der wir die Ableitung berechnen möchten
delta_direction = np.array([1.0, 1.0])

# Berechne den Gradienten an x_point
f_grad = gradient(x_point[0], x_point[1])

# Berechne die Richtungsableitung in Richtung delta
dir_deriv = directional_derivative(f_grad, delta_direction)

# 3D Plot mit der Funktion und der Richtungsableitung

# Erstelle ein Gitter für die x1- und x2-Werte
x1 = np.linspace(-3, 3, 100)
x2 = np.linspace(-3, 3, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = func(X1, X2)

# Erstelle den 3D-Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot der Oberfläche der Funktion f(x1, x2)
ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7)

# Punkt, an dem der Gradient berechnet wird
point_x1, point_x2 = x_point[0], x_point[1]
point_z = func(point_x1, point_x2)

# Plot des Punktes auf der Oberfläche
ax.scatter(point_x1, point_x2, point_z, color='red', s=100, label='Point (1, 2)')

# Gradient am Punkt (1, 2)
f_grad = gradient(point_x1, point_x2)

# Richtungsvektor normalisieren
delta_direction_normalized = delta_direction / np.linalg.norm(delta_direction)

# Zeige den Richtungsvektor in 3D
ax.quiver(point_x1, point_x2, point_z, delta_direction_normalized[0], delta_direction_normalized[1], 0, 
          color='blue', length=1.5, label='Direction of Derivative')

# Titel und Beschriftungen
ax.set_title('3D Plot of $f(x_1, x_2) = x_1^2 + x_2^2$ and Direction of Derivative')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('f(X1, X2)')
ax.legend()

plt.show()

# Ausgabe des berechneten Gradienten und der Richtungsableitung
print(f"Gradient an Punkt {x_point}: {f_grad}")
print(f"Richtungsableitung in Richtung {delta_direction}: {dir_deriv}")
