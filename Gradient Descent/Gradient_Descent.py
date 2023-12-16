import numpy as np
import matplotlib.pyplot as plt


def func2d(x, y):
    return np.sin(x) * np.cos(y)


def gradient_func2d(x, y):
    return np.cos(x) * np.cos(y), - np.sin(x) * np.sin(y)


def iterate_gradient(point, L):
    Xgrad, Ygrad = gradient_func2d(point[0], point[1])
    Xnew, Ynew = point[0] - L * Xgrad, point[1] - L * Ygrad
    return Xnew, Ynew, func2d(Xnew, Ynew)


x = np.arange(-5, 5, 0.1)
y = np.arange(-5, 5, 0.1)
X, Y = np.meshgrid(x, y)

p1 = (1, 0.5)
p2 = (-1, 3)
p3 = (-4, -0.5)

L, epochs = 0.1, 100

ax = plt.subplot(projection="3d", computed_zorder=False)

for i in range(epochs):
    p1 = iterate_gradient(p1, L)
    p2 = iterate_gradient(p2, L)
    p3 = iterate_gradient(p3, L)

    ax.plot_surface(X, Y, func2d(X, Y), cmap="viridis", zorder=0)

    ax.scatter(p1[0], p1[1], func2d(p1[0], p1[1]), c="red", s=25, zorder=1)
    ax.scatter(p2[0], p2[1], func2d(p2[0], p2[1]), c="green", s=25, zorder=1)
    ax.scatter(p3[0], p3[1], func2d(p3[0], p3[1]), c="cyan", s=25, zorder=1)

    ax.set_title(f"Epoch No: {i}")

    plt.pause(0.001)
    ax.clear()

