import matplotlib.pyplot as plt
import numpy as np
import torch

def visualise_darboux(X, Xo, Xu):
    plt.figure(figsize=(16, 12))

    plt.scatter(X[:, 0], X[:, 1], color='gray', alpha=0.1, label='Domain $X$')

    plt.scatter(Xo[:, 0], Xo[:, 1], color='green', s=20, label='Region $X_o$')

    plt.scatter(Xu[:, 0], Xu[:, 1], color='red', s=20, label='Unsafe Region $X_u$')

    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('Darboux Model Domains and Regions')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.savefig("darboux_domain_region.jpg")

def plot_model(model):
    x = np.linspace(-2, 2, 500)
    y = np.linspace(-2, 2, 500)
    X, Y = np.meshgrid(x, y)

    points = np.vstack([X.ravel(), Y.ravel()]).T

    points_tensor = torch.tensor(points, dtype=torch.float32)

    with torch.no_grad():
        B_values = model(points_tensor).numpy()

    B_values = B_values.reshape(X.shape)
    plt.figure(figsize=(16, 12))
    contour = plt.contourf(X, Y, B_values, levels=50, cmap='bwr')
    plt.colorbar(contour)
    plt.title('Neura Barrier Certificate $B(x)$ over the Domain')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.savefig("NBC.jpg")