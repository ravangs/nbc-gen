import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
import os


def system(t, Z):
    x, y = Z
    dxdt = y + 2 * x * y
    dydt = -x + 2 * x ** 2 - y ** 2
    return [dxdt, dydt]


def visualise_darboux(image_dir, X, Xo, Xu):
    plt.figure(figsize=(16, 12))

    x, y = np.meshgrid(np.linspace(-2, 2, 20), np.linspace(-2, 2, 20))

    dx, dy = system(None, [x, y])

    plt.quiver(x, y, dx, dy, color='black')

    plt.scatter(X[:, 0], X[:, 1], color='gray', alpha=0.1, label='Domain $X$')

    plt.scatter(Xo[:, 0], Xo[:, 1], color='green', s=20, label='Region $X_o$')

    plt.scatter(Xu[:, 0], Xu[:, 1], color='red', s=20, label='Unsafe Region $X_u$')

    plt.xlabel('$x$')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.ylabel('$y$')
    plt.title('Darboux Model Domains and Regions')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    image_path = os.path.join(image_dir, "darboux_domain_region.jpg")
    plt.savefig(image_path)


def plot_model(model, image_dir, tau_value, device):
    x = np.linspace(-2, 2, 500)
    y = np.linspace(-2, 2, 500)
    X, Y = np.meshgrid(x, y)

    points = np.vstack([X.ravel(), Y.ravel()]).T
    points_tensor = torch.tensor(points, dtype=torch.float32).to(device)

    with torch.no_grad():
        B_values = (model(points_tensor)).cpu().numpy()

    B_values = B_values.reshape(X.shape)

    max_abs_value = np.max(np.abs(B_values))
    norm = mcolors.TwoSlopeNorm(vmin=-max_abs_value, vcenter=0, vmax=max_abs_value)

    plt.figure(figsize=(16, 12))
    contour = plt.contourf(X, Y, B_values, levels=50, cmap='bwr', norm=norm)
    plt.colorbar(contour)
    plt.title(f"Neural Barrier Certificate $B(x)$ over the Domain with {tau_value}")
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.grid(True, linestyle='--', alpha=0.5)
    image_path = os.path.join(image_dir,
                              f"NBC_Tauo_{tau_value['tauo']}_Tauu_{tau_value['tauu']}_Taud_{tau_value['taud']}.jpg")
    plt.savefig(image_path)
