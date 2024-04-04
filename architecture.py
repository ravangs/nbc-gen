import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from log_config import Logger
import os


class PolyActivation(nn.Module):
    def __init__(self, count, order):
        super(PolyActivation, self).__init__()
        self.count = count
        self.order = order

    def forward(self, x):
        outputs = []
        for i in range(1, self.order + 1):
            # order of i for the ith slice - 1 based index
            expanded = x[:, (i - 1) * self.count:i * self.count] ** i
            outputs.append(expanded)
        return torch.cat(outputs, dim=1)


def prepare_data(X, labels):
    X_safe = X[labels == 1]
    X_unsafe = X[labels == 2]
    X_other = X[labels == 0]

    return X_safe, X_unsafe, X_other


class NeuralBarrierCertificate(nn.Module):
    def __init__(self, X, labels, input_size, count, order, output_size, learning_rate, batch_size, results_dir):
        super(NeuralBarrierCertificate, self).__init__()
        self.X_safe, self.X_unsafe, self.X_other = prepare_data(X, labels)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        log_file = os.path.join(results_dir,"training.log")
        logger_class = Logger(log_name="training", log_file=log_file)
        self.logger = logger_class.get_logger()

        neuron_count = count * order

        # layers
        self.fc1 = nn.Linear(input_size, neuron_count)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.01)
        self.fc2 = nn.Linear(neuron_count, neuron_count)
        self.poly_act = PolyActivation(count, order)
        self.fc3 = nn.Linear(neuron_count, neuron_count)
        self.leaky_relu3 = nn.LeakyReLU(negative_slope=0.01)
        self.fc_out = nn.Linear(neuron_count, output_size)

        self.optimizer = None

    def forward(self, x):
        x = self.leaky_relu1(self.fc1(x))
        x = self.poly_act(self.fc2(x))
        x = self.leaky_relu3(self.fc3(x))
        x = self.fc_out(x)
        return x

    def configure_optimizer(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def train_model(self, epochs, tauo, tauu, taud, device):
        self.logger.info(f"Training model with Tauo = {tauo}, Tauu = {tauu}, Taud = {taud}")
        if self.optimizer is None:
            self.configure_optimizer()

        safe_dataset = TensorDataset(torch.tensor(self.X_safe, dtype=torch.float32),
                                     torch.ones(len(self.X_safe), dtype=torch.long))
        unsafe_dataset = TensorDataset(torch.tensor(self.X_unsafe, dtype=torch.float32),
                                       torch.ones(len(self.X_unsafe), dtype=torch.long) * 2)
        other_dataset = TensorDataset(torch.tensor(self.X_other, dtype=torch.float32),
                                      torch.zeros(len(self.X_other), dtype=torch.long))  # Example

        loaders = {
            'safe': DataLoader(safe_dataset, batch_size=self.batch_size, shuffle=True),
            'unsafe': DataLoader(unsafe_dataset, batch_size=self.batch_size, shuffle=True),
            'other': DataLoader(other_dataset, batch_size=self.batch_size, shuffle=True)
        }

        for epoch in range(epochs):
            total_loss = 0

            for group_type, loader in loaders.items():
                for inputs, targets in loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    self.optimizer.zero_grad()
                    loss = self.nbc_loss_enhanced(inputs, group_type, tauo=tauo, tauu=tauu, taud=taud)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()
            if epoch % (epochs / 2) == 0:
                self.logger.info(f'Epoch {epoch + 1}, Combined Loss: {total_loss:.4f}')

            if epoch == epochs-1:
                self.logger.info(f"Final Loss: {total_loss:.4f}")

    def compute_B_dot(self, x):
        x.requires_grad_(True)

        B_x = self.forward(x)

        x_dot = x[:, 1] + 2 * x[:, 0] * x[:, 1]
        y_dot = -x[:, 0] + 2 * x[:, 0] ** 2 - x[:, 1] ** 2

        # Compute gradients of B with respect to x (dB/dx)
        grads = torch.autograd.grad(B_x, x, grad_outputs=torch.ones_like(B_x), create_graph=True)[0]

        # Compute the time derivative of B using the chain rule (B_dot)
        B_dot = torch.sum(grads * torch.stack([x_dot, y_dot], dim=1), dim=1, keepdim=True)

        return B_dot

    def sat_relu(self, x, saturation_limit=1.0):
        relu = F.relu(x)
        return torch.clamp(relu, max=saturation_limit)

    def nbc_loss_enhanced(self, x, group_type, tauo=-0.05, tauu=0.15, taud=-0.001, alpha=0.01, beta=1.0):
        output = self(x)
        B_dot = self.compute_B_dot(x)

        loss = 0

        if group_type == 'safe':
            loss += torch.mean(F.relu(output - tauo) - alpha * self.sat_relu(-output + tauo, beta))
        elif group_type == 'unsafe':
            loss += torch.mean(F.relu(-output - tauu) - alpha * self.sat_relu(output + tauu, beta))

        # loss += torch.mean(-self.sat_relu(-B_dot + taud, beta))

        loss += torch.mean(torch.maximum(taud * torch.ones_like(B_dot), B_dot))

        return loss
