import json
import os
import csv
from datetime import datetime
import torch
import logging


def startup():
    cwd = os.getcwd()

    results_dir = os.path.join(cwd, 'results')
    now = datetime.now().isoformat().replace(':', '-')

    current_run_dir = os.path.join(results_dir, str(now))

    images_dir = os.path.join(current_run_dir, 'images')
    models_dir = os.path.join(current_run_dir, 'models')

    for directory in [results_dir, current_run_dir, images_dir, models_dir]:
        os.makedirs(directory, exist_ok=True)

    return images_dir, models_dir, current_run_dir




def load_config(config_path='config.json'):
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    return config


def save_results_to_csv(results, results_dir):
    csv_path = os.path.join(results_dir, "model_evaluation_results.csv")

    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(['Tauo', 'Tauu', 'Taud', 'Xo_Probability', 'Xu_Probability'])

        for result in results:
            writer.writerow(result)


def save_model(model, model_dir, tau_value):
    model_path = os.path.join(model_dir,
                              f"model_{tau_value['tauo']}_Tauu_{tau_value['tauu']}_Taud_{tau_value['taud']}.pth")
    torch.save(model.state_dict(), model_path)


def evaluate_model_conditions(model, Xo, Xu, device):
    Xo_tensor = torch.tensor(Xo, dtype=torch.float32).to(device)
    Xu_tensor = torch.tensor(Xu, dtype=torch.float32).to(device)

    model.eval()

    with torch.no_grad():
        Bo = model(Xo_tensor).squeeze()
        Bu = model(Xu_tensor).squeeze()

    Xo_prob = (Bo <= 0).float().mean().item()
    Xu_prob = (Bu > 0).float().mean().item()

    return Xo_prob, Xu_prob
