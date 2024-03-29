import os
import torch
import datagen
import utils
from architecture import NeuralBarrierCertificate

model_directory = '/Users/vangs/cub/Research/neural-barriers/results/2024-03-22T13:04:12.285251Z/models'
result_directory = '/Users/vangs/cub/Research/neural-barriers/results/2024-03-22T13:04:12.285251Z/test_results'


def test():
    X, Xo, Xu, labels = datagen.generate_darboux(3000)
    results = []
    config = utils.load_config()
    for tau_value in config['tau_values']:
        filename = f"model_{tau_value['tauo']}_Tauu_{tau_value['tauu']}_Taud_{tau_value['taud']}.pth"
        path = os.path.join(model_directory, filename)
        model = NeuralBarrierCertificate(X, labels, input_size=config["input_size"],
                                         count=config["count"],
                                         order=config["order"],
                                         output_size=config["output_size"],
                                         learning_rate=config["learning_rate"],
                                         batch_size=config["batch_size"], results_dir=model_directory)
        model.load_state_dict(torch.load(path))

        Xo_prob, Xu_prob = utils.evaluate_model_conditions(model, Xo, Xu)

        results.append([tau_value['tauo'], tau_value['tauu'], tau_value['taud'], Xo_prob, Xu_prob])

    os.makedirs(result_directory, exist_ok=True)
    utils.save_results_to_csv(results, result_directory)


if __name__ == "__main__":
    test()
