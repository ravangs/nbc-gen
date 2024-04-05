from datagen import generate_darboux
from visualise import visualise_darboux, plot_model
from utils import load_config, startup, save_results_to_csv, save_model, evaluate_model_conditions
from architecture import NeuralBarrierCertificate
import torch


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    image_dir, model_dir, results_dir = startup()

    X, Xo, Xu, labels = generate_darboux(500)

    visualise_darboux(image_dir, X, Xo, Xu)

    config = load_config()

    results = []
    for tau_value in config['tau_values']:
        print(f'Tau value: {tau_value}')
        model = NeuralBarrierCertificate(X, labels, input_size=config["input_size"],
                                         count=config["count"],
                                         order=config["order"],
                                         output_size=config["output_size"],
                                         learning_rate=config["learning_rate"],
                                         batch_size=config["batch_size"], results_dir=results_dir).to(device)

        model.train()
        model.train_model(epochs=config["epochs"], tauo=tau_value['tauo'], tauu=tau_value['tauu'],
                          taud=tau_value['taud'], device=device)

        save_model(model, model_dir, tau_value)

        model.eval()
        plot_model(model, image_dir, tau_value, device)

        Xo_prob, Xu_prob = evaluate_model_conditions(model, Xo, Xu, device)

        results.append([tau_value['tauo'], tau_value['tauu'], tau_value['taud'], Xo_prob, Xu_prob])

    save_results_to_csv(results, results_dir)


if __name__ == "__main__":
    main()

# %%
