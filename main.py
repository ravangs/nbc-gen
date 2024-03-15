from datagen import generate_darboux
from visualise import visualise_darboux, plot_model
from utils import load_config
from architecture import NeuralBarrierCertificate


def main():
    X, Xo, Xu, labels = generate_darboux(10000)

    visualise_darboux(X, Xo, Xu)

    config = load_config()

    model = NeuralBarrierCertificate(X, labels, input_size=config["input_size"],
                                     count=config["count"],
                                     order=config["order"],
                                     output_size=config["output_size"],
                                     learning_rate=config["learning_rate"])

    model.train()
    model.train_model(epochs=config["epochs"])
    model.eval()
    plot_model(model)




if __name__ == "__main__":
    main()

# %%
