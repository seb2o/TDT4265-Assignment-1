import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer


def main():
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = 0.1
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = 0.9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_sigmoid = False
    use_improved_weight_init = False
    use_momentum = False
    use_relu = False

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init, use_relu
    )
    trainer = SoftmaxTrainer(
        momentum_gamma,
        use_momentum,
        model,
        learning_rate,
        batch_size,
        shuffle_data,
        X_train,
        Y_train,
        X_val,
        Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    # Example created for comparing with and without shuffling.
    # For comparison, show all loss/accuracy curves in the same plot
    # YOU CAN DELETE EVERYTHING BELOW!

    use_improved_weight_init = True

    # Train a new model with new parameters
    model_iwi = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init, use_relu
    )
    trainer_iwi = SoftmaxTrainer(
        momentum_gamma,
        use_momentum,
        model_iwi,
        learning_rate,
        batch_size,
        shuffle_data,
        X_train,
        Y_train,
        X_val,
        Y_val,
    )
    train_history_iwi, val_history_iwi = trainer_iwi.train(num_epochs)

    use_improved_sigmoid = True

    # Train a new model with new parameters
    model_is = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init, use_relu
    )
    trainer_is = SoftmaxTrainer(
        momentum_gamma,
        use_momentum,
        model_is,
        learning_rate,
        batch_size,
        shuffle_data,
        X_train,
        Y_train,
        X_val,
        Y_val,
    )
    train_history_is, val_history_is = trainer_is.train(num_epochs)

    use_momentum = True
    learning_rate = 0.08
    # Train a new model with new parameters
    model_um = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init, use_relu
    )
    trainer_um = SoftmaxTrainer(
        momentum_gamma,
        use_momentum,
        model_um,
        learning_rate,
        batch_size,
        shuffle_data,
        X_train,
        Y_train,
        X_val,
        Y_val,
    )
    train_history_um, val_history_um = trainer_um.train(num_epochs)

    plt.subplot(1, 2, 1)
    utils.plot_loss(train_history["loss"], "Task 2 Model (T2)", npoints_to_average=10)
    utils.plot_loss(
        train_history_iwi["loss"],
        "T2 + improved weight init (iwi)",
        npoints_to_average=10,
    )
    utils.plot_loss(
        train_history_is["loss"],
        "T2 + iwi + improved sigmoid (is)",
        npoints_to_average=10,
    )
    utils.plot_loss(
        train_history_um["loss"],
        "T2 + iwi + is + momentum",
        npoints_to_average=10,
    )
    plt.ylim([0, 0.4])
    plt.ylabel("Training Loss")
    plt.subplot(1, 2, 2)
    plt.ylim([0.9, 1.0])
    utils.plot_loss(val_history["accuracy"], "Task 2 Model (T2)")
    utils.plot_loss(
        val_history_iwi["accuracy"], "T2 + improved weight init (iwi)"
    )
    utils.plot_loss(
        val_history_is["accuracy"], "T2 + iwi + improved sigmoid (is)"
    )
    utils.plot_loss(
        val_history_um["accuracy"], "T2 + iwi + is + momentum"
    )
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.savefig("task3abc.png")
    plt.show()


if __name__ == "__main__":
    main()
