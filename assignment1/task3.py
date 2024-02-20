import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images
from trainer import BaseTrainer
from task3a import cross_entropy_loss, SoftmaxModel, one_hot_encode
np.random.seed(0)


def calculate_accuracy(X: np.ndarray, targets: np.ndarray, model: SoftmaxModel) -> float:
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
        model: model of class SoftmaxModel
    Returns:
        Accuracy (float)
    """
    # As explained in task 2, rint round .5 to 0 which isnt what the assignment asks, but it is way faster than round
    # so the tradoff seems worth it
    Y = model.forward(X)
    Y_categorical = np.rint(Y)
    correct_guess_array = np.all(Y_categorical == targets, axis=1)
    correct_guess_count = np.count_nonzero(correct_guess_array)
    return correct_guess_count/targets.shape[0]


class SoftmaxTrainer(BaseTrainer):

    def train_step(self, X_batch: np.ndarray, Y_batch: np.ndarray):
        """
        Perform forward, backward and gradient descent step here.
        The function is called once for every batch (see trainer.py) to perform the train step.
        The function returns the mean loss value which is then automatically logged in our variable self.train_history.

        Args:
            X: one batch of images
            Y: one batch of labels
        Returns:
            loss value (float) on batch
        """
        self.model.zero_grad()
        Out_batch = self.model.forward(X_batch)
        loss = cross_entropy_loss(Y_batch, Out_batch) + self.model.l2_reg_lambda*np.linalg.norm(self.model.w)**2
        self.model.backward(X_batch, Out_batch, Y_batch)
        self.model.w = self.model.w - self.learning_rate*self.model.grad
        return loss

    def validation_step(self):
        """
        Perform a validation step to evaluate the model at the current step for the validation set.
        Also calculates the current accuracy of the model on the train set.
        Returns:
            loss (float): cross entropy loss over the whole dataset
            accuracy_ (float): accuracy over the whole dataset
        Returns:
            loss value (float) on batch
            accuracy_train (float): Accuracy on train dataset
            accuracy_val (float): Accuracy on the validation dataset
        """
        # NO NEED TO CHANGE THIS FUNCTION
        logits = self.model.forward(self.X_val)
        loss = cross_entropy_loss(self.Y_val, logits)

        accuracy_train = calculate_accuracy(
            self.X_train, self.Y_train, self.model)
        accuracy_val = calculate_accuracy(
            self.X_val, self.Y_val, self.model)
        return loss, accuracy_train, accuracy_val


def main():
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = 0.01
    batch_size = 128
    l2_reg_lambda = 0
    shuffle_dataset = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    # ANY PARTS OF THE CODE BELOW THIS CAN BE CHANGED.

    # Intialize model
    model = SoftmaxModel(l2_reg_lambda)
    # Train model
    trainer = SoftmaxTrainer(
        model, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model.forward(X_val)))
    print("Final Train accuracy:", calculate_accuracy(X_train, Y_train, model))
    print("Final Validation accuracy:", calculate_accuracy(X_val, Y_val, model))

    plt.ylim([.2, .7])
    utils.plot_loss(train_history["loss"],
                    "Training Loss", npoints_to_average=10)
    utils.plot_loss(val_history["loss"], "Validation Loss")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    plt.savefig("task3b_softmax_train_loss.png")
    plt.show()

    # Plot accuracy
    plt.ylim([.8, 1])
    utils.plot_loss(train_history["accuracy"], "Training Accuracy")
    utils.plot_loss(val_history["accuracy"], "Validation Accuracy")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task3b_softmax_train_accuracy.png")
    plt.show()

    # Train a model with L2 regularization (task 4b)
    print("With L2 regularization")
    model1 = SoftmaxModel(l2_reg_lambda=1.0)
    trainer = SoftmaxTrainer(
        model1, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_reg01, val_history_reg01 = trainer.train(num_epochs)
    # You can finish the rest of task 4 below this point.

    # Plotting of softmax weights (Task 4b)
    vannila_bias = model.w.T[:, -1]
    vanilla_weight = np.concatenate(model.w.T[:, :-1].reshape(10,28,28) - vannila_bias[:, None, None], axis=1)
    reg_bias = model1.w.T[:, -1]
    reg_weight = np.concatenate(model1.w.T[:,:-1].reshape(10,28,28) - reg_bias[:, None, None], axis=1)
    weight = np.vstack((vanilla_weight, reg_weight))
    plt.imsave("task4b_softmax_weight.png", weight, cmap="gray")

    # testing different lambda values ( Task 4c ) and plotting the resulting weights L_2 norm
    tests_reg_lambda = [1.0, 0.1, 0.01, 0.001]
    w_norms = []
    for reg_lambda in tests_reg_lambda:
        # Intialize model
        model = SoftmaxModel(reg_lambda)
        # Train model
        trainer = SoftmaxTrainer(
            model, learning_rate, batch_size, shuffle_dataset,
            X_train, Y_train, X_val, Y_val,
        )
        _, val_history = trainer.train(num_epochs)
        w_norms.append(np.linalg.norm(model.w) ** 2)
        utils.plot_loss(val_history["accuracy"], label=f"Lambda={reg_lambda}")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.title("Validation accuracy graph under different lambda values")
    plt.legend()
    plt.savefig("task4c_l2_reg_accuracy.png")
    plt.show()

    plt.plot(tests_reg_lambda, w_norms, marker='o', linestyle='--')
    plt.xlabel('lambda value ')
    plt.ylabel('L_2 norm of the weight matrix')
    plt.title('norm of the weigh matrix in terms of the regularization parameter')
    plt.legend()
    plt.savefig("task4d_l2_reg_norms.png")
    plt.show()


if __name__ == "__main__":
    main()
