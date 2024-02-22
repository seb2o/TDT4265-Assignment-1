import numpy as np
import utils
import typing

np.random.seed(1)


def pre_process_images(X: np.ndarray):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785] normalized as described in task2a
    """
    mean = 33.318421449829934
    std = 78.56748998339798
    assert X.shape[1] == 784, f"X.shape[1]: {X.shape[1]}, should be 784"
    X = np.column_stack((X, np.ones(X.shape[0])))
    result = (X - mean) / std
    """
    The paper states input should be normalized per channel,
    so we implemented this for test reasons in the commented section.
    It performed worse.
    """
    # mX = np.mean(X, axis=0)
    # mX = X - mX
    # stdX = np.std(X, axis=0)
    # stdX = np.nan_to_num(mX / stdX)
    # result = stdX
    return result


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """
    assert (
        targets.shape == outputs.shape
    ), f"Targets shape: {targets.shape}, outputs: {outputs.shape}"
    return -np.mean(np.sum(targets * np.log(outputs), axis=1))


def broacasted_sigmoid(Z: np.ndarray) -> np.ndarray:
    return np.divide(1, 1 + np.exp(-Z))


def broacasted_sigmoid_prime(Z: np.ndarray) -> np.ndarray:
    fz = broacasted_sigmoid(Z)
    return fz * (1 - fz)


def softmax(Z: np.ndarray) -> np.ndarray:
    exp_z = np.exp(Z)
    sum_exp = np.sum(exp_z, axis=1, keepdims=True)
    return np.divide(exp_z, sum_exp)


class SoftmaxModel:

    def __init__(
            self,
            # Number of neurons per layer
            neurons_per_layer: typing.List[int],
            use_improved_sigmoid: bool,  # Task 3b hyperparameter
            use_improved_weight_init: bool,  # Task 3a hyperparameter
            use_relu: bool,  # Task 3c hyperparameter
    ):
        np.random.seed(
            1
        )  # Always reset random seed before weight init to get comparable results.
        # Define number of input nodes
        # We reduce the input layer back to its original dimension but add
        # an extra bias when initializing the weight matrices
        self.I = 784
        self.use_improved_sigmoid = use_improved_sigmoid
        self.use_relu = use_relu
        self.use_improved_weight_init = use_improved_weight_init

        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer

        # store number of layers because often accessed
        self.n_layers = len(neurons_per_layer)
        # store the intermediate results z for each layer.
        self.layers_z = [np.ndarray(0)] * self.n_layers

        # Initialize the weights
        self.ws = []
        prev = self.I
        for size in self.neurons_per_layer:
            # add bias
            w_shape = (prev + 1, size)
            print("Initializing weight to shape:", w_shape)
            w = np.random.uniform(-1, 1, w_shape)
            self.ws.append(w)
            prev = size
        self.grads = [None for i in range(len(self.ws))]

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """

        # multiply each layer output by the weights of the next layer. The first layer output is the input of the
        # network and the last layer output is the output of the network
        self.layers_z[0] = X @ self.ws[0]
        for layer_index in range(1, self.n_layers):
            prev_layer = broacasted_sigmoid(self.layers_z[layer_index - 1])
            # we add the bias during forward
            prev_layer = np.column_stack((prev_layer, np.ones(prev_layer.shape[0])))
            self.layers_z[layer_index] = prev_layer @ self.ws[layer_index]
        return softmax(self.layers_z[-1])

    def backward(self, X: np.ndarray, outputs: np.ndarray, targets: np.ndarray) -> None:
        """
        Computes the gradient and saves it to the variable self.grad

        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        assert (
            targets.shape == outputs.shape
        ), f"Output shape: {outputs.shape}, targets: {targets.shape}"

        batch_size = X.shape[0]
        # A list of gradients.
        # For example, self.grads[0] will be the gradient for the first hidden layer
        self.grads = [np.zeros_like(layer) for layer in self.ws]
        delta = [np.zeros_like(layer) for layer in self.ws]

        # Compute the error vectors, starting with the last layer.
        delta[-1] = outputs - targets
        for j in range(2, self.n_layers + 1):
            prev_layer = broacasted_sigmoid_prime(self.layers_z[-j])
            # we add the bias for updating it's weight
            prev_layer = np.column_stack((prev_layer, np.ones(prev_layer.shape[0])))
            delta[-j] = prev_layer * (delta[-j + 1] @ self.ws[-j + 1].T)

        # we do however remove the bias when going backwards
        self.grads[0] = (X.T @ delta[0][:, :-1]) / batch_size
        for j in range(1, self.n_layers):
            prev_layer = broacasted_sigmoid(self.layers_z[j - 1])
            prev_layer = np.column_stack((prev_layer, np.ones(prev_layer.shape[0])))
            if j < self.n_layers - 1:
                self.grads[j] = (prev_layer.T @ delta[j][:, :-1]) / batch_size
            else:
                # the output layer has no bias
                self.grads[j] = (prev_layer.T @ delta[j]) / batch_size

        for grad, w in zip(self.grads, self.ws):
            assert (
                grad.shape == w.shape
            ), f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."

    def zero_grad(self) -> None:
        self.grads = [None for i in range(len(self.ws))]


def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """
    i = np.eye(num_classes)
    r = i[Y.flatten()]
    return r


def gradient_approximation_test(model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
    Numerical approximation for gradients. Should not be edited.
    Details about this test is given in the appendix in the assignment.
    """

    assert isinstance(X, np.ndarray) and isinstance(
        Y, np.ndarray
    ), f"X and Y should be of type np.ndarray!, got {type(X), type(Y)}"

    epsilon = 1e-3
    for layer_idx, w in enumerate(model.ws):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                difference = gradient_approximation - \
                    model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon ** 1, (
                    f"Calculated gradient is incorrect. "
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n"
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n"
                    f"If this test fails there could be errors in your cross entropy loss function, "
                    f"forward function or backward function"
                )


def main():
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert (
        Y[0, 3] == 1 and Y.sum() == 1
    ), f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    Y_train = one_hot_encode(Y_train, 10)
    assert (
        X_train.shape[1] == 785
    ), f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64, 10]
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_relu = True
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init, use_relu
    )

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)


if __name__ == "__main__":
    main()
