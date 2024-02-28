import pathlib
import matplotlib.pyplot as plt
import utils
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer


class ExampleModel(nn.Module):
    def __init__(self, image_channels, num_classes):
        """
        Is called when model is initialized.
        Args:
            image_channels. Number of color channels in image (3)
            num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        self.num_classes = num_classes
        # Assumes same kernel, stride and padding for all conv layers
        self.conv_kernel_size = 5
        self.conv_stride = 1
        self.conv_padding = 2
        # Assumes same kernel and stride for all pool layers
        self.pool_stride = 2
        self.pool_kernel_size = 2
        # Number of output filters for each conv layer
        self.num_filters = [image_channels, 32, 64, 128]
        # Number of output of the feature extractor is the number of filter times size of the last pool layer output.
        # Hardcoded because no input image parameter. Computed in the report
        self.feature_extractor_output_size = self.num_filters[-1] * 4 * 4
        # Assumes a single hidden fcLayer
        self.hidden_units = 64

        # Creates feature extractor as a single layer
        # A conv layer is Conv2D -> ReLu -> MaxPool2D
        self.conv_layers = [
            nn.Sequential(
                nn.Conv2d(input_depth, output_depth, self.conv_kernel_size, self.conv_stride, self.conv_padding),
                nn.ReLU(),
                nn.MaxPool2d(self.pool_kernel_size, self.pool_stride))
            for input_depth, output_depth in zip(self.num_filters, self.num_filters[1:])
        ]
        self.feature_extractor = nn.Sequential(*self.conv_layers, nn.Flatten())

        # Creates the classifier as a single layer
        self.fc_layers = [
            nn.Linear(self.feature_extractor_output_size, self.hidden_units),
            nn.ReLU(),
            nn.Linear(self.hidden_units, num_classes)
        ]  # No softmax here because already included in cross entropy loss
        self.classifier = nn.Sequential(*self.fc_layers)

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        batch_size = x.shape[0]
        features = self.feature_extractor(x)
        out = self.classifier(features)
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (
            batch_size,
            self.num_classes,
        ), f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out


def create_plots(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(
        trainer.train_history["loss"], label="Training loss", npoints_to_average=10
    )
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()


def main():
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result!
    utils.set_seed(0)
    print(f"Using device: {utils.get_device()}")
    epochs = 10
    batch_size = 64
    learning_rate = 5e-2
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size)
    model = ExampleModel(image_channels=3, num_classes=10)
    trainer = Trainer(
        batch_size, learning_rate, early_stop_count, epochs, model, dataloaders
    )
    trainer.train()
    create_plots(trainer, "task2")


if __name__ == "__main__":
    main()
