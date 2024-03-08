import pathlib
import matplotlib.pyplot as plt
import torch

import utils
from torch import nn
from improvedTrainer import ImprovedTrainer
from dataloaders import load_cifar10_task4
from trainer import Trainer, compute_loss_and_accuracy
import torchvision

from torch import nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, 10)  # No need to apply softmax,
                                            # as this is done in nn.CrossEntropyLoss

        for param in self.model.parameters(): # Freeze all parameters
            param.requires_grad = False
        for param in self.model.fc.parameters(): # Unfreeze the last fully-connected
            param.requires_grad = True # layer
        for param in self.model.layer4.parameters(): # Unfreeze the last 5 convolutional
            param.requires_grad = True # layers

    def forward(self, x):
        x = self.model(x)
        return x

def create_plots(trainer: ImprovedTrainer, name: str):
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
    batch_size = 32
    early_stop_count = 4
    dataloaders = load_cifar10_task4(batch_size)


    model = Model()
    #optimizer_adam = torch.optim.Adam(model.parameters(),5e-4)
    optimizer_sgd = torch.optim.SGD(model.parameters(),2e-2)
    trainer = ImprovedTrainer(
        batch_size, early_stop_count, epochs, model, dataloaders, optimizer_sgd
    )
    trainer.train()


    create_plots(trainer, "task4")
    loss_train, acc_train = compute_loss_and_accuracy(dataloaders[0],model,trainer.loss_criterion)
    loss_val, acc_val = compute_loss_and_accuracy(dataloaders[1],model,trainer.loss_criterion)
    loss_test, acc_test = compute_loss_and_accuracy(dataloaders[2],model,trainer.loss_criterion)
    print("Loss Training/Validation/Test: %.02f/%.02f/%.02f" %(loss_train,loss_val,loss_test))
    print("Accuracy Training/Validation/Test: %.02f/%.02f/%.02f" %(acc_train,acc_val,acc_test))

if __name__ == "__main__":
    main()