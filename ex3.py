import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
%matplotlib inline

data = FashionMNIST(root='data/', download=True, transform=ToTensor())

validation_size = 10000
train_size = len(data) - validation_size

train_data, val_data = random_split(data, [train_size, validation_size])

batch_size=128

train_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_data, batch_size*2, num_workers=4, pin_memory=True)


class FFNN(nn.Module):
    def __init__(self, input_size, num_hidden_layers, hidden_size, out_size, accuracy_function):
        super().__init__()
        self.accuracy_function = accuracy_function
        self.input_layer = nn.Linear(input_size, hidden_size)

        self.hidden_layers = nn.ModuleList()
        for i in range(0, num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))

        self.output_layer = nn.Linear(hidden_size, out_size)

    def forward(self, input_image):  # connects all layers we defined in the constructor
        input_image = input_image.view(input_image.size(0), -1)

        output = self.input_layer(input_image)
        output = F.relu(output)  # ReLu activation function

        for layer in self.hidden_layers:
            output = layer(output)
            output = F.relu(output)

        output = self.output_layer(output)
        return output

    def training_step(self,
                      batch):  # this function takes the batch of images by the DataLoader and pushes them to get the prediction
        images, labels = batch
        output = self(images)
        loss = F.cross_entropy(output, labels)  # cross_entropy omputes the cross entropy loss between input and target
        return loss

    def validation_step(self, batch):
        images, labels = batch
        output = self(images)
        loss = F.cross_entropy(output, labels)
        acc = self.accuracy_function(output, labels)

        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # combine losses and return mean value

        batch_accs = [x['val_acc'] for x in outputs]  # Combine accuracies and return mean value
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch: {} - Validation Loss: {:.4f}, Validation Accuracy: {:.4f}".format(epoch, result['val_loss'],
                                                                                        result['val_acc']))


class ModelTrainer():
    def fit(self, epochs, learning_rate, model, train_loader, val_loader, opt_func=torch.optim.SGD):
        history = []
        optimizer = opt_func(model.parameters(), learning_rate)

        for epoch in range(epochs):
            for batch in train_loader:
                loss = model.training_step(batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            result = self._evaluate(model, val_loader)
            model.epoch_end(epoch, result)
            history.append(result)

        return history

    def _evaluate(self, model, val_loader):
        outputs = [model.validation_step(batch) for batch in val_loader]
        return model.validation_epoch_end(outputs)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def plot_history(history):
    losses = [x['val_loss'] for x in history]
    plt.plot(losses, '-x')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Loss and Accuracy');

input_size = 784
num_classes = 10

model = FFNN(input_size, 3, 32, out_size=num_classes, accuracy_function=accuracy)
print(model)

model_trainer = ModelTrainer()

training_history = []
training_history += model_trainer.fit(10, 0.2, model, train_loader, val_loader)

plot_history(training_history)