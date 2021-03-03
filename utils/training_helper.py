import matplotlib.pyplot as plt
import numpy as np
import os
import torch

import neptune


class TrainingHelper:

    tracked_values = ['epoch_loss_train',
                      'epoch_acc_train',
                      'epoch_loss_test',
                      'epoch_acc_test',
                      'batch_loss_train']
    CHECKPOINTS_PATH = 'checkpoints/'

    def __init__(self, trainset_len, trainloader, testset_len, testloader, device, is_neptune, name):
        self.trainset_len = trainset_len
        self.trainloader = trainloader
        self.testset_len = testset_len
        self.testloader = testloader
        self.device = device
        self.is_neptune = is_neptune
        self.name = name

    def train_and_evaluate_model(self,
                                 model,
                                 criterion,
                                 optimizer,
                                 num_epochs=10,
                                 save_every_nth_batch_loss=50):
        """Train and evaluate the classification model."""
        run_hist = {key: [] for key in self.tracked_values}
        try:
            for epoch in range(num_epochs):
                print('Epoch {}/{}'.format(epoch+1, num_epochs))
                print('-' * 10)

                # training phase
                model.train()

                running_loss_train = 0.0
                running_corrects_train = 0

                i = 0
                for inputs, labels in self.trainloader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    _, preds = torch.max(outputs, 1)
                    running_loss_train += loss.detach() * inputs.size(0)
                    running_corrects_train += torch.sum(preds == labels.data)

                    if i % save_every_nth_batch_loss == 0:
                        run_hist['batch_loss_train'].append(loss.item())
                    i += 1

                epoch_loss_train = running_loss_train / self.trainset_len
                epoch_acc_train = running_corrects_train.float() / self.trainset_len

                print('train loss: {:.4f}, train acc: {:.4f}'.\
                format(epoch_loss_train.item(),
                       epoch_acc_train.item()))

                run_hist['epoch_loss_train'].append(epoch_loss_train.item())
                run_hist['epoch_acc_train'].append(epoch_acc_train.item())

                if self.is_neptune:
                    neptune.log_metric('loss_train_' + self.name, epoch_loss_train.item())
                    neptune.log_metric('acc_train_' + self.name, epoch_acc_train.item())

                # evaluating phase
                model.eval()

                running_loss_test = 0.0
                running_corrects_test = 0

                for inputs, labels in self.testloader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)
                    running_loss_test += loss.detach() * inputs.size(0)
                    running_corrects_test += torch.sum(preds == labels.data)

                epoch_loss_test = running_loss_test / self.testset_len
                epoch_acc_test = running_corrects_test.float() / self.testset_len

                print('test loss: {:.4f}, test acc: {:.4f}\n'.\
                format(epoch_loss_test.item(),
                       epoch_acc_test.item()))

                run_hist['epoch_loss_test'].append(epoch_loss_test.item())
                run_hist['epoch_acc_test'].append(epoch_acc_test.item())

                if self.is_neptune:
                    neptune.log_metric('loss_test_' + self.name, epoch_loss_test.item())
                    neptune.log_metric('acc_test_' + self.name, epoch_acc_test.item())

        except KeyboardInterrupt:
            pass
        return run_hist

    def plot_training(self, run_hist):
        """Plot the training history of the classification model."""
        fig, ax = plt.subplots(1, 2, figsize=(20, 6), sharex=True)
        x = np.arange(len(run_hist["epoch_loss_train"])) + 1
        ax[0].plot(x, run_hist["epoch_loss_train"], 'b', marker='.', label="epoch train loss")
        ax[0].plot(x, run_hist["epoch_loss_test"], 'r', marker='.', label="epoch test loss")
        ax[0].legend()
        ax[1].plot(x, run_hist["epoch_acc_train"], 'b', marker='.', label="epoch train accuracy")
        ax[1].plot(x, run_hist["epoch_acc_test"], 'r', marker='.', label="epoch test accuracy")
        ax[1].legend()

        fig, ax = plt.subplots(1, 1, figsize=(20, 6), sharex=True)
        x = np.arange(len(run_hist["batch_loss_train"])) + 1
        ax.plot(x, run_hist["batch_loss_train"], 'b', marker='.', label="batch train loss")
        ax.legend()

    def save(self, model, name):
        torch.save(model, os.path.join(self.CHECKPOINTS_PATH, name))

    def load(self, name):
        model = torch.load(os.path.join(self.CHECKPOINTS_PATH, name))
        model.eval()
        return model
