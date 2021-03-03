import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import neptune

from models.fcnet import FCNet
from models.convnet import ConvNet
from models.logreg import LogReg

from utils.training_helper import TrainingHelper
from utils.data_helper import DataHelper

NEPTUNE_LOG = False

data_helper = DataHelper()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_set_raw = data_helper.train_set_raw
train_set_flip = data_helper.train_set_flip
train_set_rotate = data_helper.train_set_rotate
test_set = data_helper.test_set_raw

training_raw = TrainingHelper(len(train_set_raw),
                              data_helper.get_loader_train(train_set_raw),
                              len(test_set),
                              data_helper.get_loader_test(test_set),
                              device,
                              NEPTUNE_LOG, 'raw')
training_flip = TrainingHelper(len(train_set_flip),
                              data_helper.get_loader_train(train_set_flip),
                              len(test_set),
                              data_helper.get_loader_test(test_set),
                              device,
                              NEPTUNE_LOG, 'flip')
training_rotate = TrainingHelper(len(train_set_rotate),
                              data_helper.get_loader_train(train_set_rotate),
                              len(test_set),
                              data_helper.get_loader_test(test_set),
                              device,
                              NEPTUNE_LOG, 'rotation')

models = {
    'raw': ConvNet().to(device),
    'flip': ConvNet().to(device),
    'rotate': ConvNet().to(device)
}

criterion = nn.CrossEntropyLoss()
optimizer_raw = optim.Adam(models['raw'].parameters())

print("----------RAW----------")
model = models['raw']
run_hist = training_raw.train_and_evaluate_model(model,
                                             nn.CrossEntropyLoss(),
                                             optim.Adam(model.parameters()),
                                             num_epochs=50)
training_raw.save(model, 'raw1')

print("----------FLIP----------")
model = models['flip']
run_hist = training_flip.train_and_evaluate_model(model,
                                             nn.CrossEntropyLoss(),
                                             optim.Adam(model.parameters()),
                                             num_epochs=50)
training_flip.save(model, 'flip1')

print("----------ROTATE----------")
model = models['rotate']
run_hist = training_rotate.train_and_evaluate_model(model,
                                             nn.CrossEntropyLoss(),
                                             optim.Adam(model.parameters()),
                                             num_epochs=50)
training_rotate.save(model, 'rotate1')

if NEPTUNE_LOG:
    neptune.init(project_qualified_name='uw-niedziol/fashion-mnist')
    neptune.create_experiment(name='Convolutional Network',
                              upload_source_files=['main.py', 'models/*.py'])

# models = {
#     'raw': training_rotate.load('raw1'),
#     'flip': training_rotate.load('flip1'),
#     'rotate': training_rotate.load('rotate1')
# }

predictions = []

running_loss_test = 0.0
running_corrects_test = 0

for inputs, labels in data_helper.get_loader_test(test_set):
    inputs = inputs.to(device)
    labels = labels.to(device)

    outputs = []
    for model in models.values():
        output = model(inputs)
        output = F.softmax(output, dim=1)
        outputs.append(output)

    output = (outputs[0] + outputs[1] + outputs[2]) / 3
    loss = criterion(output, labels)

    _, preds = torch.max(output, 1)
    running_loss_test += loss.detach() * inputs.size(0)
    running_corrects_test += torch.sum(preds == labels.data)
    predictions += list(preds.cpu().detach().numpy())

acc_test = running_corrects_test.float() / len(test_set)
print(acc_test)

predictions = np.array(predictions)

submission = pd.DataFrame()
submission['Id'] = np.arange(len(test_set))
submission['Class'] = predictions
submission.to_csv('submission4.csv', index=False)
