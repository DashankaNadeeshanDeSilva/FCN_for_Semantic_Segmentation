from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import sys
import os

from matplotlib import pyplot as plt
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader

from Network_models import VGGNet, FCN32, FCN16, FCN8, FCN
from Data_loader import Dataset_Prep

n_class = 20

batch_size = 6
epochs = 500
lr = 1e-4
momentum = 0
w_decay = 1e-5
step_size = 50
gamma = 0.5

configs = "FCNs-BCEWithLogits_batch: {} \nepoch: {}\nRMSprop_scheduler-step: {}\ngamma: {}learning_rate: {}\nmomentum: {}\nweight_decay: {}".format(
    batch_size, epochs, step_size, gamma, lr, momentum, w_decay)
print("Configs:", configs)

root_dir = "CityScapes/"
train_file = os.path.join(root_dir, "train.csv")
val_file = os.path.join(root_dir, "val.csv")

# create dir for model
model_dir = "models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_path = os.path.join(model_dir, configs)

# Setup GPU
use_gpu = torch.cuda.is_available()
num_gpu = list(range(torch.cuda.device_count()))

# Data preparation
train_data = Dataset_Prep(csv_file=train_file, phase='train')
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)

val_data = Dataset_Prep(csv_file=val_file, phase='val', flip_rate=0)
val_loader = DataLoader(val_data, batch_size=1, num_workers=8)

# Model setup
vgg_model = VGGNet(requires_grad=True, remove_fc=True)
fcn_model = FCN(pretrained_net=vgg_model, n_class=n_class)

# GPU setup for model
if use_gpu:
    ts = time.time()
    vgg_model = vgg_model.cuda()
    fcn_model = fcn_model.cuda()
    fcn_model = nn.DataParallel(fcn_model, device_ids=num_gpu)
    print("Finish cuda loading, time elapsed {}".format(time.time() - ts))

# Optimizer and loss function setup
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.RMSprop(fcn_model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
# decay learning rate by a factor of 0.5 every 30 epochs
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# create directory for score
score_dir = os.path.join("scores", configs)

if not os.path.exists(score_dir):
    os.makedirs(score_dir)

IU_scores = np.zeros((epochs, n_class))
pixel_scores = np.zeros(epochs)


# Train function
def train():
    for epoch in range(epochs):
        scheduler.step()

        ts = time.time()
        for iter, batch in enumerate(train_loader):
            optimizer.zero_grad()

            if use_gpu:
                inputs = Variable(batch['X'].cuda())
                labels = Variable(batch['Y'].cuda())
            else:
                inputs, labels = Variable(batch['X']), Variable(batch['Y'])

            outputs = fcn_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.data[0]))

        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        # save models weights
        torch.save(fcn_model, model_path)

        # Validation after every epoch
        validation(epoch)


def validation(epoch):
    fcn_model.eval()
    total_ious = []
    pixel_accs = []
    for iter, batch in enumerate(val_loader):
        if use_gpu:
            inputs = Variable(batch['X'].cuda())
        else:
            inputs = Variable(batch['X'])

        output = fcn_model(inputs)
        output = output.data.cpu().numpy()

        N, _, h, w = output.shape
        pred = output.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w)

        target = batch['l'].cpu().numpy().reshape(N, h, w)
        for p, t in zip(pred, target):
            total_ious.append(intersection_over_union(p, t))
            pixel_accs.append(pixel_acc(p, t))

    # Calculate average IoU
    total_ious = np.array(total_ious).T  # n_class * val_len
    ious = np.nanmean(total_ious, axis=1)
    pixel_accs = np.array(pixel_accs).mean()
    print("epoch{}, pix_acc: {}, meanIoU: {}, IoUs: {}".format(epoch, pixel_accs, np.nanmean(ious), ious))
    IU_scores[epoch] = ious
    # save scores
    np.save(os.path.join(score_dir, "meanIU"), IU_scores)
    pixel_scores[epoch] = pixel_accs
    np.save(os.path.join(score_dir, "meanPixel"), pixel_scores)


# Calculates class intersections over unions
def intersection_over_union(pred, target):
    ious = []
    for cls in range(n_class):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            # if no ground truth is found, do not include in the evaluation
            ious.append(float('nan'))
        else:
            ious.append(float(intersection) / max(union, 1))
    return ious


def pixel_acc(pred, target):
    correct = (pred == target).sum()
    total   = (target == target).sum()
    return correct / total


if __name__ == "__main__":
    # show the accuracy before training
    validation(0)
    train()