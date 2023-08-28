#!pip install torch-summary

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets
import torchvision.transforms as T
import cv2

import os

class CNN_Model_Torch(nn.Module):
  def __init__(self):
    super(CNN_Model_Torch, self).__init__()

    #self.expected_input_shape = (1, 3, 224, 224)
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding='same')
    self.relu1 = nn.ReLU()
    self.bnn1 = nn.BatchNorm2d(32)
    self.maxpool1 = nn.MaxPool2d(kernel_size=2)

    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same')
    self.relu2 = nn.ReLU()
    self.bnn2 = nn.BatchNorm2d(64)
    self.maxpool2 = nn.MaxPool2d(kernel_size=2)

    self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same')
    self.relu3 = nn.ReLU()
    self.bnn3 = nn.BatchNorm2d(128)
    self.maxpool3 = nn.MaxPool2d(kernel_size=2)

    self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same')
    self.relu4 = nn.ReLU()
    self.bnn4 = nn.BatchNorm2d(128)
    self.maxpool4 = nn.MaxPool2d(kernel_size=2)
    
    self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding='same')
    self.relu5 = nn.ReLU()
    self.bnn5 = nn.BatchNorm2d(256)

    self.flat = nn.Flatten()
    self.fc1 = nn.Linear(50176,512)
    self.bnn6 = nn.BatchNorm1d(512)
    self.fc2 = nn.Linear(512,75)
    #self.softmax = nn.Softmax(dim=1)
    #self.fc3 = nn.Linear(256,75)

  def forward(self, x):
    out = self.conv1(x)
    out = self.relu1(out)
    out = self.bnn1(out)
    out = self.maxpool1(out)

    out = self.conv2(out)
    out = self.relu2(out)
    out = self.bnn2(out)
    out = self.maxpool2(out)

    out = self.conv3(out)
    out = self.relu3(out)
    out = self.bnn3(out)
    out = self.maxpool3(out)

    out = self.conv4(out)
    out = self.relu4(out)
    out = self.bnn4(out)
    out = self.maxpool4(out)
    
    out = self.conv5(out)
    out = self.relu5(out)
    out = self.bnn5(out)

    out = self.flat(out)
    out = self.fc1(out)
    out = self.bnn6(out)
    out = self.fc2(out)
    #out = self.softmax(out)
    #out = self.fc3(out)

    return out

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CNN_Model_Torch()
model = model.to(device)
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.001)


### PRINT MODEL LAYERS

print(model)

from torchsummary import summary
#summary(model, (3, 224, 224))
summary(model, input_image_tensor_shape)