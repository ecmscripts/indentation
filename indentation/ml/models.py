import numpy as np 
import pandas as pd
from scipy.optimize import fmin
from scipy.signal import convolve
from scipy.signal.windows import gaussian
from scipy import interpolate
from tqdm import tqdm
import os, sys, glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ConvClassifier_1(nn.Module):
    def __init__(self, N):
        super().__init__()
        
        # N is both the input size and the number of classes
        self.N = N
        
        # Convolutional layers with max pooling
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=13, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=1)
        
        # Calculate the size after convolutions and pooling
        with torch.no_grad():
            x = torch.zeros(1, 1, N, N)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = self.pool(F.relu(self.conv4(x)))
            self.conv_output_size = x.view(1, -1).size(1)
        
        # Dense layers
        self.fc1 = nn.Linear(self.conv_output_size, 256)
        self.fc2 = nn.Linear(256, N)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Convolutional layers with ReLU activation and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        
        # Flatten the output for the dense layer
        x = x.view(-1, self.conv_output_size)
        
        # Dense layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        # Softmax activation
        smax = F.log_softmax(x, dim=1)
        #print(np.argmax(smax.detach().numpy(), axis=1))
        return smax

class ConvClassifier_2(nn.Module):
    def __init__(self, N):
        super().__init__()
        
        # N is both the input size and the number of classes
        self.N = N
        
        # Multi-scale convolutional layers
        self.conv1x1 = nn.Conv2d(1, 16, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.conv13x13 = nn.Conv2d(1, 16, kernel_size=13, stride=1, padding=6)  # Same padding
        
        # Second convolutional layer and max pooling
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calculate the size after convolutions and pooling
        with torch.no_grad():
            x = torch.zeros(1, 1, N, N)
            x1 = self.conv1x1(x)
            x2 = self.conv3x3(x)
            x3 = self.conv5x5(x)
            x4 = self.conv13x13(x)
            x = torch.cat([x1, x2, x3, x4], dim=1)
            x = self.pool(F.relu(x))
            x = self.pool(F.relu(self.conv2(x)))
            self.conv_output_size = x.view(1, -1).size(1)
        
        # Dense layers
        self.fc1 = nn.Linear(self.conv_output_size, 256)
        self.fc2 = nn.Linear(256, N)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Multi-scale feature extraction
        x1 = F.relu(self.conv1x1(x))
        x2 = F.relu(self.conv3x3(x))
        x3 = F.relu(self.conv5x5(x))
        x4 = F.relu(self.conv13x13(x))
        
        # Concatenate the feature maps along the channel dimension
        x = torch.cat([x1, x2, x3, x4], dim=1)
        
        # Apply pooling and subsequent convolutional layers
        x = self.pool(x)
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten the output for the dense layer
        x = x.view(-1, self.conv_output_size)
        
        # Dense layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        # Softmax activation
        smax = F.log_softmax(x, dim=1)
        return smax