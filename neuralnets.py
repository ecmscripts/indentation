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

def train_network(net, data, num_epochs=10, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    
    # Use CrossEntropyLoss instead of NLLLoss
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    total_loss = 0.0
    accuracy   = 0

    for epoch in range(num_epochs):
        clear_output(wait=True)
        net.train()
        running_loss = 0.0

        print("Total Progress:\t\t", np.round((epoch+1)/num_epochs*100, 2), "%")
        print(f'Accuracy: \t\t {accuracy:.2f}%\n')
        
        for batch_idx, (X, y) in enumerate(data):
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            output = net(X)
            
            loss = F.nll_loss(output, y)

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Print batch loss every 10 batches
            if batch_idx % 10 == 9:
                print(f'Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {running_loss/10:.4f}')
                running_loss = 0.0
        
        # Evaluate on the entire dataset after each epoch
        net.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        with torch.no_grad():
            for X, y in data:
                X, y = X.to(device), y.to(device)
                outputs = net(X)
                total_loss += F.nll_loss(output, y).item()
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
        
        avg_loss = total_loss / len(data)
        accuracy = 100. * correct / total
        print(f'Epoch {epoch+1}: Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')


def create_data_sets(displ_max = 100, step_size = 0.05, alpha = 0.1, alpha_var = 0.01, 
                     noise_scale1 = 2, noise_scale2 = 0.1, noise_var=0.5, baseline_tilt_max=0.1,
                     batch_size = 32, N_batches  = 50, N=100):

    N_gen_data = N_batches*batch_size

    data_all_tests     = []
    data_all_tests_img = []
    for i in range(N_gen_data):

        # base indentation to be modded
        displ_base, force_base = indentation_force(np.linspace(0, displ_max, int(displ_max/step_size)), displ_max, alpha + alpha_var*np.random.rand())

        # random shift data to obtain new reference point
        xs = displ_max/3 + 0.7*displ_max*np.random.rand()
        
        # add noise to indentation data
        ys = 3*np.random.rand()
        displ, force = shift(displ_base, force_base, displ_max, xs, ys)
        force        = add_noise(force, noise_scale1, noise_scale2, noise_var)
        force        = add_baseline_tilt(force, baseline_tilt_max)

        # clip data
        displ, force = clip_data(displ, force)
        xs_norm      = xs/np.max(displ)
        displ, force = normalize(displ, force, N)
        
        # calculate shift index
        ix_shift = np.argmin(np.abs(np.linspace(0,1,N)-xs_norm))

        # compute image
        force_img = create_convolution_image(force, N=N)
                            
        # append data signal-ground_truth pair 
        data_all_tests.append([torch.tensor(force, dtype=torch.float).view(-1 ,N), int(ix_shift)])
        data_all_tests_img.append([torch.tensor(force_img, dtype=torch.float).view(-1 ,N, N), int(ix_shift)])

    # create batches
    data     = []
    data_img = []

    for i in range(N_batches):
        datax = []
        datay = []
        dataximg = []
        datayimg = []
        for j in range(batch_size):
            ix = i*batch_size+j
            X, y = data_all_tests[ix]
            Ximg, yimg = data_all_tests_img[ix]
            datax.append(X)
            datay.append(y)
            dataximg.append(Ximg)
            datayimg.append(yimg)
        data.append([torch.stack(datax), torch.tensor(datay)])
        data_img.append([torch.stack(dataximg), torch.tensor(datayimg)])
    return data, data_img
