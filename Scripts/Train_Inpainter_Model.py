"""
This script trains an inpainting NN. Training behaviour is dictated by
which config file is loaded, with the following variables:

- trainingData (str): indicates the file that training/test data should be loaded from
- 

The training metric is ____

Model weights are saved like ____

Metrics

"""

import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import csv
from torch.optim.lr_scheduler import StepLR
import Utils
import NNs

""" Establish data inputs and outputs """

# Indicate config to use, initalise name for this model, and make model folder
configFile = "Configs/Ice_2C_NN_Simple.json"
time = datetime.now().strftime("%Y-%m-%d_%H")
modelName = f"{time}_{os.path.basename(os.path.splitext(configFile)[0])}"
currentDir = os.path.dirname(os.path.abspath(__file__))
modelPath = os.path.join(os.path.dirname(currentDir), f"Models/{modelName}_/")

# Load config
configDir = os.path.join(os.path.dirname(currentDir), configFile)
with open(configDir, 'r') as f:
    config = json.load(f)

# Make required dirs
if not os.path.isdir(modelPath):
    os.makedirs(modelPath)

saveMetricsPath = rf"{modelPath}TrainingMetrics/"
if not os.path.isdir(saveMetricsPath):
    os.makedirs(saveMetricsPath)

saveFolder = f"{modelPath}model_Weights/"
if not os.path.isdir(saveFolder):
    os.makedirs(saveFolder)

""" Initalise model and data """

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Load training data
dataPath = config["trainingData"]
trainset = NNs.TwoChannelDataset(f'{dataPath}/Train')#, transform=transform_train)
testset = NNs.TwoChannelDataset(f'{dataPath}/Test')#, transform=transform_test)

# Create DataLoader
batchSize = 32
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False)

# Initialize model, optimizer, and loss function
model = NNs.models[config["model"]]()
optimizer = optim.Adam(model.parameters(), lr=config["learningRate"])
scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
criterion = nn.MSELoss()
num_epochs = config["epochs"]

trainMSE = []
testMSE = []
trainSSIM = []
testSSIM = []
resultsTrain = []
resultsTest = []

# Model training and validation loop
for epoch in range(num_epochs):
    print(f'Start Epoch [{epoch+1}/{num_epochs}]')
    model.train()
    runningLoss = 0.0
    runningSim = 0.0

    for i, (batchMasked, batchTruth, batchMasks, batchLabels) in enumerate(trainloader):
        # Zero gradients and make forward pass

        optimizer.zero_grad()
        outputs = model(batchMasked)
        # Calculate losses on ONLY chem channel
        
        chemChannelOutput = outputs[:, 0, :, :]
        chemChannelTruth = batchTruth[:, 0, :, :]
        
        loss = criterion(chemChannelOutput, chemChannelTruth)
        similarity = Utils.calculate_ssim(chemChannelOutput, chemChannelTruth)
        
        # Backpropagation
        loss.backward()
        optimizer.step()

        runningLoss += loss.item()
        runningSim += similarity
        
    scheduler.step()

    # Print and record loss data for training
    averageLoss = runningLoss / len(trainloader)
    averageSim = runningSim / len(trainloader)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {averageLoss}, SSIM: {averageSim}')
    trainMSE.append(averageLoss)
    trainSSIM.append(averageSim)

    # Check performance on test set
    model.eval()
    testLoss = 0.0
    testSim = 0.0

    with torch.no_grad():
        for batchMasked, batchTruth, batchMasks, batchLabels in testloader:
            outputs = model(batchMasked)
            
            # Calculate losses on ONLY chem channel
            chemChannelOutput = outputs[:, 0, :, :]
            chemChannelTruth = batchTruth[:, 0, :, :]
            
            loss = criterion(chemChannelOutput, chemChannelTruth)
            similarity = Utils.calculate_ssim(chemChannelOutput, chemChannelTruth)

            testLoss += loss.item()
            testSim += similarity

    averageTestLoss = testLoss / len(testloader)
    averageTestSSIM = testSim / len(testloader)

    print(f'Validation Loss after Epoch {epoch+1}: {averageTestLoss}, SSIM: {averageTestSSIM}')
    testMSE.append(averageTestLoss)
    testSSIM.append(averageTestSSIM)

    resultsTest.append([epoch, averageTestLoss, averageTestSSIM])
    resultsTrain.append([epoch, averageLoss, averageSim])

    # Periodically print & save examples
    if (epoch+1) % 5 == 0:
        # Plot example data
        sampleMasked, sampleTruth, sampleMasks, sampleLabel = next(iter(testloader))
        with torch.no_grad():
            predicted = model(sampleMasked)
        Utils.plot_Test(predicted, sampleTruth, sampleMasked, sampleMasks, save = f'{modelPath}/Epoch_{epoch}')
    torch.save(model, f'{saveFolder}epoch{epoch}.pth')

print('Finished Training')
    
""" Save training data"""
with open(os.path.join(saveMetricsPath, 'testMetrics.txt'), 'w', newline='') as f:
    writer = csv.writer(f)
    # Write the header
    writer.writerow(['epoch', 'MSE', 'SSIM'])
    # Write the data rows
    writer.writerows(resultsTest)
    
with open(os.path.join(saveMetricsPath, 'trainMetrics.txt'), 'w', newline='') as f:
    writer = csv.writer(f)
    # Write the header
    writer.writerow(['epoch', 'MSE', 'SSIM'])
    # Write the data rows
    writer.writerows(resultsTrain)  

""" Plot and save training metrics"""
fig, ax = plt.subplots()
ax.plot(testMSE, label = "Validation data")
ax.plot(trainMSE, color = "red", label = "Training data")
ax.legend()
ax.set_ylabel('MSE Loss', fontsize=28)
ax.set_xlabel('Epoch', fontsize=28)
plt.savefig(f'{saveMetricsPath}_MSE.png', dpi=300)
plt.close()

fig, ax = plt.subplots()
ax.plot(testSSIM, label = "Validation data")
ax.plot(trainSSIM, color = "red", label = "Training data")
ax.legend()
ax.set_ylabel('SSIM', fontsize=28)
ax.set_xlabel('Epoch', fontsize=28)
plt.savefig(f'{saveMetricsPath}_SSIM.png', dpi=300)
plt.close()

    
    
    
    
    
    