""" Data loaders. One and two channel loaders are provided here,
to extend to higher dimensions the groundTruth and maskedData channels
should be concatenated with more channels. The network structure should
then be adjusted to incorporate more in_channels and out_channels, and
the internal architecture possibly changed.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import os

class LA_Dataset(Dataset):
    """ Dataset class for patched LA-ICP-MS data, with mask channel.
    Reads data generated using Generate_Training_Data script and 
    outputs in the form [obscuredData, groundTruth, chemicalMask] """
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.fileList = []

        fileNames = [f for f in os.listdir(folder)]
        for fileName in fileNames:
            filePath = os.path.join(folder, fileName)
            if not filePath.endswith("desktop.ini"): 
                self.fileList.append(filePath)

    def __len__(self):
        return len(self.fileList)

# Subclass to return two-channeled data
class TwoChannelDataset(LA_Dataset):

    def __getitem__(self, idx):
        # Assuming each file is a numpy array containing 
        # [chemical, obscure mask, GB mask], resulting from Generate_Training_Data.py
        filePath = self.fileList[idx]
        label = idx
        data = np.load(filePath, allow_pickle=True)
        chemicalChannel = torch.Tensor(data[0]).unsqueeze(0)
        chemicalMask = torch.Tensor(data[1]).unsqueeze(0)
        boundaryChannel = torch.Tensor(data[2]).unsqueeze(0)
        #boundaryMask = torch.Tensor(np.ones_like(boundaryChannel)).unsqueeze(0)
        maskedChemical = torch.Tensor(data[0]*data[1]).unsqueeze(0)
        groundTruth = torch.cat((chemicalChannel, boundaryChannel), dim=0)
        maskedData = torch.cat((maskedChemical, boundaryChannel), dim=0)
        label = torch.tensor(int(label))
        return maskedData, groundTruth, chemicalMask, label
  
# Subclass to return one-channeled data
class OneChannelDataset(LA_Dataset):
    def __getitem__(self, idx):
        filePath = self.fileList[idx]
        label = idx
        data = np.load(filePath, allow_pickle=True)
        groundTruth = torch.Tensor(data[0]).unsqueeze(0)
        mask = torch.Tensor(data[1]).unsqueeze(0)
        maskedChemical = torch.Tensor(data[0]*data[1]).unsqueeze(0)
        label = torch.tensor(int(label))

        return maskedChemical, groundTruth, mask, label

""" CNNs """

class CustomNet_2Channel(nn.Module):
    def __init__(self):
        super(CustomNet_2Channel, self).__init__()
        # Define layers
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.deconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Apply convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Apply deconvolutional layers
        x = F.relu(self.deconv1(x))
        x = self.deconv2(x)
        return x


class CustomNet_1Channel(nn.Module):
    def __init__(self):
        super(CustomNet_1Channel, self).__init__()
        # Define layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.deconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Apply convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Apply deconvolutional layers
        x = F.relu(self.deconv1(x))
        x = self.deconv2(x)
        return x

""" Collection of models available"""
models = {
"CustomNet_2Channel": CustomNet_2Channel,
"CustomNet_1Channel": CustomNet_1Channel
}