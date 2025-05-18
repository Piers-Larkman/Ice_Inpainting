import numpy as np
import cv2 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import Utils

# Function to inpaint using Telea algorithm
def CV_inpaint_onechannel(array, mask):
    # Convert mask to the format required by OpenCV (uint8 with 0 or 255 values)
    mask_cv2 = (mask.astype(np.uint8) * 255)
    # Inpaint using Telea
    inpainted = cv2.inpaint(array.astype(np.float32), mask_cv2, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return inpainted

def copy_Inpaint(array, mask):
    # Reviewer suggestion - copy previous line inpainter
    inpainted = array.copy()
    rows, cols = array.shape
    for i in range(rows):
        if np.any(mask[i]):
            if i > 0:
                inpainted[i, mask[i]] = array[i - 1, mask[i]]
            elif i < rows - 1:
                inpainted[i, mask[i]] = array[i + 1, mask[i]]
    return inpainted

def CV_inpaint_twochannel(image, mask, missingMask, neighbourhood):
    """
    Inpaint a 2D image (first channel is chemical data, second is segmentation mask),
    Inpainting a missing pixel is achieved using only the pixels of like-classification
    """
    # Operate on copy
    inpaintedImage = np.copy(image)
    rows, cols = image.shape

    # Generate all possible neighbors in a 2n + 1 grid, excluding itself
    neighbours = [(di, dj) for di in range(-neighbourhood, neighbourhood + 1)
                                 for dj in range(-neighbourhood, neighbourhood + 1)
                                 if not (di == 0 and dj == 0)]

    # Iterate over every pixel in the image
    for i in range(rows):
        for j in range(cols):
            # Only process missing pixels
            if missingMask[i, j] == 1:
                # Get the mask value of the current pixel
                region = mask[i, j]
                # Gather like neighbors from the same region
                likeNeigbours = []
                for offset in neighbours:
                    ni, nj = i + offset[0], j + offset[1]
                    if 0 <= ni < rows and 0 <= nj < cols:  # Check bounds
                        if missingMask[ni, nj] == 0 and mask[ni, nj] == region:
                            likeNeigbours.append(inpaintedImage[ni, nj])
                # If there are valid neighbors, inpaint using their average
                if likeNeigbours:
                    inpaintedImage[i, j] = np.mean(likeNeigbours)
    
    return inpaintedImage


# --- Functions for deep learning inpainting --- #

class inputDataset_1C(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = idx
        data = self.data[idx]
        # data is [chem]
        chemicalChannel = torch.Tensor(data).unsqueeze(0)
        label = torch.tensor(int(label))
        inputData = chemicalChannel
        return inputData, label

class inputDataset_2C(Dataset):
    def __init__(self, channel1, channel2):
        self.channel1 = channel1
        self.channel2 = channel2

    def __len__(self):
        return len(self.channel1)

    def __getitem__(self, idx):
        label = idx
        channel1 = self.channel1[idx]
        channel2 = self.channel2[idx]
        chemicalChannel = torch.Tensor(channel1).unsqueeze(0)
        boundaryChannel = torch.Tensor(channel2).unsqueeze(0)
        inputData = torch.cat((chemicalChannel, boundaryChannel), dim=0)
        label = torch.tensor(int(label))
        return inputData, label

"""
def decompose_Patches(array, patchSize):
    # CNN deals with patches of some square size, create these patches
    x, y = array.shape
    n = patchSize
    # Calculate padded dimensions to ensure complete patches
    newX = (x + n - 1) // n * n
    newY = (y + n - 1) // n * n
    paddedArr = np.pad(array, ((0, newX - x), (0, newY - y)), mode='constant')
    # Extract & return patches
    patches = []
    for i in range(0, newX, n):
        for j in range(0, newY, n):
            patch = paddedArr[i:i+n, j:j+n]
            patches.append(patch)
    return patches

def reassemble_Patches(patches, original_size, patchSize):
    # After inpainting, reassemble the patches to original image shape
    x, y = original_size
    n = patchSize
    # Calculate padded dimensions to ensure complete patches
    newX = (x + n - 1) // n * n
    newY = (y + n - 1) // n * n
    fullArr = np.zeros((newX, newY))
    # Place patches back into the full array
    patchIdx = 0
    for i in range(0, newX, n):
        for j in range(0, newY, n):
            fullArr[i:i+n, j:j+n] = patches[patchIdx]
            patchIdx += 1
    # Crop to the original size and return
    return fullArr[:x, :y]
"""

# New patching (avoid edge-effects?)

def decompose_Patches(array, patchSize=64, stride=32):
    x, y = array.shape
    padX = (patchSize - (x - patchSize) % stride - 1) % stride + 1
    padY = (patchSize - (y - patchSize) % stride - 1) % stride + 1
    padded = np.pad(array, ((0, padX), (0, padY)), mode='reflect')

    patches = []
    for i in range(0, padded.shape[0] - patchSize + 1, stride):
        for j in range(0, padded.shape[1] - patchSize + 1, stride):
            patch = padded[i:i+patchSize, j:j+patchSize]
            patches.append(patch.copy())
    return patches

def reassemble_Patches(patches, original_size, patchSize=64, stride=32):
    x, y = original_size
    # Compute size after padding
    padX = (patchSize - (x - patchSize) % stride - 1) % stride + 1
    padY = (patchSize - (y - patchSize) % stride - 1) % stride + 1
    paddedX = x + padX
    paddedY = y + padY

    full = np.zeros((paddedX, paddedY), dtype=np.float32)
    weight = np.zeros((paddedX, paddedY), dtype=np.float32)

    idx = 0
    for i in range(0, paddedX - patchSize + 1, stride):
        for j in range(0, paddedY - patchSize + 1, stride):
            full[i:i+patchSize, j:j+patchSize] += patches[idx]
            weight[i:i+patchSize, j:j+patchSize] += 1
            idx += 1

    # Avoid division by zero
    weight[weight == 0] = 1
    full /= weight

    return full[:x, :y]



def nn_inpaint_1Channel(arr, mask, model):
    # Break down into patches and save patches    
    patchSize = 64
    patches = decompose_Patches(arr, patchSize)
    # Load patches into net-loader
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inputset = inputDataset_1C(patches)#, transform=transform_train)
    # Create DataLoader
    batchSize = len(patches)
    inputloader = torch.utils.data.DataLoader(inputset, batch_size=batchSize, shuffle=False)
    for input, batchLabels in inputloader:
        # Initalise and apply model
        with torch.no_grad():
          outputs = model(input)
    chemPredicts = []
    for i in range(len(outputs)):
        patch = outputs[i].detach().cpu().numpy()
        chemPredicts.append(patch[0])
    assembled = reassemble_Patches(chemPredicts, arr.shape, patchSize)
    assembled = Utils.clipNeg(assembled)
    # Re-overlay original data    
    assembled[~mask] = arr[~mask]    
    return assembled

def nn_inpaint_2Channel(channel1, channel2, mask, model):
    # Break down into patches and save patches
    patchSize = 64
    patchesC1 = decompose_Patches(channel1, patchSize)
    patchesC2 = decompose_Patches(channel2, patchSize)
    
    # Load patches into net-loader
    inputset = inputDataset_2C(patchesC1, patchesC2)#, transform=transform_train)

    # Create DataLoader
    batchSize = len(inputset)
    inputloader = torch.utils.data.DataLoader(inputset, batch_size=batchSize, shuffle=False)
    
    for modelInput, batchLabels in inputloader:
        # Forward pass
        with torch.no_grad():
          outputs = model(modelInput)
    chemPredicts = []
    for i in range(len(outputs)):
        patch = outputs[i].detach().cpu().numpy()
        chemPredicts.append(patch[0])
    assembled = reassemble_Patches(chemPredicts, channel1.shape, patchSize)
    # Remove 0s from inpainted
    assembled = Utils.clipNeg(assembled)
    # Re-overlay original data
    assembled[~mask] = channel1[~mask]     
    return assembled