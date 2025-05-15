"""
Use this script to generate a training data set for the NN inpainters.
It reads spatially co-registered pairs of segmentation masked and chemical data
and generates patches from these data.

Use the inputFolder argument of create_Inpainter_Training() to point to the desired folder,
which should contain a pair of .npy files with the overlapping regions.

In the path specified by outputFolder, a test and train folder will be made containing data
suitable for training the inpainting NNs

"""

import os
import numpy as np
import random
  
def create_Inpainter_Training(inputFolder, outputFolder, notUse):
    """
    Reads pairs of chemcal and mask .npy files in subfolders
    of inputFolder
    """
    for entry in os.scandir(inputFolder):
        if entry.is_dir() and notUse not in entry.path:
            # Load data
            # Get chem and clip zeros
            chem = np.load(f'{entry.path}/_Overlap_Chem.npy')
            chem = np.where(chem < 0, 0, chem)
            # Get mask with correct 1s/0s arrangement
            GBMask = np.load(f'{entry.path}/_Overlap_Mask.npy')

            # Generate patches from the array
            chemPatches = create_Patches(chem)
            maskPatches = create_Patches(GBMask)
                            
            # Saves data in form [chempatch, mask, GBmask]
            create_Dataset(chemPatches, maskPatches, 0.9, entry.path, outputFolder)

def create_Patches(array, patchSize=64, overlap=30):
    """Chops up input array into patches of the specified patch size 
    and overlap."""
    # Inialise array to collect patches
    patches = []
    # Iterate over array, with strides corresponding to the specified
    # overlap. Iterate along both rows and columns
    for i in range(0, array.shape[0], patchSize - overlap):
        for j in range(0, array.shape[1], patchSize - overlap):
            # Calculate patch end pixel coordinate
            iEnd = i + patchSize
            jEnd = j + patchSize
            # Extract the patch at specified place
            if iEnd > array.shape[0] or jEnd > array.shape[1]:
                # Initalise empty 0s patch then copy data on top
                patch = np.zeros((patchSize, patchSize))
                iPatch = min(array.shape[0] - i, patchSize)
                jPatch = min(array.shape[1] - j, patchSize)
                patch[:iPatch, :jPatch] = array[i:i+iPatch, j:j+jPatch]
            else:
                patch = array[i:iEnd, j:jEnd]
            # Collect patches in array
            patches.append(patch)
    # Return all patches
    return patches
                
def create_Dataset(patches, structurePatches, trainRatio, fileName, saveFolder):
    """Takes patched data and generates masks used to obscure data during inpainting
    training for nns. Saves the training data, ready to be loaded during nn training.
    Data is saved in array with each entry in form [chemical, obscure mask, GB mask]"""
    patches = generate_Masks(patches, structurePatches)
    # Calculate the number of training patches
    numTrain = int(len(patches) * trainRatio)
    # Shuffle place of patches in list so test/train are mixed
    random.shuffle(patches)    
    # Split all patches into subsets, training and testing
    train = patches[:numTrain]
    test = patches[numTrain:]
    # Create save directories if they don't exist
    trainDir = f'{saveFolder}/Train'
    testDir = f'{saveFolder}/Test'

    os.makedirs(trainDir, exist_ok=True)
    os.makedirs(testDir, exist_ok=True)
    # Save training patches
    for i, patch in enumerate(train):
        np.save(os.path.join(trainDir, f'{os.path.basename(os.path.splitext(fileName)[0])}_patch{i}.npy'), patch)
    # Save testing patches
    for i, patch in enumerate(test, start=numTrain):
        np.save(os.path.join(testDir, f'{os.path.basename(os.path.splitext(fileName)[0])}_patch{i}.npy'), patch)
    return test, train

def generate_Masks(data, structurePatches):
    """Generates obscuring masks to apply to data during inpainting training. Masks
    are binary patches, with rows set to 1s (obscure) or 0s (keep)"""
    patches = []
    for i, datum in enumerate(data):
        for spacing in range (1, 6):
            mask = np.zeros_like(datum)
            mask = striped_Mask(mask, [spacing], start = random.randint(0, 2), setTo = 1)
            patches.append([datum, mask, structurePatches[i]])
            mask = np.ones_like(datum)
            mask = striped_Mask(mask, [spacing], start = random.randint(0, 2), setTo = 0)
            patches.append([datum, mask, structurePatches[i]])
    # Return list in form [chemical, obscure mask, GB mask]        
    return patches

def striped_Mask(arr, indicies, start = 0, setTo = 0):
    for value in indicies:    
        arr[start::value, :] = setTo
    return arr

# Setup to generate dataset for ice data

# Find current directory path
currentDir = os.path.dirname(os.path.abspath(__file__))

# Construct path to data folder
dataDir = os.path.join(os.path.dirname(currentDir), "Data/Measured_Data/Ice/40mic")

# Create path for saving output data
saveDir = os.path.join(os.path.dirname(currentDir), "Data/Model_Training_Data/Ice")
if not os.path.exists(saveDir):
    os.makedirs(saveDir)

# Run function to generate dataset
toOmit = "EDC_1994-6-A2" # flag one folder to not use. This is the test data
create_Inpainter_Training(dataDir, saveDir, toOmit)