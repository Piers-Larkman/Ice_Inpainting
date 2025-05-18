""" Demonstration of inpainting techniques applied to LA-ICP-MS data. Shows
the validation and application of 4 techniques:

1CV - uses the telea algorithm
2MR - replaces missing pixels with mean of nearby points, conditioned on a mask derived from optical data
1NN - a trained neural network to replace missing pixels
2NN - a trained neural network conditioned also on optical mask
"""

import numpy as np
import os
import csv
import torch
import Utils
import Inpainters
import NNs
import time
from NNs import CustomNet_2Channel, CustomNet_1Channel
torch.serialization.add_safe_globals([NNs.CustomNet_2Channel])
torch.serialization.add_safe_globals([NNs.CustomNet_1Channel])

# --- Data loading --- #

# Choose case to see validation, glass or ice
ice = 0
glass = 1

if glass == 1:
        
    targetName = "Murrina"
    chemPath = 'Data/Measured_Data/Glass/Application Data/Chem.npy'
    segPath = 'Data/Measured_Data/Glass/Application Data/Mask.npy'

    # Make output folders
    currentDir = os.path.dirname(os.path.abspath(__file__))
    outPath = os.path.join(os.path.dirname(currentDir), f"Outputs/{targetName}/")
    if not os.path.isdir(outPath):
        os.makedirs(outPath)

    chem = np.load(chemPath)[1:,1:]
    seg = np.load(segPath)[1:,1:]

    modelOneChannelFilePath = os.path.join(os.path.dirname(currentDir), "Models/2025-05-17_21_Glass_1C_NN_Simple/model_Weights/epoch20.pth")
    modelOneChannel = torch.load(modelOneChannelFilePath, weights_only=False)
    modelOneChannel.eval()

    modelTwoChannelFilePath = os.path.join(os.path.dirname(currentDir), "Models/2025-05-17_20_Glass_2C_NN_Simple/model_Weights/epoch20.pth")
    modelTwoChannel = torch.load(modelTwoChannelFilePath, weights_only=False)
    modelTwoChannel.eval()

if ice == 1:
    targetName = "20231011_EDC_1994-6-A2"
    chemPath = 'Data/Measured_Data/Ice/40mic/20231011_EDC_1994-6-A2/_Overlap_Chem.npy'
    GBPath = 'Data/Measured_Data/Ice/40mic/20231011_EDC_1994-6-A2/_Overlap_Mask.npy'

    chem = np.load(chemPath)[20:-20,20:-20]
    seg = np.load(GBPath)[20:-20,20:-20]

    # Make output folders
    currentDir = os.path.dirname(os.path.abspath(__file__))
    outPath = os.path.join(os.path.dirname(currentDir), f"Outputs/{targetName}/")
    if not os.path.isdir(outPath):
        os.makedirs(outPath)

    # Load ML models
    modelTwoChannelFilePath = os.path.join(os.path.dirname(currentDir), "Models/2025-05-17_20_Ice_2C_NN_Simple_/model_Weights/epoch40.pth")
    modelTwoChannel = torch.load(modelTwoChannelFilePath, weights_only=False)
    modelTwoChannel.eval()

    modelOneChannelFilePath = os.path.join(os.path.dirname(currentDir), "Models/2025-05-17_21_Ice_1C_NN_Simple_/model_Weights/epoch23.pth")
    modelOneChannel = torch.load(modelOneChannelFilePath, weights_only=False)
    modelOneChannel.eval()

Utils.plot_save_array(chem, f'{outPath}Chemical_Input', 'Original Chemical Data')
Utils.plot_save_array(seg, f'{outPath}Segmentation_Input', 'Optical Segmentation')

# --- Data processing --- #
# Remove/keep every nth line
ns = [2,3,4,5]
results  = []

timeTaken = [] #stores [approach, ID, time taken]

for n in ns:
    # Remove every nth line
    if n != 2:
        chemSparse, mask = Utils.remove_every_nth_line(chem, n)
        # Plot the arrays with removed lines
        Utils.plot_save_array(chemSparse, f'{outPath}density_{1-1/n:.2f}', f'Array with information density {1-1/n:.2f}')
        Utils.plot_save_array(mask, f'{outPath}density_{1-1/n:.2f}_Mask', f'Mask with information density {1-1/n:.2f}')

        # Inpaint the missing data
        t1 = time.time()
        cvInpaintedOneChannel = Inpainters.CV_inpaint_onechannel(chemSparse, ~mask)   
        t2 = time.time()
        timeTaken.append(["1C Telea", 1-1/n, t2-t1])
        cvInpaintedTwoChannel = Inpainters.CV_inpaint_twochannel(chemSparse, seg, ~mask, 1)
        t1 = time.time()
        timeTaken.append(["2C MR", 1-1/n, t1-t2])
        nnInpaintedOneChannel = Inpainters.nn_inpaint_1Channel(chemSparse, ~mask, modelOneChannel)
        t2 = time.time()
        timeTaken.append(["1C NN", 1-1/n, t2-t1])
        nnInpaintedTwoChannel = Inpainters.nn_inpaint_2Channel(chemSparse, seg, ~mask, modelTwoChannel) 
        t1 = time.time()
        timeTaken.append(["2C NN", 1-1/n, t1-t2]) 
        #copyInpainted = Inpainters.copy_Inpaint(chemSparse, ~mask)

        # Plot the inpainted array
        Utils.plot_save_array(cvInpaintedOneChannel, f'{outPath}1Channel_CV_Inpainted_Density{1-1/n:.2f}', f'1C Tela inpainted. Input density {1-1/n:.2f}')
        Utils.plot_save_array(cvInpaintedTwoChannel, f'{outPath}2Channel_CV_Inpainted_Density{1-1/n:.2f}', f'2C MR inpainted. Input density {1-1/n:.2f}')
        Utils.plot_save_array(nnInpaintedOneChannel, f'{outPath}1Channel_nn_Inpainted_Density{1-1/n:.2f}', f'1C NN channel nn inpainted. Input density {1-1/n:.2f}')
        Utils.plot_save_array(nnInpaintedTwoChannel, f'{outPath}2Channel_nn_Inpainted_Density{1-1/n:.2f}', f'2C NN channel nn inpainted. Input density {1-1/n:.2f}')
    
        # Calculate and store metrics
        mse, ssim_index = Utils.calculate_Metrics(chem, cvInpaintedOneChannel, mask)
        results.append(("1C Tela", 1-1/n, mse, ssim_index))
        mse, ssim_index = Utils.calculate_Metrics(chem, cvInpaintedTwoChannel, mask)
        results.append(("2C MR", 1-1/n, mse, ssim_index))
        mse, ssim_index = Utils.calculate_Metrics(chem, nnInpaintedOneChannel, mask)
        results.append(("1C NN", 1-1/n, mse, ssim_index))
        mse, ssim_index = Utils.calculate_Metrics(chem, nnInpaintedTwoChannel, mask)
        results.append(("2C NN", 1-1/n, mse, ssim_index))
        #mse, ssim_index = Utils.calculate_Metrics(chem, copyInpainted, mask)
        #results.append(("Copy", 1-1/n, mse, ssim_index))
    
    # WITH KEPT LINES
    # Keep only every nth line
    chemSparse, mask = Utils.keep_every_nth_line(chem, n)

    # Plot the arrays with removed lines
    Utils.plot_save_array(chemSparse, f'{outPath}density_{1/n:.2f}', f'Array with information density {1/n:.2f}')
    Utils.plot_save_array(mask, f'{outPath}density_{1/n:.2f}_Mask', f'Mask with information density {1/n:.2f}')

    # Inpaint the missing data
    t1 = time.time()
    cvInpaintedOneChannel = Inpainters.CV_inpaint_onechannel(chemSparse, ~mask)
    t2 = time.time()
    timeTaken.append(["1C Telea", 1/n, t2-t1])
    cvInpaintedTwoChannel = Inpainters.CV_inpaint_twochannel(chemSparse, seg, ~mask, int((n)/2))
    t1 = time.time()
    timeTaken.append(["2C MR", 1/n, t1-t2])
    nnInpaintedOneChannel = Inpainters.nn_inpaint_1Channel(chemSparse, ~mask, modelOneChannel)
    t2 = time.time()
    timeTaken.append(["1C NN", 1/n, t2-t1])
    nnInpaintedTwoChannel = Inpainters.nn_inpaint_2Channel(chemSparse, seg, ~mask, modelTwoChannel)
    t1 = time.time()
    timeTaken.append(["2C NN", 1/n, t1-t2])
    #copyInpainted = Inpainters.copy_Inpaint(chemSparse, ~mask)

    # Plot the inpainted array
    Utils.plot_save_array(cvInpaintedOneChannel, f'{outPath}1Channel_CV_Inpainted_Density{1/n:.2f}', f'1C Tela inpainted. Input density {1/n:.2f}')
    Utils.plot_save_array(cvInpaintedTwoChannel, f'{outPath}2Channel_CV_Inpainted_Density{1/n:.2f}', f'2C MR inpainted. Input density {1/n:.2f}')
    Utils.plot_save_array(nnInpaintedOneChannel, f'{outPath}1Channel_nn_Inpainted_Density{1/n:.2f}', f'1C NN inpainted. Input density {1/n:.2f}')
    Utils.plot_save_array(nnInpaintedTwoChannel, f'{outPath}2Channel_nn_Inpainted_Density{1/n:.2f}', f'2C NN inpainted. Input density {1/n:.2f}')

    # Calculate and store metrics
    mse, ssim_index = Utils.calculate_Metrics(chem, cvInpaintedOneChannel, mask)
    results.append(("1C Tela", 1/n, mse, ssim_index))
    mse, ssim_index = Utils.calculate_Metrics(chem, cvInpaintedTwoChannel, mask)
    results.append(("2C MR", 1/n, mse, ssim_index))
    mse, ssim_index = Utils.calculate_Metrics(chem, nnInpaintedOneChannel, mask)
    results.append(("1C NN", 1/n, mse, ssim_index))
    mse, ssim_index = Utils.calculate_Metrics(chem, nnInpaintedTwoChannel, mask)
    results.append(("2C NN", 1/n, mse, ssim_index))
    #mse, ssim_index = Utils.calculate_Metrics(chem, copyInpainted, mask)
    #results.append(("Copy", 1/n, mse, ssim_index))

# Create two reference arrays - all 0s and all 1s - and measure their ssim/mse
reference = np.ones_like(chem)

mse, ssim_index = Utils.calculate_Metrics(chem, reference, mask)
results.append(("Zeros", 1, mse, ssim_index))

resultsPath = os.path.join(outPath, 'metrics.txt')
with open(resultsPath, 'w', newline='') as f:
    writer = csv.writer(f)
    # Write the header
    writer.writerow(['Approach', 'Density', 'MSE', 'SSIM'])
    # Write the data rows
    writer.writerows(results)

Utils.read_Plot_Metrics(resultsPath, f'{outPath}Performance_Plot')

resultsPath = os.path.join(outPath, 'times.txt')
with open(resultsPath, 'w', newline='') as f:
    writer = csv.writer(f)
    # Write the header
    writer.writerow(['Approach', 'Density', 'Time'])
    # Write the data rows
    writer.writerows(timeTaken)

Utils.read_Plot_Times(resultsPath, f'{outPath}Times_Plot')


