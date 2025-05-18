import numpy as np
import os
import csv
import torch
import Utils
import Inpainters
import NNs

targetName = "Murrina"
chemPath = 'Data/Measured_Data/Glass/Application Data/Chem.npy'
segPath = 'Data/Measured_Data/Glass/Application Data/Mask.npy'

# Make output folders
currentDir = os.path.dirname(os.path.abspath(__file__))
outPath = os.path.join(os.path.dirname(currentDir), f"Outputs/{targetName}/")

resultsPath = os.path.join(outPath, 'metrics.txt')

Utils.read_Plot_Metrics(resultsPath, f'{outPath}Performance_Plot')