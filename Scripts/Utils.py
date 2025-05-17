   
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim
import math

""" Plotting functions """
def plot_Batch(inputs, truths, masks, savePath = None):
  # Assuming batch is of dims [batch Size, 2, X,Y] ONLY 2 CHANNELS
  # Create grid of 2 by 2
  # Plot Truths
  fig, axes = plt.subplots(2, 2)
  for i, ax in enumerate(axes.flatten()):
    ax.imshow(truths[i][0].numpy())
  plt.suptitle(f'Chemical channel (Ground truth) for first 4 images in batch of size {len(inputs)}')
  plt.show()
  if savePath != None:
      plt.savefig(f'{savePath}_Batch_Chemicals.png', bbox_inches='tight')
      plt.close()

  # Plot masked chemicals
  fig, axes = plt.subplots(2, 2)
  for i, ax in enumerate(axes.flatten()):
    ax.imshow(inputs[i][0].numpy())
  plt.suptitle(f'Masked chemical data for first 4 images in batch of size {len(inputs)}')
  plt.show()
  if savePath != None:
      plt.savefig(f'{savePath}_Batch_MaskedChems.png', bbox_inches='tight')
      plt.close()

  # Plot GBs
  fig, axes = plt.subplots(2, 2)
  for i, ax in enumerate(axes.flatten()):
    ax.imshow(inputs[i][1].numpy())
  plt.suptitle(f'Grain boundary data for first 4 images in batch of size {len(inputs)}')
  plt.show()
  if savePath != None:
      plt.savefig(f'{savePath}_Batch_GBs.png', bbox_inches='tight')
      plt.close()

  # Plot masks
  fig, axes = plt.subplots(2, 2)
  for i, ax in enumerate(axes.flatten()):
    ax.imshow(masks[i][0].numpy())
  plt.suptitle(f'Masks for first 4 images in batch of size {len(inputs)}')
  plt.show()
  if savePath != None:
      plt.savefig(f'{savePath}_Batch_Masks.png', bbox_inches='tight')
      plt.close()
      
  return 0

def plot_Test(reconstructeds, truths, inputs, masks, title = None, save = None):
        
    offsets = [0, 8]
    for offset in offsets:
      
        fig, axs = plt.subplots(3, 4, figsize=(10, 8))
          
        for i in range(offset,offset+3):
            axs[i-offset][0].imshow(inputs[i][0].numpy())
            axs[i-offset][1].imshow(inputs[i][1].numpy())
            axs[i-offset][2].imshow(reconstructeds[i][0].numpy())
            axs[i-offset][3].imshow(truths[i][0].numpy())
          
        subtitles = ["Input chemical", "Input boundaries", "Prediction", "Target"]
        for i, ax in enumerate(axs[0]):
            ax.set_title(subtitles[i])
            
        row_labels = ['Example 1', 'Example 2', 'Example 3']
        for i, label in enumerate(row_labels):
            fig.text(0.07, 0.75 - i*0.25, label, ha='center', va='center', fontsize=12, rotation=90)
          
        for ax in axs.flatten():
          ax.set_xticks([])
          ax.set_yticks([])
            
        plt.suptitle(title)
          
        if save !=0:
          plt.savefig(f'{save}_{offset}.png', dpi=300)
          plt.close()

    return 0


""" Comparison metrics """

# Function to calculate SSIM over a batch
def calculate_ssim(arr1, arr2):
    SSIMs = []

    for i in range(arr1.shape[0]):
        arr1NP = arr1[i].detach().cpu().numpy()
        arr2NP = arr2[i].detach().cpu().numpy()
        val = ssim(arr1NP, arr2NP, data_range=arr2NP.max() - arr2NP.min())
        # Ignore if SSIM is undefined (ususally as the ground truth is all 0s)
        if not np.isnan(val):
            SSIMs.append(val)
    # Return mean SSIM in batch

    print(np.mean(SSIMs))
    return np.mean(SSIMs)