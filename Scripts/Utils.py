import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
import os

""" Plotting functions """
def plot_Test_2C(reconstructeds, truths, inputs, masks, title = None, save = None):
        
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


def plot_Test_1C(reconstructeds, maskedChemical, groundTruth, mask, title = None, save = None):
    
    offsets = [4,8]
    
    for offset in offsets:
        fig, axs = plt.subplots(3, 3, figsize=(10, 8))

        for i in range(offset,offset+3):
            axs[i-offset][0].imshow(maskedChemical[i][0].numpy())
            axs[i-offset][1].imshow(reconstructeds[i][0].numpy())
            axs[i-offset][2].imshow(groundTruth[i][0].numpy())
          
        subtitles = ["Input chemical", "Prediction", "Target"]
        for i, ax in enumerate(axs[0]):
            ax.set_title(subtitles[i])
            
        row_labels = ['Example 1', 'Example 2', 'Example 3']
        for i, label in enumerate(row_labels):
            fig.text(0.135, 0.77 - i*0.275, label, ha='center', va='center', fontsize=12, rotation=90)
          
        for ax in axs.flatten():
          ax.set_xticks([])
          ax.set_yticks([])
            
        plt.suptitle(title)
          
        if save !=0:
          plt.savefig(f'{save}_{offset}.png', dpi=300)
         
        plt.close()


def plot_save_array(array, saveName, title='Array'):
    np.save(saveName + '.npy', array)
    plt.figure(figsize = (12,10))
    #array = np.clip(array, np.min(array)*1.1, np.max(array)*0.1)
    plt.imshow(array, cmap='inferno', interpolation='nearest')
    plt.title(title)
    #plt.colorbar()
    plt.axis('off') 
    ax = plt.gca()
    add_Scale(ax)
    #plt.show()
    plt.savefig(saveName + '.png', dpi = 400, bbox_inches='tight', pad_inches=0)
    plt.close()


def add_Scale(ax, pixSize = 40, loc = 30):

    barLength = 25  # Scale bar length in pixels
    barLengthUnits = barLength * pixSize  # Convert length to physical units
    
    # Set location
    ylim = ax.get_ylim()
    bary = ylim[0] - loc
    barx = 10
    scalebar_width = 4

    # Draw bar and add text
    ax.hlines(bary, barx, barx + barLength, colors="white", linewidth=scalebar_width)

    ax.text(barx + barLength / 2, bary - 3, f"{barLengthUnits/1000:.0f} mm", color="white",
        ha="center", va="bottom", fontsize=26)
        
    return 0

""" Comparison metrics """

# Function to calculate SSIM over a batch
def calculate_ssim(arr1, arr2):

    SSIMs = []
    for i in range(arr1.shape[0]):
        arr1NP = arr1[i].detach().cpu().numpy()
        arr2NP = arr2[i].detach().cpu().numpy()

        # SSIM is undefined for a constant array, exclude case where
        # measured array is all 0s
        if not np.all(arr2NP == 0):
            val = ssim(arr1NP, arr2NP, data_range=arr2NP.max() - arr2NP.min())
            SSIMs.append(val)
    # Return mean SSIM in batch
    return np.mean(SSIMs)

def read_Plot_Times(resultsPath, savePath):
    infoDensity = []
    times = []
    approach = []
    with open(resultsPath, 'r') as file:
        for line in file:
            if not line.startswith('Approach'):
                # Parse the line
                parts = line.split(',')
                infoDensity.append(float(parts[1]))
                times.append(float(parts[2]))
                approach.append(parts[0])
    
    infoDensity = np.array(infoDensity)
    times = np.array(times)
    approach = np.array(approach)

    # Isolate approaches
    approaches = np.unique(approach)
    colours = plt.cm.get_cmap('tab10', len(approaches))
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']

    plt.figure(figsize=(20, 12))

    for i, label in enumerate(approaches):
        mask = (approach == label)
        if label != "Zeros":
            mask = (approach == label)
            flteredX, filteredY = infoDensity[mask], times[mask]
            # Order by x values
            sorted_indices = np.argsort(flteredX)
            orderedX = flteredX[sorted_indices]
            orderedY = filteredY[sorted_indices]
            # Plot ordered data
            plt.plot(orderedX, orderedY, color=colours(i), label=f'{label}', marker=markers[i])

        else:
            y = times[mask]

    #plt.axhline(y=y, color='r', linestyle='--', label='All 0s')
    
    plt.xlabel('Information density', fontsize=28)
    plt.ylabel('Time taken (s)', fontsize=28)
    plt.legend(title='Inpainting approach', title_fontsize=30, fontsize=28)
    plt.grid(True)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.savefig(savePath + '_times.png', dpi = 400, bbox_inches='tight', pad_inches=0)

    return 0

def read_Plot_Metrics(metricsPath, savePath):
    ssim = []
    infoDensity = []
    mse = []
    approach = []
    with open(metricsPath, 'r') as file:
        for line in file:
            if not line.startswith('Approach'):
                # Parse the line
                parts = line.split(',')
                infoDensity.append(float(parts[1]))
                mse.append(float(parts[2]))
                ssim.append(float(parts[3]))
                approach.append(parts[0])
    
    infoDensity = np.array(infoDensity)
    ssim = np.array(ssim)
    mse = np.array(mse)
    approach = np.array(approach)

    # Isolate approaches
    approaches = np.unique(approach)
    colours = plt.cm.get_cmap('tab10', len(approaches))
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']

    plt.figure(figsize=(20, 12))

    for i, label in enumerate(approaches):
        mask = (approach == label)
        if label != "Zeros":
            mask = (approach == label)
            flteredX, filteredY = infoDensity[mask], mse[mask]
            # Order by x values
            sorted_indices = np.argsort(flteredX)
            orderedX = flteredX[sorted_indices]
            orderedY = filteredY[sorted_indices]
            # Plot ordered data
            plt.plot(orderedX, orderedY, color=colours(i), label=f'{label}', marker=markers[i])

        else:
            y = mse[mask]

    #plt.axhline(y=y, color='r', linestyle='--', label='All 0s')
    
    plt.xlabel('Information density', fontsize=28)
    plt.ylabel('MSE', fontsize=28)
    plt.legend(title='Inpainting approach', title_fontsize=30, fontsize=28)
    plt.grid(True)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.savefig(savePath + '_MSE.png', dpi = 400, bbox_inches='tight', pad_inches=0)

    #plt.show()

    plt.figure(figsize=(20, 12))
    for i, label in enumerate(approaches):
        mask = (approach == label)
        if label != "Zeros":
            mask = (approach == label)
            flteredX, filteredY = infoDensity[mask], ssim[mask]
            # Order by x values
            sorted_indices = np.argsort(flteredX)
            orderedX = flteredX[sorted_indices]
            orderedY = filteredY[sorted_indices]
            # Plot ordered data
            plt.plot(orderedX, orderedY, color=colours(i), label=f'{label}', marker=markers[i])
        else:
            y=ssim[mask]
    #plt.axhline(y=y, color='r', linestyle='--', label='All 0s')

    plt.xlabel('Information density', fontsize=28)
    plt.ylabel('SSIM', fontsize=28)
    plt.legend(title='Inpainting approach', title_fontsize=30, fontsize=28)
    plt.grid(True)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.savefig(savePath + '_SSIM.png', dpi = 400, bbox_inches='tight', pad_inches=0)

    #plt.show()
    return 0


""" Applying inpainting functions """

# Metrics to quantify inpainting quality
def calculate_Metrics(original, inpainted, mask):
    # Calculate the data range
    dataRange = original.max() - original.min()
    # Calculate MSE, only over inpainted region (defined by mask)
    mse = mean_squared_error(original[mask == 0], inpainted[mask == 0])
    # Calculate SSIM, over full image (requires full spatial context)
    sim, _ = ssim(original, inpainted, data_range=dataRange, full=True)
    return mse, sim

# Function to remove every nth line
def remove_every_nth_line(array, n):
    mask = np.ones_like(array, dtype=bool)
    mask[::n, :] = False
    reduced = array.copy()
    reduced[~mask] = 0
    return reduced, mask

# Function to keep only every nth line
def keep_every_nth_line(array, n):
    mask = np.zeros_like(array, dtype=bool)
    mask[::n, :] = True
    reduced = np.zeros_like(array)
    reduced[mask] = array[mask]
    return reduced, mask

def clipNeg(im):
    return np.where(im < 0, 0, im)