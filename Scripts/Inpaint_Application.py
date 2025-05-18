import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import NNs
import Utils
import Inpainters
from NNs import CustomNet_2Channel, CustomNet_1Channel
torch.serialization.add_safe_globals([NNs.CustomNet_2Channel])
torch.serialization.add_safe_globals([NNs.CustomNet_1Channel])

### MAIN ###
if __name__ == '__main__': 

    channel = "56Fe cps"

    currentDir = os.path.dirname(os.path.abspath(__file__))
    inPath = os.path.join(os.path.dirname(currentDir), f"Data/Measured_Data/Glass/Training_Demo/{channel}")

    chemDat = np.load(f"{inPath}/section_0_Chemical.npy")

    densities = [1, 0.5, 0.25, 0.33, 0.13, 0.2, 0.33, 0.5]

    norm = [np.percentile(chemDat, 5), np.percentile(chemDat, 95)]
        
    modelOneChannelFilePath = os.path.join(os.path.dirname(currentDir), "Models/2025-05-17_21_Glass_1C_NN_Simple/model_Weights/epoch20.pth")
    modelOneChannel = torch.load(modelOneChannelFilePath, weights_only=False)
    modelOneChannel.eval()

    modelTwoChannelFilePath = os.path.join(os.path.dirname(currentDir), "Models/2025-05-17_20_Glass_2C_NN_Simple/model_Weights/epoch20.pth")
    modelTwoChannel = torch.load(modelTwoChannelFilePath, weights_only=False)
    modelTwoChannel.eval()
    
    # For each chem, load a certain channel, inpaint the missing data
    
    for q in range(6):
                                
        chemDat = np.load(f"{inPath}/section_{q+1}_Chemical.npy")
        regionsMask = np.load(f"{inPath}/section_{q+1}_Mask.npy").astype(np.uint8)  
        mask = np.isnan(chemDat).astype(np.uint8)
        
        column = mask[:, 1]
        zerosCount = np.sum(column == 0)
        dataPercent = (zerosCount / len(column)) * 100
                
        toInpaint = np.nan_to_num(chemDat, nan = 0.0)
        
        cvInpaintedOneChannel = Inpainters.CV_inpaint_onechannel(toInpaint, mask)
        
        nnInpaintedOneChannel = Inpainters.nn_inpaint_1Channel(toInpaint, mask, modelOneChannel)
        
        cvInpaintedTwoChannel = Inpainters.CV_inpaint_twochannel(toInpaint, regionsMask, mask, int(1/densities[q]))
        
        nnInpaintedTwoChannel = Inpainters.nn_inpaint_2Channel(toInpaint, regionsMask, mask, modelTwoChannel)

        approaches = ["1CV", "1NN", "2CV", "2NN"]
        for i, arr in enumerate([cvInpaintedOneChannel, nnInpaintedOneChannel, cvInpaintedTwoChannel, nnInpaintedTwoChannel]):
            plt.figure(figsize = (12,10))
            #array = np.clip(array, np.min(array)*1.1, np.max(array)*0.1)
            plt.imshow(arr, cmap='inferno', interpolation='nearest')
            plt.axis('off') 
            Utils.add_Scale(plt.gca())
            plt.savefig(approaches[i] + '.png', dpi = 400, bbox_inches='tight', pad_inches=0)
                
        np.save(f'Processed_Data/Inpainted/Measured_{channel}_{q}', toInpaint)
        np.save(f'Processed_Data/Inpainted/CV1_Inpainted_{channel}_{q}', cvInpaintedOneChannel)
        np.save(f'Processed_Data/Inpainted/NN1_Inpainted_{channel}_{q}', nnInpaintedOneChannel)
        np.save(f'Processed_Data/Inpainted/NN2_Inpainted_{channel}_{q}', nnInpaintedTwoChannel)
        np.save(f'Processed_Data/Inpainted/CV2_Inpainted_{channel}_{q}', cvInpaintedTwoChannel)


