import numpy as np
from scipy.ndimage import label
import matplotlib.pyplot as plt
def takeou_bleeing(data,bleeding_centers,threshold,background_value):
    #get rid of bleeding regions
    #define bleeding region

    # Create a mask for the bleeding regions
    mask = np.zeros_like(data, dtype=bool)
    #create binary mask to see when over the threshold
    binary_mask= data > threshold
    # Create a mask for the bleeding regions
    bleeding_region = np.zeros_like(data, dtype=bool)

    #create labels and take aaway all pixels that are true in the binary mask that are part of the same label
    labels, num_labels = label(binary_mask)
 
    
    print(f"Number of labels: {num_labels}")
    for centers in bleeding_centers:
        #find label they belong to
        y, x = centers
        y, x = int(y), int(x)
        star_label = labels[y, x]
        print(f"Star label: {star_label}")
        #take away all pixels that are part of the same label
        bleeding_region[labels==star_label]=True


    galaxy_mask=np.copy(data)
    galaxy_mask[bleeding_region==True]=background_value
    
    #show galaxy mask
    plt.imshow(bleeding_region, cmap='gray', origin='lower')
    plt.colorbar()
    plt.title('bleeding_region')
    plt.show()
    return galaxy_mask

    #galaxy_mask is data without the bleeding regions