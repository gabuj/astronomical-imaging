from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label, find_objects
import pandas as pd

# from photutils import aperture_photometry, CircularAperture
path="fits_file/mosaic.fits"
#get threshold from other program!!!!!!!!
threshold = 3481

#open file and get data
hdulist = fits.open(path)
data = hdulist[0].data


#create mask
binary_mask= data > threshold

# Use label to identify connected regions in the binary mask
labeled_image, num_galaxies = label(binary_mask)

print(f"Number of detected objects (galaxies): {num_galaxies}")

print(labeled_image)

#close fits file
hdulist.close()