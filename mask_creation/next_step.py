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

print(binary_mask)

#close fits file
hdulist.close()