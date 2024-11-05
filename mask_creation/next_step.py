from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, label
# from photutils import aperture_photometry, CircularAperture
path="fits_file/mosaic.fits"
#get threshold from other program!!!!!!!!
threshold = 3481

#open file and get data
hdulist = fits.open(path)
data = hdulist[0].data


#create mask
mask= data > threshold

#close fits file
hdulist.close()