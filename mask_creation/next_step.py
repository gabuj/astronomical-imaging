from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

path="fits_file/mosaic.fits"
#parameters to calcualte background:
num_bins = 10000


def gaussian(x, a, std, m):
    return a *(1/(np.sqrt(2*np.pi)*std))* np.exp(-0.5 * ((x - m) / std) ** 2)
#open file
hdulist = fits.open(path)

data = hdulist[0].data

# Calculate histogram bin counts and bin edges
counts, bin_edges = np.histogram(data.ravel(), bins=num_bins)

# Calculate bin centers as the midpoint between each bin edge
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

#create mask
data_copy = np.copy(data)






#close file
hdulist.close()