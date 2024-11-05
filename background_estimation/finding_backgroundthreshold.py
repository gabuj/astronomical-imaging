from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

path="Astro/Astro/Fits_Data/mosaic.fits"
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

# Plot histogram of pixel intensities
# plt.figure(figsize=(10, 6))
# plt.plot(bin_centers, counts, color='black', ls='-',alpha=0.8)
# plt.yscale('log')  # Log scale to better visualize differences in intensity
# plt.xlabel('Pixel Intensity')
# plt.ylabel('Frequency (Log Scale)')
# plt.title('Histogram of Pixel Intensities')
# plt.show()

#find the maximum value
max_value = np.max(counts)
max_index = np.where(counts == max_value) 
max_index = max_index[0][0]
x_max = bin_centers[max_index]
print(x_max)

#only analyse the part of data close to max value
area = int(num_bins/200)
new_counts = counts[max_index-area:max_index+area]
new_bin_centers = bin_centers[max_index-area:max_index+area]

#we expect the maximum value to correspond to the mean of the background and the left part to correspond to a gaussian
#we will fit a gaussian to the left part of the histogram

a_guess = max_value
std_guess = max_value/60000
m_guess = x_max
p0 = [a_guess, std_guess, m_guess]

popt, pcov = curve_fit(gaussian, new_bin_centers, new_counts, p0=p0, maxfev=10000)

print(popt)

# Plot histogram of pixel intensities
plt.figure(figsize=(10, 6))
plt.plot(new_bin_centers, new_counts, color='black', ls='-',alpha=0.8,label='Data')

plt.plot(new_bin_centers, gaussian(new_bin_centers, *popt), color='red', ls='--',label='Gaussian fit')
plt.plot(new_bin_centers, gaussian(new_bin_centers, *p0), color='blue', ls='--',label='Gaussian guess')
#show where found max value
# plt.axvline(x=x_max, color='red', ls='--',label='Max Value')
#plt.yscale('log')  # Log scale to better visualize differences in intensity
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency (Log Scale)')
plt.title('Histogram of Pixel Intensities')
plt.legend()
plt.show()


#set the threshold to 5 times the standard deviation of the gaussian meaning if the pixel intensity is greater than 5*std then it is a star
thresh = 5*popt[1]+popt[2]
thersh_err = np.sqrt(np.diag(pcov)[1]*25 + np.diag(pcov)[2])

print(f"Threshold: {thresh} +- {thersh_err}")

rel_err = thersh_err/thresh

print(f"Relative error: {rel_err}")
#close file
hdulist.close()