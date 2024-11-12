
import matplotlib.pyplot as plt
import numpy as np

#data is contained in cat file
filename='xy_intensity/galaxy_catalog.cat'
fluxes,fluxes_err = np.loadtxt(filename, delimiter=' ', skiprows=1,usecols=(2,3), unpack=True)
print(f"number of galaxies: {len(fluxes)}")
num_bins = 8
# Calculate histogram bin counts and bin edges
#convert from pixell to magnitude
magnitude = -2.5 * np.log10(fluxes)
magnitude_err = 2.5/np.log(10)*fluxes_err/fluxes

counts, bin_edges = np.histogram(fluxes, bins=num_bins)

#how do i estimate magnitude error?
# Calculate bin centers as the midpoint between each bin edge
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Plot histogram of pixel intensities
plt.figure(figsize=(10, 6))
plt.plot(bin_centers, counts, color='black', ls='-',alpha=0.8)
plt.yscale('log')  # Log scale to better visualize differences in intensity
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency (Log Scale)')
plt.yscale('log')
plt.title('Histogram of Pixel Intensities')
plt.show()
