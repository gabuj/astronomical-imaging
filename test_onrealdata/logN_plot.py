
import matplotlib.pyplot as plt
import numpy as np

#data is contained in cat file
filename='test_onrealdata/galaxy_catalog.cat'
fluxes,fluxes_err = np.loadtxt(filename, delimiter=' ', skiprows=1,usecols=(2,3), unpack=True)
galaxy_nuber=len(fluxes)
print(f"number of galaxies: {galaxy_nuber}")
num_bins = int(galaxy_nuber/10)
# Calculate histogram bin counts and bin edges
#convert from pixell to magnitude
magnitude = -2.5 * np.log10(fluxes)
magnitude_err = 2.5/np.log(10)*fluxes_err/fluxes

counts, bin_edges = np.histogram(fluxes, bins=num_bins)


#substitute all nan values in flux err with sqrt(flux)
for i in range(len(fluxes_err)):
    if np.isnan(fluxes_err[i]):
        fluxes_err[i]=np.sqrt(fluxes[i])

#calculate errors of fluxes in bin
bin_flux_errors=[]
for i in range(num_bins):
    # Define bin range
    bin_min, bin_max = bin_edges[i], bin_edges[i + 1]
    
    # Get flux errors for galaxies in the current bin
    in_bin = (fluxes >= bin_min) & (fluxes < bin_max)
    errors_in_bin = fluxes_err[in_bin]
    
    if len(errors_in_bin) > 0:
        #Mean flux error in the bin
        mean_error = errors_in_bin.mean()
    else:
        # Handle empty bins if they occur
        bin_flux_errors.append(0)
        
yerror = np.sqrt(counts)

#how do i estimate magnitude error?
# Calculate bin centers as the midpoint between each bin edge
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Plot histogram of pixel intensities
plt.figure(figsize=(10, 6))
plt.errorbar(bin_centers, counts, xerr=bin_flux_errors, yerr=yerror, fmt='o', color='black', ls='-',alpha=0.8)
plt.yscale('log')  # Log scale to better visualize differences in intensity
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency (Log Scale)')
plt.yscale('log')
plt.title('Histogram of Pixel Intensities')
plt.show()