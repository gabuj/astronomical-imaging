import matplotlib.pyplot as plt
import numpy as np

#data is contained in cat file
<<<<<<< HEAD
filename = "/Users/yuri/Desktop/Year 3 Lab/Astronomical Image Processing/Git repository/astronomical-imaging/test_onrealdata/galaxy_catalog.cat"
=======
filename = "C:/SAOImageDS9/my_project_outputs/test_onrealdata/flux/size_full/galaxy_catalog.cat"
>>>>>>> 58ca902c786742e31731509f5be2da3cd3700d41
fluxes, fluxes_err = np.loadtxt(filename, delimiter=' ', skiprows=1, usecols=(2,3), unpack=True)
# Create a mask to filter out rows with invalid values (NaN or inf) in fluxes or fluxes_err
valid_indices = ~np.isnan(fluxes) & ~np.isnan(fluxes_err) & ~np.isinf(fluxes) & ~np.isinf(fluxes_err)
# Apply the mask to both fluxes and fluxes_err
fluxes = fluxes[valid_indices]
fluxes_err = fluxes_err[valid_indices]


#take out nan values from fluxes and fluxes_err and negative values
indexes = np.where(fluxes > 0)
fluxes = fluxes[indexes]
fluxes_err = fluxes_err[indexes]




galaxy_nuber=len(fluxes)
print(f"number of galaxies: {galaxy_nuber}")

#add instrumental zero point found in fits file header as MAGZPT
#open fits file
from astropy.io import fits
fits_file='fits_file/mosaic.fits'
hdul = fits.open(fits_file)

# Get the zero point from the FITS header
zero_point = hdul[0].header['MAGZPT']
zero_point_error = hdul[0].header['MAGZRR']

# Close the FITS file
hdul.close()


faint_range=16
end_range=19
num_bins = int(galaxy_nuber/2)

#convert from pixell to magnitude
magnitude = -2.5 * np.log10(fluxes)
magnitude_err = 2.5/np.log(10)*fluxes_err/fluxes

#convert to real magnitude
magnitudes = magnitude + zero_point
magnitude_errors = np.sqrt(magnitude_err*2 + zero_point_error*2)




#count galaxies under certain magnitude
counts, bin_edges = np.histogram(magnitudes, bins=num_bins)



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
        

#we want to plot logN(<m) vs m so counts are summed
for i in range(len(counts)-1):
    counts[i+1]=counts[i]+counts[i+1]

yerror = np.sqrt(counts) # Poisson error on counts


# Calculate bin centers as the midpoint between each bin edge
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

#plot logN(<m) vs m


# Plot histogram of pixel intensities
plt.figure(figsize=(10, 6))
plt.errorbar(bin_centers, counts, xerr=bin_flux_errors, yerr=yerror, fmt='o', color='black', ls='-',alpha=0.8,label='Data')
plt.yscale('log')  # Log scale to better visualize differences in intensity
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency (Log Scale)')
plt.yscale('log')
plt.title('Histogram of Pixel Intensities')
plt.show()

#fit line to data using np.polyfit with error
# Fit a line to the data
p,cov = np.polyfit(bin_centers, np.log10(counts), 1, w=1/yerror, cov=True)
m, c = p
m_err, c_err = np.sqrt(np.diag(cov))

print(f"m: {m} +/- {m_err}")
print(f"c: {c} +/- {c_err}")


# Plot the data and the fit
plt.figure(figsize=(10, 6))
plt.errorbar(bin_centers, counts, xerr=bin_flux_errors, yerr=yerror, fmt='o', color='black', ls='-',alpha=0.8,label='Data')
plt.plot(bin_centers, 10**(m * bin_centers + c), color='red', label=f'Fit: y = {m:.2f}x + {c:.2f}')
plt.yscale('log')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency (Log Scale)')
plt.title('Galaxy distribution with average fit')
plt.legend()
plt.show()


#fit only first part of the plot now
# Define the range of the data to fit
fit_bright_range = bin_centers < faint_range
#faint range bin centers between 12 and 15
fit_faint_range= bin_centers < end_range
fit_faint_range= fit_faint_range & ~fit_bright_range   

x_bright=bin_centers[fit_bright_range]
y_bright=counts[fit_bright_range]
yerror_bright=yerror[fit_bright_range]

x_faint=bin_centers[fit_faint_range]
y_faint=counts[fit_faint_range]
yerror_faint=yerror[fit_faint_range]

# Fit a line to the data
p_bright,cov_bright = np.polyfit(x_bright, np.log10(y_bright), 1, w=1/yerror_bright, cov=True)
m_bright, c_bright = p_bright
m_err_bright, c_err_bright = np.sqrt(np.diag(cov_bright))

print(f"m_bright: {m_bright} +/- {m_err_bright}")
print(f"c_bright: {c_bright} +/- {c_err_bright}")

# Fit a line to the faint galax paart of the data
p_faint,cov_faint = np.polyfit(x_faint, np.log10(y_faint), 1, w=1/yerror_faint, cov=True)
m_faint, c_faint = p_faint
m_err_faint, c_err_faint = np.sqrt(np.diag(cov_faint))

print(f"m_faint: {m_faint} +/- {m_err_faint}")
print(f"c_faint: {c_faint} +/- {c_err_faint}")

# Plot the data and the fits
plt.figure(figsize=(10, 6))
plt.errorbar(bin_centers, counts, xerr=bin_flux_errors, yerr=yerror, fmt='o', color='black', ls='-',alpha=0.8,label='Data')
plt.plot(x_bright, 10**(m_bright * x_bright + c_bright), color='red', label=f'Bright Fit: y = {m_bright:.2f}x + {c_bright:.2f}')
plt.plot(x_faint, 10**(m_faint * x_faint + c_faint), color='blue', label=f'Faint Fit: y = {m_faint:.2f}x + {c_faint:.2f}')
plt.yscale('log')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency (Log Scale)')
plt.title('Galaxy distribution with faint and bright galaxy fits')
plt.legend()
plt.show()