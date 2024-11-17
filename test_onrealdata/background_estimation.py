import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from astropy.io import fits



def gaussian(x, a, std, m):
    return a *(1/(np.sqrt(2*np.pi)*std))* np.exp(-0.5 * ((x - m) / std) ** 2)

def finding_background(data, fraction_bin, sigmas_thershold):
    num_bins = np.mean(data.shape)/fraction_bin
    #only analyse the part of data close to max value when fitting the gaussian
    if num_bins>20*fraction_bin:
        near_max = int(num_bins/(200))
    else:
        near_max = int(20)
   
    
    
    
    
    num_bins = int(num_bins)
    print(f"num bins are {num_bins}")
    # Calculate histogram bin counts and bin edges
    counts, bin_edges = np.histogram(data.ravel(), bins=num_bins)

    # Calculate bin centers as the midpoint between each bin edge
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Plot histogram of pixel intensities
    plt.figure(figsize=(10, 6))
    plt.plot(bin_centers, counts, color='black', ls='-',alpha=0.8)
    # plt.yscale('log')  # Log scale to better visualize differences in intensity
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency (Log Scale)')
    plt.title('Histogram of Pixel Intensities')
    plt.show()

    #find the maximum value
    max_value = np.max(counts)
    max_index = np.where(counts == max_value) 
    max_index = max_index[0][0]
    x_max = bin_centers[max_index]
    print(f"Max value: {max_value} at {x_max}")
    
    min_near = max_index-near_max if max_index-near_max>0 else 0
    max_near = max_index+near_max if max_index+near_max<num_bins else num_bins

    new_counts = counts[min_near:max_near]
    
    new_bin_centers = bin_centers[min_near:max_near]

    print(f"new counts are {new_counts}")
    #we will fit a gaussian to the histogram

    a_guess = max_value
    half_max = max_value / 2
    # Find the left and right half max points
    indices = np.where(counts > half_max)[0]
    if indices.any():
        left_base = bin_centers[indices[0]]
        right_base = bin_centers[indices[-1]]
        fwhm = right_base - left_base
        std_guess = fwhm / 2.355  # Convert FWHM to std deviation
    else:
        std_guess = max_value / 60000  # fallback if FWHM cannot be determined
    m_guess = x_max
    p0 = [a_guess, std_guess, m_guess]

    near_max = int(fwhm * 1.5 / (bin_edges[1] - bin_edges[0]))

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


    #set the threshold to 5 times the standard deviation of the gaussian meaning if the pixel intensity is greater than 5*std then it is a a
    thresh = sigmas_thershold*popt[1]+popt[2]
    thersh_err =np.sqrt(np.diag(pcov)[1]*sigmas_thershold**2+np.diag(pcov)[2])

    print(f"Threshold: {thresh} +- {thersh_err}")

    rel_err = thersh_err/thresh

    print(f"Relative error: {rel_err}")
    return thresh


def finding_local_background(data, fraction_bin, sigmas_thershold):
    num_bins = np.mean(data.shape)/fraction_bin
    #only analyse the part of data close to max value when fitting the gaussian
    if num_bins>20*fraction_bin:
        near_max = int(num_bins/fraction_bin)
    else:
        near_max = int(20)
   
    
    
    
    
    num_bins = int(num_bins)

    # Calculate histogram bin counts and bin edges
    counts, bin_edges = np.histogram(data.ravel(), bins=num_bins)

    # Calculate bin centers as the midpoint between each bin edge
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Plot histogram of pixel intensities
    # plt.figure(figsize=(10, 6))
    # plt.plot(bin_centers, counts, color='black', ls='-',alpha=0.8)
    # # plt.yscale('log')  # Log scale to better visualize differences in intensity
    # plt.xlabel('Pixel Intensity')
    # plt.ylabel('Frequency (Log Scale)')
    # plt.title('Histogram of Pixel Intensities')
    # plt.show()

    #find the maximum value
    max_value = np.max(counts)
    max_index = np.where(counts == max_value) 
    max_index = max_index[0][0]
    x_max = bin_centers[max_index]
    print(f"local background value: {max_value} at {x_max}")
    #set the threshold to 5 times the standard deviation of the gaussian meaning if the pixel intensity is greater than 5*std then it is a star
    thresh = x_max

    return thresh

path= '/Users/yuri/Desktop/Year 3 Lab/Astronomical Image Processing/Git repository/astronomical-imaging/fits_file/mosaic.fits'
hdulist = fits.open(path)
data = hdulist[0].data
finding_background(data, 0.3, 5)