import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os

def gaussian(r, mu, sigma):
    """Gaussian profile function."""
    return np.exp(-np.power(r - mu, 2.) / (2 * np.power(sigma, 2.)))

def sersic(r, I_e, r_e, n):
    """Sersic profile function."""
    b_n = 1.9992 * n - 0.3271
    return I_e * np.exp(-b_n * ((r / r_e)**(1/n) - 1))

def add_galaxy(image_data, center_x, center_y, galaxy_peak, sigma, n):
    """Add a galaxy with a specified profile to the image."""
    image_size = image_data.shape
    for y in range(image_size[0]):  # Loop over rows (y-coordinate)
        for x in range(image_size[1]):  # Loop over columns (x-coordinate)
            distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            if n == 0:
                image_data[y, x] += galaxy_peak * gaussian(distance, 0, sigma)
            else:
                image_data[y, x] += sersic(distance, galaxy_peak, sigma, n)
    return image_data

def add_background_noise(image_data, noise_level, image_size, background_level):
    """Add Gaussian background noise to the image."""
    noise = np.random.normal(background_level, noise_level, image_size)
    image_data += noise
    image_data = np.clip(image_data, 0, None)  # Ensure no negative values
    return image_data

def create_fake_image(image_size, centers, galaxy_peaks, sigmas, background_level, noise_level, ns):
    """Create an image with multiple galaxies and background noise."""
    image_data = np.zeros(image_size)
    image_data = add_background_noise(image_data, noise_level, image_size, background_level)
    for center, galaxy_peak, sigma, n in zip(centers, galaxy_peaks, sigmas, ns):
        image_data = add_galaxy(image_data, center[0], center[1], galaxy_peak, sigma, n)
    return image_data

# Parameters for creating images
image_size = (1028, 1028)
centers = [(500, 500), (560, 500), (500, 560), (560, 560)]
peaks = [180, 150, 150, 120]
sigmas = [25, 20, 20, 15]
ns = [0.5, 0.5, 0.5, 0.5]

noise_level = 5
background_level = 3415

# Create the synthetic image
image_data = create_fake_image(image_size, centers, peaks, sigmas, background_level, noise_level, ns)

# Save the generated image as a FITS file
name = "4_cluster"
output_path = f"fake_files/{name}.fits"
hdu = fits.PrimaryHDU(image_data)
hdul = fits.HDUList([hdu])
hdul.writeto(output_path, overwrite=True)
print(f"File saved to {output_path}")

# Automatically open the FITS file after creation
os.system(f"open {output_path}")

# Uncomment below if you'd like to view the last generated image
# plt.imshow(image_data, cmap='gray')
# plt.colorbar()
# plt.show()