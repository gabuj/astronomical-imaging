import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os

def gaussian(x, mu, sigma):
    """Gaussian profile function."""
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))

def sersic(x, I_e, r_e, n):
    """Sersic profile function."""
    b_n = 1.9992 * n - 0.3271
    return I_e * np.exp(-b_n * ((x / r_e)**(1/n) - 1))

def add_galaxy(image_data, center_x, center_y, galaxy_peak, sigma, n):
    """Add a galaxy with a specified profile to the image."""
    image_size = image_data.shape
    for x in range(image_size[0]):
        for y in range(image_size[1]):
            distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            if n == 0:
                image_data[x, y] += galaxy_peak * gaussian(distance, 0, sigma)
            else:
                image_data[x, y] += sersic(distance, galaxy_peak, sigma, n)
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
centers = [(500, 500), (508, 500)]  # Close centers to cause overlap
peaks = [200, 200]
sigmas = [4, 4]
ns = [1.0, 1.0]


noise_level = 5
background_level = 3415


# Create the synthetic image
image_data = create_fake_image(image_size, centers, peaks, sigmas, background_level, noise_level, ns)

# Save each generated image as a FITS file
name = "fakeimage_2_blending_realistic"
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