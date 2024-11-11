from astropy.io import fits
import numpy as np

# Parameters for the image and galaxies
image_size = (512, 512)  # Size of the image (512x512 pixels)

# Parameters for Galaxy 1
center1 = (80, 80)       # Position of the first galaxy
galaxy_peak1 = 100       # Peak brightness at the center of the first galaxy
sigma1 = 15              # Spread of the first galaxy (standard deviation)

# Parameters for Galaxy 2
center2 = (430, 100)     # Position of the second galaxy
galaxy_peak2 = 90        # Peak brightness at the center of the second galaxy
sigma2 = 20              # Spread of the second galaxy (standard deviation)

# Parameters for Galaxy 3
center3 = (200, 400)     # Position of the third galaxy
galaxy_peak3 = 80        # Peak brightness at the center of the third galaxy
sigma3 = 25              # Spread of the third galaxy (standard deviation)

# Parameters for Galaxy 4
center4 = (350, 300)     # Position of the fourth galaxy
galaxy_peak4 = 70        # Peak brightness at the center of the fourth galaxy
sigma4 = 18              # Spread of the fourth galaxy (standard deviation)

# Parameters for Galaxy 5
center5 = (150, 350)     # Position of the fifth galaxy
galaxy_peak5 = 95        # Peak brightness at the center of the fifth galaxy
sigma5 = 22              # Spread of the fifth galaxy (standard deviation)

# Create a zero-valued background image
image_data = np.zeros(image_size)

# Function to add a Gaussian galaxy profile to the image
def add_galaxy(image, center, galaxy_peak, sigma):
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            # Calculate distance from the center of the galaxy
            distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
            # Apply a Gaussian profile
            image[x, y] += galaxy_peak * np.exp(-(distance ** 2) / (2 * sigma ** 2))

# Add each galaxy to the image
add_galaxy(image_data, center1, galaxy_peak1, sigma1)
add_galaxy(image_data, center2, galaxy_peak2, sigma2)
add_galaxy(image_data, center3, galaxy_peak3, sigma3)
add_galaxy(image_data, center4, galaxy_peak4, sigma4)
add_galaxy(image_data, center5, galaxy_peak5, sigma5)

# Optional: Add slight random noise to the background for realism
background_noise_level = 0.5
noise = np.random.normal(0, background_noise_level, image_size)
image_data += noise

# Ensure no negative values after adding noise
image_data = np.clip(image_data, 0, None)

# Save the resulting image as a FITS file
output_path = "/Users/yuri/Desktop/Year 3 Lab/Astronomical Image Processing/Astro/Fits_Data/fakeimage - 5 - blending.fits"
hdu = fits.PrimaryHDU(image_data)
hdul = fits.HDUList([hdu])
hdul.writeto(output_path, overwrite=True)

output_path