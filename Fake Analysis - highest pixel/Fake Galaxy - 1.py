from astropy.io import fits
import numpy as np

# Parameters for the fake galaxy image
image_size = (512, 512)  # Size of the image (512x512 pixels)
center_x, center_y = image_size[0] // 2, image_size[1] // 2  # Galaxy at the center
galaxy_peak = 100  # Peak brightness of the galaxy center
sigma = 20  # Standard deviation of the Gaussian (controls the spread of the galaxy)

# Create a zero-valued background
image_data = np.zeros(image_size)

# Generate a Gaussian profile for the galaxy
for x in range(image_size[0]):
    for y in range(image_size[1]):
        # Calculate distance from the center
        distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        # Apply a Gaussian profile
        image_data[x, y] = galaxy_peak * np.exp(-(distance ** 2) / (2 * sigma ** 2))

# Optional: Add slight random noise to the background
background_noise_level = 0.5
noise = np.random.normal(0, background_noise_level, image_size)
image_data += noise

# Ensure no negative values after adding noise
image_data = np.clip(image_data, 0, None)

# Save to a FITS file
#Image created, saved in a file in the same folder as Mosaic
hdu = fits.PrimaryHDU(image_data)
hdul = fits.HDUList([hdu])
output_path = "/Users/yuri/Desktop/Year 3 Lab/Astronomical Image Processing/Git repository/astronomical-imaging/fake_files/fakeimage_1.fits"
hdul.writeto(output_path, overwrite=True)

