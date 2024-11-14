from astropy.io import fits
import numpy as np

# Parameters for the image and galaxies
image_size = (512, 512)  # Size of the image (512x512 pixels)

# Parameters for each galaxy
# Centers are chosen to be at least 100 pixels apart
galaxies = [
    {"center": (80, 80), "peak": 100, "sigma": 15},    # Galaxy 1
    {"center": (400, 80), "peak": 90, "sigma": 20},    # Galaxy 2
    {"center": (80, 400), "peak": 80, "sigma": 25},    # Galaxy 3
    {"center": (400, 400), "peak": 70, "sigma": 18},   # Galaxy 4
    {"center": (250, 250), "peak": 95, "sigma": 22},   # Galaxy 5
    {"center": (150, 150), "peak": 85, "sigma": 17},   # Galaxy 6
    {"center": (350, 150), "peak": 78, "sigma": 19},   # Galaxy 7
    {"center": (150, 350), "peak": 88, "sigma": 21},   # Galaxy 8
    {"center": (350, 350), "peak": 77, "sigma": 16},   # Galaxy 9
    {"center": (200, 100), "peak": 92, "sigma": 20},   # Galaxy 10
    {"center": (300, 400), "peak": 82, "sigma": 23}    # Galaxy 11
]

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
for galaxy in galaxies:
    add_galaxy(image_data, galaxy["center"], galaxy["peak"], galaxy["sigma"])

# Optional: Add slight random noise to the background for realism
background_noise_level = 0.5
noise = np.random.normal(0, background_noise_level, image_size)
image_data += noise

# Ensure no negative values after adding noise
image_data = np.clip(image_data, 0, None)

# Save the resulting image as a FITS file
output_path = "/Users/yuri/Desktop/Year 3 Lab/Astronomical Image Processing/Git repository/astronomical-imaging/fake_files/fakeimage_11.fits"
hdu = fits.PrimaryHDU(image_data)
hdul = fits.HDUList([hdu])
hdul.writeto(output_path, overwrite=True)

print(f"File saved to {output_path}")