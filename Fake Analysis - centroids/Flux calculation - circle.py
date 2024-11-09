from astropy.io import fits
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

# Load the FITS file
file_path = "/Users/yuri/Desktop/Year 3 Lab/Astronomical Image Processing/Astro/Fits_Data/fakeimage.fits"
with fits.open(file_path) as hdul:
    image_data = hdul[0].data.copy()  # Get a copy of the 2D array of pixel values

# Step 1: Define the galaxy center (assuming it's approximately at the center of the image)
center_x, center_y = image_data.shape[0] // 2, image_data.shape[1] // 2

# Step 2: Compute the radial distances from the center
y, x = np.indices(image_data.shape)
radii = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
radii = radii.astype(int)  # Convert distances to integer for binning

# Step 3: Calculate the radial intensity profile by averaging pixel values at each radius
max_radius = 1000  # Strictly limit the radius to 1000 pixels
radial_profile = np.array([image_data[radii == r].mean() for r in range(max_radius)])

# Step 4: Smooth the profile to reduce noise
smoothed_profile = gaussian_filter(radial_profile, sigma=1)

# Step 5: Compute the gradient of the smoothed profile
gradient = np.diff(smoothed_profile)

# Step 6: Filter gradients that drop to values between 0 and 0.7
background_value_range = (0, 0.7)
valid_drops = []
for i in range(len(gradient) - 1):  # Go up to max_radius - 1 for safe indexing
    if smoothed_profile[i] > background_value_range[1] and background_value_range[0] <= smoothed_profile[i + 1] <= background_value_range[1]:
        valid_drops.append((i, gradient[i]))  # Append (index, gradient) tuples

# Step 7: Select the highest gradient among the valid drops that is also closest to the center
if valid_drops:
    # Sort first by gradient magnitude (descending), then by radius (ascending) to find the closest highest drop
    valid_drops = sorted(valid_drops, key=lambda x: (-x[1], x[0]))
    threshold_radius = valid_drops[0][0]  # Get the radius corresponding to the largest valid drop
else:
    threshold_radius = max_radius - 1  # Fallback to max_radius - 1 if no valid drop is found

# Step 8: Convert radius to a threshold intensity value
isophotal_threshold = smoothed_profile[threshold_radius]

# Step 9: Create a mask for pixels within the threshold radius to calculate total flux
isophotal_mask = radii <= threshold_radius
flux = np.sum(image_data[isophotal_mask])  # Calculate the flux

print("Isophotal flux of the galaxy:", flux)
print("Threshold intensity used:", isophotal_threshold)
print("Threshold radius:", threshold_radius)

# Step 10: Draw a circle at the detected galaxy boundary on a copy of the image data
circle_mask = (radii >= threshold_radius - 1) & (radii <= threshold_radius + 1)
image_data[circle_mask] = image_data.max()  # Set the circle to the max pixel value for visibility

# Step 11: Save the modified data to a new FITS file with the circle overlay
output_path = "/Users/yuri/Desktop/Year 3 Lab/Astronomical Image Processing/Astro/Fits_Data/fakeimagecircle.fits"
hdu = fits.PrimaryHDU(image_data)
hdul_with_circle = fits.HDUList([hdu])
hdul_with_circle.writeto(output_path, overwrite=True)