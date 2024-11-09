from astropy.io import fits
import numpy as np
from scipy.ndimage import gaussian_filter, label, center_of_mass
from scipy.signal import find_peaks

# Load the FITS file
file_path = "/Users/yuri/Desktop/Year 3 Lab/Astronomical Image Processing/Astro/Fits_Data/fakeimage - 4.fits"
with fits.open(file_path) as hdul:
    image_data = hdul[0].data.copy()  # Get a copy of the 2D array of pixel values

# Step 1: Detect multiple galaxy centers using thresholding and segmentation
threshold_value = 30.0  # Set this (relatively well) above the background level
binary_image = image_data > threshold_value

# Label connected regions (each region is a potential galaxy)
labeled_image, num_features = label(binary_image)
print(f"Number of detected regions (potential galaxies): {num_features}")

# Calculate centroids of each labeled region (galaxy)
centroids = center_of_mass(image_data, labeled_image, range(1, num_features + 1))

# Set the minimum pixel value to mark the centroid in black
min_value = image_data.min()

# Parameters for galaxy boundary detection
background_value_range = (0, 0.7)
max_radius = 100  # Adjusted max radius to 100 pixels for a more realistic boundary

# Create a copy of the image data for marking circles and centroids without modifying the original
output_image = image_data.copy()

# Process each detected galaxy
for i, (center_y, center_x) in enumerate(centroids):
    print(f"Processing Galaxy {i + 1} at Centroid: ({center_x:.2f}, {center_y:.2f})")

    # Step 2: Calculate the radial distances from the centroid
    y_indices, x_indices = np.indices(image_data.shape)
    radii = np.sqrt((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2)
    radii = radii.astype(int)  # Convert distances to integer for binning

    # Step 3: Calculate the radial intensity profile by averaging pixel values at each radius
    radial_profile = np.array([image_data[radii == r].mean() for r in range(max_radius)])

    # Step 4: Smooth the profile to reduce noise
    smoothed_profile = gaussian_filter(radial_profile, sigma=1)

    # Step 5: Compute the gradient of the smoothed profile
    gradient = np.diff(smoothed_profile)

    # Step 6: Find the maximum gradient drop that transitions to the background level (between 0 and 0.7)
    valid_drops = []
    for j in range(len(gradient) - 1):  # Go up to max_radius - 1 for safe indexing
        # Refine the conditions for detecting a valid drop
        if smoothed_profile[j] > background_value_range[1] and background_value_range[0] <= smoothed_profile[j + 1] <= background_value_range[1]:
            valid_drops.append((j, gradient[j]))  # Append (index, gradient) tuples

    # Refine drop selection: pick the closest, largest drop meeting our refined criteria
    if valid_drops:
        # Sort first by gradient magnitude (descending), then by radius (ascending) to find the closest highest drop
        valid_drops = sorted(valid_drops, key=lambda x: (-x[1], x[0]))
        threshold_radius = valid_drops[0][0]  # Get the radius corresponding to the largest valid drop
    else:
        threshold_radius = max_radius - 1  # Fallback to max_radius - 1 if no valid drop is found

    # Step 7: Draw a circle around each detected galaxy at the calculated radius
    circle_mask = (radii >= threshold_radius - 1) & (radii <= threshold_radius + 1)
    output_image[circle_mask] = output_image.max()  # Set the circle to the max pixel value for visibility

    # Mark the centroid in black in the output image
    output_image[int(center_y), int(center_x)] = min_value  # Set the centroid to the minimum pixel value for black color

    print(f"Galaxy {i + 1} - Radius: {threshold_radius}, Centroid: ({center_x}, {center_y})")

# Step 8: Save the modified data to a new FITS file with circles and centroids marked
output_path = "/Users/yuri/Desktop/Year 3 Lab/Astronomical Image Processing/Astro/Fits_Data/fakeimage_center.fits"
hdu = fits.PrimaryHDU(output_image)
hdul_with_circles = fits.HDUList([hdu])
hdul_with_circles.writeto(output_path, overwrite=True)

print(f"Output saved to {output_path}")