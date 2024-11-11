from astropy.io import fits
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
import creating_fake_image
import matplotlib.pyplot as plt
#parameters to create fake image:
image_size = (512, 512)  # Size of the image (512x512 pixels)
centers = [(200, 200)]
galaxy_peaks = [200]
sigmas = [20]
noise_level = 10 
image_data=creating_fake_image.create_fake_image(image_size, centers, galaxy_peaks, sigmas, noise_level)
radius=sigmas[0]*1.6
#show the image
plt.imshow(image_data, cmap='gray')
plt.colorbar()
plt.show()

# Step 1: Find the center of the galaxy
center_y, center_x = np.unravel_index(np.argmax(image_data), image_data.shape)

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

#for now use threshold radius as sigma
threshold_radius=sigmas[0]*1.6
#convert to integer for binning
threshold_radius = int(threshold_radius)
# Step 8: Convert radius to a threshold intensity value
isophotal_threshold = smoothed_profile[threshold_radius]

# Step 9: Create a mask for pixels within the threshold radius to calculate total flux
isophotal_mask = radii <= threshold_radius


flux = np.sum(image_data[isophotal_mask])  # Calculate the flux

print("Isophotal flux of the galaxy:", flux)
print("Threshold intensity used:", isophotal_threshold)
print("Threshold radius:", threshold_radius)



# Step 10: Draw a circle at the detected galaxy boundary on a copy of the image data
thickness = 1  # Thickness of the circle boundary
circle_mask = (radii >= threshold_radius - thickness) & (radii <= threshold_radius + thickness)
image_data[circle_mask] = 1  # Set the boundary pixels to 1

# Display the image with the detected galaxy boundary
plt.imshow(image_data, cmap='gray')
plt.colorbar()
plt.title('Detected Galaxy Boundary')
plt.show()