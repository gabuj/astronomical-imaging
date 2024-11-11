from astropy.io import fits
import numpy as np
from scipy.ndimage import gaussian_filter

# Load the FITS file
file_path = "/Users/yuri/Desktop/Year 3 Lab/Astronomical Image Processing/Astro/Fits_Data/fakeimage - 6 - blending.fits"
with fits.open(file_path) as hdul:
    image_data = hdul[0].data.copy()  # Copy of the 2D array of pixel values

# Parameters
background_threshold = 1  # Consider anything less than 1 as background
max_radius = 100  # Limit search radius for a more realistic boundary
output_image = image_data.copy()  # Image for marking centers and circles
galaxy_count = 0  # Counter for detected galaxies

# Loop to detect galaxies until no significant peak remains
while True:
    # Step 1: Find the highest pixel in the image (galaxy center)
    highest_pixel_value = image_data.max()
    if highest_pixel_value < 10:  # Stop if no significant peaks remain
        break

    # Find the coordinates of the highest pixel in the image
    highest_pixel_coords = np.unravel_index(np.argmax(image_data), image_data.shape)
    center_y, center_x = highest_pixel_coords
    galaxy_count += 1
    print(f"Galaxy {galaxy_count} detected at Center: ({center_x}, {center_y}), Peak Brightness: {highest_pixel_value}")

    # Mark the center of the first galaxy in black on the output image
    output_image[center_y, center_x] = image_data.min()  # First galaxy center in black
    print(f"Marked center of Galaxy {galaxy_count} at ({center_x}, {center_y}) in black")

    # Step 2: Calculate radial distances from the highest pixel (assumed center)
    y_indices, x_indices = np.indices(image_data.shape)
    radii = np.sqrt((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2)
    radii = radii.astype(int)

    # Step 3: Calculate the radial intensity profile by averaging pixel values at each radius
    radial_profile = np.array([image_data[radii == r].mean() for r in range(max_radius)])

    # Smooth the radial profile to reduce noise
    smoothed_profile = gaussian_filter(radial_profile, sigma=2)

    # Compute the gradient of the smoothed profile
    gradient = np.diff(smoothed_profile)

    # Check for blending using the specified condition
    blending_detected = False
    boundary_radius = None

    for i in range(len(gradient) - 1):
        current_radius = i + 1
        radius_values = image_data[radii == current_radius]  # Extract pixel values at this radius

        if np.sum(radius_values < 2) >= 0.7 * len(radius_values) and np.any(radius_values > 20):
            blending_detected = True
            boundary_radius = i  # Mark the boundary for blending
            print(f"Blending detected for Galaxy {galaxy_count} at radius {boundary_radius}")
            break

    # After the loop, if blending was not detected, set blending_detected to False
    if not blending_detected:
        print(f"No blending detected for Galaxy {galaxy_count} after checking up to radius {max_radius}.")
        blending_detected = False

    # Step 4: Determine the boundary for a single galaxy if no blending is detected
    if not blending_detected:
        valid_drops = []
        for j in range(1, len(gradient) - 1):
            p1 = smoothed_profile[j]
            p2 = smoothed_profile[j + 1]
            if 0.8 <= p2 < 1:
                gradient_value = gradient[j]
                valid_drops.append((j, gradient_value))

        if valid_drops:
            valid_drops = sorted(valid_drops, key=lambda x: x[0])
            threshold_radius = valid_drops[0][0]
        else:
            raise ValueError("No valid drop points found - check the image or algorithm settings.")
    else:
        # For blending case: use the boundary_radius detected
        threshold_radius = boundary_radius

    # Draw the circle for the detected boundary
    circle_mask = (radii >= threshold_radius - 1) & (radii <= threshold_radius + 1)
    output_image[circle_mask] = output_image.max()  # First boundary in white

    # Mask out the detected galaxy region in image_data to prevent re-detection
    mask_radius = threshold_radius + 1  # Slightly larger than the detected radius
    image_data[radii <= mask_radius] = background_threshold  # Mask with background level to avoid re-detection

    print(f"Galaxy {galaxy_count} - Radius: {threshold_radius}, Center: ({center_x}, {center_y})")

# Step 9: Save the modified data to a new FITS file with circles and centers marked
output_path = "/Users/yuri/Desktop/Year 3 Lab/Astronomical Image Processing/Astro/Fits_Data/fakeimage_results.fits"
hdu = fits.PrimaryHDU(output_image)
hdul_with_circles = fits.HDUList([hdu])
hdul_with_circles.writeto(output_path, overwrite=True)

print(f"Final output saved to {output_path}")