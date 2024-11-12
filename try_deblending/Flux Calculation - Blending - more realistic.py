from astropy.io import fits
import numpy as np
from scipy.ndimage import gaussian_filter

# Load the FITS file
file_path = "/Users/yuri/Desktop/Year 3 Lab/Astronomical Image Processing/Git repository/astronomical-imaging/fake_files/fake_image_1_realistic.fits"
with fits.open(file_path) as hdul:
    image_data = hdul[0].data.copy()  # Copy of the 2D array of pixel values

# Parameters
background_threshold = 3481  # Consider anything less than 3481 as background
output_image = image_data.copy()  # Image for marking centers and circles
galaxy_count = 0  # Counter for detected galaxies

# Loop to detect galaxies until no significant peak remains
while True:
    # Step 1: Find the highest pixel in the image (galaxy center)
    highest_pixel_value = image_data.max()
    if highest_pixel_value < 3700:  # Stop if no significant peaks remain
        break

    # Find the coordinates of the highest pixel in the image
    highest_pixel_coords = np.unravel_index(np.argmax(image_data), image_data.shape)
    center_y, center_x = highest_pixel_coords
    galaxy_count += 1
    print(f"Galaxy {galaxy_count} detected at Center: ({center_x}, {center_y}), Peak Brightness: {highest_pixel_value}")

    # Mark the center of the galaxy in black on the output image
    output_image[center_y, center_x] = image_data.min()  # Mark galaxy center in black
    print(f"Marked center of Galaxy {galaxy_count} at ({center_x}, {center_y}) in black")

    # Step 2: Calculate radial distances from the highest pixel (assumed center)
    y_indices, x_indices = np.indices(image_data.shape)
    radii = np.sqrt((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2)
    radii = radii.astype(int)

    # Step 3: Calculate the radial intensity profile by averaging pixel values at each radius
    max_possible_radius = min(image_data.shape) // 2  # Maximum radius based on image dimensions
    radial_profile = np.array([image_data[radii == r].mean() if np.any(radii == r) else 0 for r in range(max_possible_radius)])

    # Smooth the radial profile to reduce noise
    smoothed_profile = gaussian_filter(radial_profile, sigma=2)

    # Compute the gradient of the smoothed profile
    gradient = np.diff(smoothed_profile)

    # Check for blending using the specified condition
    blending_detected = False
    boundary_radius = None

    # Step 4: Blending detection condition
    for i in range(len(smoothed_profile) - 1):
        current_radius = i + 1
        radius_values = image_data[radii == current_radius]  # Extract pixel values at this radius

        # Check blending condition: 70% of values below 2 and at least one value above 20
        if np.sum(radius_values < background_threshold) >= 0.7 * len(radius_values) and np.any(radius_values > background_threshold + 1000):
            blending_detected = True
            boundary_radius = i  # Mark the boundary for blending
            print(f"Blending detected for Galaxy {galaxy_count} at radius {boundary_radius}")
            break
        
        # If we've reached a radius of 70 without detecting blending, exit and proceed with non-blending
        elif current_radius >= 70:
            blending_detected = False
            print(f"No blending detected within radius 70 for Galaxy {galaxy_count}. Proceeding with non-blending case.")
            break

    # If blending is not detected, determine the boundary based on pixel extraction for non-blending
    if not blending_detected:
        threshold_radius = None
        for r in range(len(smoothed_profile)):
            current_radius = r + 1
            radius_values = image_data[radii == current_radius]  # Extract pixel values at this radius
            
            # Stop expanding as soon as at least one pixel falls below background threshold
            if np.any(radius_values < background_threshold):
                threshold_radius = current_radius
                print(f"Galaxy {galaxy_count} boundary detected at radius {threshold_radius}")
                break

        # Error handling if no boundary is found (unlikely if there is a background threshold)
        if threshold_radius is None:
            raise ValueError("No valid boundary found - check the image or algorithm settings.")
    else:
        # For blending case: use the boundary_radius detected
        threshold_radius = boundary_radius

    # Draw the circle for the detected boundary
    circle_mask = (radii >= threshold_radius - 1) & (radii <= threshold_radius + 1)
    output_image[circle_mask] = output_image.max()  # Boundary in white

    # Mask out the detected galaxy region in image_data to prevent re-detection
    mask_radius = threshold_radius + 1  # Slightly larger than the detected radius
    image_data[radii <= mask_radius] = background_threshold  # Mask with background level to avoid re-detection

    print(f"Galaxy {galaxy_count} - Radius: {threshold_radius}, Center: ({center_x}, {center_y})")

# Print the total number of galaxies detected
print(f"Total number of galaxies detected: {galaxy_count}")

# Step 9: Save the modified data to a new FITS file with circles and centers marked
output_path = "/Users/yuri/Desktop/Year 3 Lab/Astronomical Image Processing/Git repository/astronomical-imaging/fake_files/fakeimage_results_realistic.fits"
hdu = fits.PrimaryHDU(output_image)
hdul_with_circles = fits.HDUList([hdu])
hdul_with_circles.writeto(output_path, overwrite=True)

print(f"Final output saved to {output_path}")