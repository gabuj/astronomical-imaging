from astropy.io import fits
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
import os

# Load the FITS file
name = "fakeimage_2_blending_realistic"
file_path = f"/Users/yuri/Desktop/Year 3 Lab/Astronomical Image Processing/Git repository/astronomical-imaging/fake_files/{name}.fits"
with fits.open(file_path) as hdul:
    image_data = hdul[0].data.copy()  # Copy of the 2D array of pixel values

# Parameters
background_level = 3415
noise_level = 5
std_multiplier = 5
background_threshold = background_level + std_multiplier * noise_level  # Consider anything less as background
std_highest_pixel = 6
std_threshold = background_level + std_highest_pixel * noise_level

output_image = image_data.copy()  # Image for marking centers and circles
galaxy_count = 0  # Counter for detected galaxies

# Keep track of processed pixels
processed_pixels = np.zeros_like(image_data, dtype=bool)

# Loop to detect galaxies until no significant peak remains
while True:
    # Step 1: Find the highest pixel in the image (galaxy center)
    # Ignore already processed pixels
    masked_image = np.where(processed_pixels, background_level, image_data)
    highest_pixel_value = masked_image.max()
    if highest_pixel_value < std_threshold:  # Stop if no significant peaks remain
        break

    # Find the coordinates of the highest pixel in the masked image
    highest_pixel_coords = np.unravel_index(np.argmax(masked_image), masked_image.shape)
    center_y1, center_x1 = highest_pixel_coords
    galaxy_count += 1
    print(f"\nGalaxy {galaxy_count} detected at Center: ({center_x1}, {center_y1}), Peak Brightness: {highest_pixel_value}")

    # Do not mark the center or draw circles yet to avoid interfering with detection of other galaxies

    # Step 2: Calculate radial distances from the first galaxy's center
    y_indices, x_indices = np.indices(image_data.shape)
    radii1 = np.sqrt((x_indices - center_x1) ** 2 + (y_indices - center_y1) ** 2)
    radii1 = radii1.astype(int)

    # Step 3: Calculate the radial intensity profile for the first galaxy
    max_possible_radius = min(image_data.shape) // 2  # Maximum radius based on image dimensions
    radial_profile1 = np.array([
        image_data[(radii1 == r) & ~processed_pixels].mean() if np.any((radii1 == r) & ~processed_pixels) else 0
        for r in range(max_possible_radius)
    ])

    # Smooth the radial profile to reduce noise
    smoothed_profile1 = gaussian_filter(radial_profile1, sigma=2)

    # Blending detection variables
    blending_detected = False
    boundary_radius1 = None

    # Step 4: Blending detection for the first galaxy
    for i in range(len(smoothed_profile1) - 1):
        current_radius = i + 1
        radius_values = image_data[(radii1 == current_radius) & ~processed_pixels]  # Exclude already processed pixels

        # Check blending condition
        if np.sum(radius_values < background_threshold) >= 0.7 * len(radius_values) and np.any(
            radius_values > background_threshold + 10):
            blending_detected = True
            boundary_radius1 = i  # Mark the boundary for blending
            print(f"Blending detected for Galaxy {galaxy_count} at radius {boundary_radius1}")
            break

        elif current_radius >= 70:
            blending_detected = False
            print(f"No blending detected within radius 70 for Galaxy {galaxy_count}. Proceeding with non-blending case.")
            break

    # Step 5: Determine the boundary for the first galaxy
    if blending_detected:
        # Use the boundary_radius1 detected during blending detection
        threshold_radius1 = boundary_radius1
    else:
        # For non-blending case, determine the boundary based on the first galaxy's profile
        threshold_radius1 = None
        for r in range(len(smoothed_profile1)):
            if smoothed_profile1[r] < background_threshold:
                threshold_radius1 = r
                print(f"Galaxy {galaxy_count} boundary detected at radius {threshold_radius1}")
                break
        if threshold_radius1 is None:
            threshold_radius1 = max_possible_radius

    # Do not draw the circle or mark pixels yet

    # Step 6: If blending was detected, find the second galaxy before modifying the image
    if blending_detected:
        # Define a local region around the first galaxy to search for additional peaks
        region_size = threshold_radius1 + 20  # Extend beyond the first galaxy's boundary
        y_min = max(center_y1 - region_size, 0)
        y_max = min(center_y1 + region_size, image_data.shape[0])
        x_min = max(center_x1 - region_size, 0)
        x_max = min(center_x1 + region_size, image_data.shape[1])

        local_region = masked_image[y_min:y_max, x_min:x_max]
        local_processed = processed_pixels[y_min:y_max, x_min:x_max]

        # Apply maximum filter to find local maxima
        from scipy.ndimage import maximum_filter, label
        neighborhood_size = 3  # Adjust as needed
        local_max = maximum_filter(local_region, size=neighborhood_size) == local_region

        # Exclude background and already processed pixels
        background = (local_region < std_threshold) | local_processed
        peaks = local_max & ~background

        # Get coordinates of the peaks
        peak_coords = np.argwhere(peaks)
        # Adjust coordinates to the global image
        peak_coords[:, 0] += y_min
        peak_coords[:, 1] += x_min

        # Remove the first galaxy's center from the list
        peak_coords = peak_coords[(peak_coords[:, 0] != center_y1) | (peak_coords[:, 1] != center_x1)]

        if peak_coords.size > 0:
            # Process each detected peak as a potential galaxy
            for coord in peak_coords:
                center_y2, center_x2 = coord
                # Check if this peak has already been processed
                if processed_pixels[center_y2, center_x2]:
                    continue
                second_peak_value = image_data[center_y2, center_x2]
                if second_peak_value > std_threshold:
                    galaxy_count += 1
                    print(f"Second Galaxy {galaxy_count} detected at Center: ({center_x2}, {center_y2}), Peak Brightness: {second_peak_value}")

                    # Do not mark the center or draw circles yet

                    # Step 7: Calculate radial distances from the second galaxy's center
                    radii2 = np.sqrt((x_indices - center_x2) ** 2 + (y_indices - center_y2) ** 2)
                    radii2 = radii2.astype(int)

                    # Calculate the radial intensity profile for the second galaxy
                    radial_profile2 = np.array([
                        image_data[(radii2 == r) & ~processed_pixels].mean() if np.any((radii2 == r) & ~processed_pixels) else 0
                        for r in range(max_possible_radius)
                    ])

                    # Smooth the radial profile
                    smoothed_profile2 = gaussian_filter(radial_profile2, sigma=2)

                    # Determine the boundary for the second galaxy
                    threshold_radius2 = None
                    for i in range(len(smoothed_profile2) - 1):
                        current_radius = i + 1
                        radius_values = image_data[(radii2 == current_radius) & ~processed_pixels]  # Exclude already processed pixels

                        # Check blending condition
                        if np.sum(radius_values < background_threshold) >= 0.7 * len(radius_values) and np.any(
                            radius_values > background_threshold + 10):
                            threshold_radius2 = i  # Mark the boundary for blending
                            print(f"Second Galaxy {galaxy_count} boundary detected at radius {threshold_radius2}")
                            break

                    # for r in range(len(smoothed_profile2)):
                    #     if smoothed_profile2[r] < background_threshold:
                    #         threshold_radius2 = r
                    #         print(f"Second Galaxy {galaxy_count} boundary detected at radius {threshold_radius2}")
                    #         break
                    if threshold_radius2 is None:
                        raise ValueError("No Threshold Radius 2 found")

                    # After determining both galaxies, we can now draw circles and mark pixels

                    # Draw the circle for the first galaxy
                    circle_mask1 = (radii1 >= threshold_radius1 - 1) & (radii1 <= threshold_radius1 + 1)
                    output_image[circle_mask1] = output_image.max()  # Boundary in white
                    # Mark the center of the first galaxy in black
                    output_image[center_y1, center_x1] = image_data.min()

                    # Draw the circle for the second galaxy
                    circle_mask2 = (radii2 >= threshold_radius2 - 1) & (radii2 <= threshold_radius2 + 1)
                    output_image[circle_mask2] = output_image.max()  # Boundary in white
                    # Mark the center of the second galaxy in black
                    output_image[center_y2, center_x2] = image_data.min()

                    # Combine the masks for both galaxies
                    total_mask = (radii1 <= threshold_radius1) | (radii2 <= threshold_radius2)
                    # Mark pixels as processed for both galaxies
                    processed_pixels[total_mask] = True

                    print(f"Galaxy {galaxy_count - 1} - Radius: {threshold_radius1}, Center: ({center_x1}, {center_y1})")
                    print(f"Galaxy {galaxy_count} - Radius: {threshold_radius2}, Center: ({center_x2}, {center_y2})")
                    break  # Only process the first additional peak for simplicity
        else:
            print("No additional peaks found in the blending region.")
            # Draw the circle and mark pixels for the first galaxy only
            # Draw the circle for the first galaxy
            circle_mask1 = (radii1 >= threshold_radius1 - 1) & (radii1 <= threshold_radius1 + 1)
            output_image[circle_mask1] = output_image.max()  # Boundary in white
            # Mark the center of the first galaxy in black
            output_image[center_y1, center_x1] = image_data.min()
            # Mark pixels as processed for the first galaxy
            processed_pixels[radii1 <= threshold_radius1] = True
    else:
        # Non-blending case
        # Draw the circle and mark pixels for the first galaxy
        # Draw the circle for the first galaxy
        circle_mask1 = (radii1 >= threshold_radius1 - 1) & (radii1 <= threshold_radius1 + 1)
        output_image[circle_mask1] = output_image.max()  # Boundary in white
        # Mark the center of the first galaxy in black
        output_image[center_y1, center_x1] = image_data.min()
        # Mark pixels as processed for the first galaxy
        processed_pixels[radii1 <= threshold_radius1] = True

        print(f"Galaxy {galaxy_count} - Radius: {threshold_radius1}, Center: ({center_x1}, {center_y1})")

# Print the total number of galaxies detected
print(f"\nTotal number of galaxies detected: {galaxy_count}")

# Save the modified data to a new FITS file with circles and centers marked
output_path = "fake_files/fakeimage_results_realistic.fits"
hdu = fits.PrimaryHDU(output_image)
hdul_with_circles = fits.HDUList([hdu])
hdul_with_circles.writeto(output_path, overwrite=True)

print(f"Final output saved to {output_path}")

os.system(f"open {output_path}")