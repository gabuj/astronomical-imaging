import numpy as np
from scipy.ndimage import gaussian_filter
import background_estimation
import matplotlib.pyplot as plt
import subprocess
from astropy.io import fits
# Loop to detect galaxies until no significant peak remains


def finding_centers_radii(image_data, background_threshold, max_possible_radius, overexposed_threshold, file_path):
    # PARAMETERS WE CAN CORRECT

    # Important tuning percentage parameters

    # Radius relative to hole radius in respect to the first centre where to start
    min_rad = 0.3

    # Relative intensity of second centre
    relative_int = 0.01

    # Initialize lists to store centers and radii of detected galaxies
    centers_list = []  # List to store tuples of (center_x, center_y)
    radii_list = []     # List to store threshold_radius for each galaxy

    with fits.open(file_path) as hdul:
        image_data = hdul[0].data.copy()  # Copy of the 2D array of pixel values

    # Parameters
    background_level = 3415
    noise_level = 5
    # std_multiplier = 5
    # background_threshold = background_level + std_multiplier * noise_level  # Consider anything less as background
    std_multiplier_highest_pixel = 30
    std_highest_pixel_threshold = background_level + std_multiplier_highest_pixel * noise_level

    output_image = image_data.copy()  # Image for marking centers and circles
    galaxy_count = 0  # Counter for detected galaxies

    # Keep track of processed pixels
    processed_pixels = np.zeros_like(image_data, dtype=bool)

    # Loop to detect galaxies until no significant peak remains
    while True:
        detected_galaxies = []  # Initialize list for detected galaxies in this iteration
        # Step 1: Find the highest pixel in the image (galaxy center)
        # Ignore already processed pixels
        masked_image = np.where(processed_pixels, background_level, image_data)
        highest_pixel_value = masked_image.max()
        if highest_pixel_value < std_highest_pixel_threshold:  # Stop if no significant peaks remain
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
                radius_values > background_threshold + 60):
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
            # Proceed to detect the second galaxy
            detected_galaxies.append((center_y1, center_x1, threshold_radius1))
            
            # Append to centers_list and radii_list
            centers_list.append((center_x1, center_y1))
            radii_list.append(threshold_radius1)
            
            # Start from radius 10 from the first galaxy's center
            min_radius = int(threshold_radius1 * min_rad)
            max_radius = threshold_radius1 + 200  # Extend beyond the first galaxy's boundary
            found_second_center = False
            for current_radius in range(min_radius, max_radius):
                # Get the indices of pixels at this radius
                shell_mask = (radii1 == current_radius) & ~processed_pixels
                y_indices_shell, x_indices_shell = np.where(shell_mask)
                for y, x in zip(y_indices_shell, x_indices_shell):
                    # Exclude the first center
                    if (y == center_y1) and (x == center_x1):
                        continue
                    # Get the value of the current pixel
                    pixel_value = image_data[y, x]
                    # Check if the pixel value is greater than its immediate neighbors
                    neighbors = []
                    if y > 0:
                        neighbors.append(image_data[y - 1, x])
                    if y < image_data.shape[0] - 1:
                        neighbors.append(image_data[y + 1, x])
                    if x > 0:
                        neighbors.append(image_data[y, x - 1])
                    if x < image_data.shape[1] - 1:
                        neighbors.append(image_data[y, x + 1])
                    # Check if the pixel value is greater than all its neighbors
                    if all(pixel_value > neighbor for neighbor in neighbors):
                        # Check if pixel value is within percentage of the first center's value
                        if abs(pixel_value - highest_pixel_value) <= relative_int * highest_pixel_value:
                            # Found the second center
                            center_y2, center_x2 = y, x
                            second_peak_value = pixel_value
                            galaxy_count += 1
                            print(f"Second Galaxy {galaxy_count} detected at Center: ({center_x2}, {center_y2}), Peak Brightness: {second_peak_value}")
                            # Calculate radial distances for the second galaxy
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
                            for j in range(len(smoothed_profile2) - 1):
                                current_radius2 = j + 1
                                radius_values2 = image_data[(radii2 == current_radius2) & ~processed_pixels]  # Exclude already processed pixels

                                # Check blending condition for second galaxy
                                if np.any(radius_values2 <= background_level + 10):
                                    threshold_radius2 = current_radius2  # Mark the boundary for blending
                                    print(f"Blending detected for Galaxy {galaxy_count} at radius {threshold_radius2}")
                                    break

                            if threshold_radius2 is None:
                                threshold_radius2 = max_possible_radius
                            detected_galaxies.append((center_y2, center_x2, threshold_radius2))
                            
                            # Append to centers_list and radii_list
                            centers_list.append((center_x2, center_y2))
                            radii_list.append(threshold_radius2)
                            
                            found_second_center = True
                            break  # Exit the inner loop
                if found_second_center:
                    break  # Exit the outer loop
            if not found_second_center:
                print("Second galaxy not detected in the blending region.")
                # Proceed to process the first galaxy only
                # Draw the circle and mark pixels for the first galaxy only
                # Draw the circle for the first galaxy
                circle_mask1 = (radii1 >= threshold_radius1 - 1) & (radii1 <= threshold_radius1 + 1)
                output_image[circle_mask1] = output_image.max()  # Boundary in white
                # Mark the center of the first galaxy in black
                output_image[center_y1, center_x1] = image_data.min()
                # Mark pixels as processed for the first galaxy
                processed_pixels[radii1 <= threshold_radius1] = True
        else:
            # After detecting a single galaxy
            # Determine the boundary based on the first galaxy's profile
            threshold_radius1 = None
            for r in range(len(smoothed_profile1)):
                if smoothed_profile1[r] < background_threshold:
                    threshold_radius1 = r
                    print(f"Galaxy {galaxy_count} boundary detected at radius {threshold_radius1}")
                    break
            if threshold_radius1 is None:
                threshold_radius1 = max_possible_radius
            detected_galaxies.append((center_y1, center_x1, threshold_radius1))
            
            # Append to centers_list and radii_list
            centers_list.append((center_x1, center_y1))
            radii_list.append(threshold_radius1)

        # Apply masking for all detected galaxies in this iteration
        for galaxy in detected_galaxies:
            center_y, center_x, threshold_radius = galaxy
            mask_radius = threshold_radius + 1  # Slightly larger than the detected radius
            y_indices_mask, x_indices_mask = np.indices(image_data.shape)
            radii_mask = np.sqrt((x_indices_mask - center_x) ** 2 + (y_indices_mask - center_y) ** 2)
            radii_mask = radii_mask.astype(int)

            # Masking the galaxy region
            image_data[radii_mask <= mask_radius] = background_threshold - 1  # Set to background - 1

            # Additional conditions to continue or skip
            if threshold_radius < 3:
                print(f"Galaxy at ({center_x}, {center_y}) has a threshold radius less than 3. Skipping masking.")
                continue
            if image_data[center_y, center_x] > overexposed_threshold:
                print(f"Galaxy at ({center_x}, {center_y}) is overexposed. Skipping masking.")
                continue

            # Mark pixels as processed for this galaxy
            processed_pixels[radii_mask <= mask_radius] = True  # Ensure these pixels are not reprocessed

        # Continue with the loop
    # Print the total number of galaxies detected
    print(f"\nTotal number of galaxies detected: {galaxy_count}")

    # Optional: Display the image
    # plt.imshow(output_image, cmap='gray')
    # plt.colorbar()x
    # plt.title('Detected Galaxies')
    # plt.show()
    
    return centers_list, radii_list


# name6 = "1_extended_diffuses"
# file_path = f"/Users/yuri/Desktop/Year 3 Lab/Astronomical Image Processing/Git repository/astronomical-imaging/fake_files/{name6}.fits"
# with fits.open(file_path) as hdul:
#     image_data = hdul[0].data.copy()  # Copy of the 2D array of pixel values
# background_threshold = 
# finding_centers_radii(image_data, background_threshold, max_possible_radius, overexposed_threshold, file_path)