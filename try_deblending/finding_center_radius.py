import numpy as np
from scipy.ndimage import gaussian_filter
import background_estimation
import matplotlib.pyplot as plt
import subprocess
# Loop to detect galaxies until no significant peak remains


def finding_centers_radii(image_data,background_threshold,max_possible_radius,overexposed_threshold):
    #PARAMETERS WE CAN CORRECT
    
    # Important tuning percentage parameters

# Radius relative to hole radius in respect to the first center where to start
min_rad = 0.2

# Relative intensity of additional centers
relative_int = 0.01

# Load the FITS file
with fits.open(file_path) as hdul:
    image_data = hdul[0].data.copy()  # Copy of the 2D array of pixel values

# Parameters
background_level = 3415
noise_level = 5
std_multiplier = 5
background_threshold = background_level + std_multiplier * noise_level  # Consider anything less as background
std_multiplier_highest_pixel = 30
std_highest_pixel_threshold = background_level + std_multiplier_highest_pixel * noise_level

output_image = image_data.copy()  # Image for marking centers and circles
galaxy_count = 0  # Counter for detected galaxies

# Keep track of processed pixels
processed_pixels = np.zeros_like(image_data, dtype=bool)

# Precompute indices for efficiency
y_indices, x_indices = np.indices(image_data.shape)

# Function to find additional centers
def find_additional_centers(center_y, center_x, galaxy_centers, highest_pixel_value, max_additional=2):
    """
    Find additional galaxy centers within a blended region.
    Returns a list of tuples: (y, x, peak_value)
    """
    additional_centers = []
    for _ in range(max_additional):
        found_center = False
        # Define search region around the current center
        min_radius_search = int(threshold_radius1 * min_rad)
        max_radius_search = threshold_radius1 + 200  # Adjust as necessary

        # Calculate radial distances from the current center
        radii = np.sqrt((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2)
        radii = radii.astype(int)

        for current_radius in range(min_radius_search, max_radius_search):
            # Get the indices of pixels at this radius
            shell_mask = (radii == current_radius) & ~processed_pixels
            y_shell, x_shell = np.where(shell_mask)
            for y, x in zip(y_shell, x_shell):
                # Exclude existing centers
                if any((y == cy and x == cx) for cy, cx, _ in galaxy_centers):
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
                        # Found an additional center
                        global galaxy_count
                        galaxy_count += 1
                        print(f"Additional Galaxy {galaxy_count} detected at Center: ({x}, {y}), Peak Brightness: {pixel_value}")
                        additional_centers.append((y, x, pixel_value))
                        found_center = True
                        break  # Exit the inner loop
            if found_center:
                break  # Exit the outer loop
        if not found_center:
            print("No additional galaxy detected in the blending region.")
            break
    return additional_centers

# Loop to detect galaxies until no significant peak remains
while True:
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

    # Step 6: If blending was detected, find additional galaxy centers before modifying the image
    if blending_detected:
        # Initialize list to hold galaxy centers
        galaxy_centers = [(center_y1, center_x1, highest_pixel_value)]

        # Find up to two additional centers for a total of three galaxies
        additional_centers = find_additional_centers(center_y1, center_x1, galaxy_centers, highest_pixel_value, max_additional=2)
        galaxy_centers.extend(additional_centers)

        # If no additional centers found, proceed with processing the first galaxy only
        if not additional_centers:
            print("Proceeding with processing the first galaxy only.")
            # Draw the circle and mark pixels for the first galaxy
            circle_mask1 = (radii1 >= threshold_radius1 - 1) & (radii1 <= threshold_radius1 + 1)
            output_image[circle_mask1] = output_image.max()  # Boundary in white
            # Mark the center of the first galaxy in black
            output_image[center_y1, center_x1] = image_data.min()
            # Mark pixels as processed for the first galaxy
            processed_pixels[radii1 <= threshold_radius1] = True
        else:
            # Process each detected galaxy
            for idx, (cy, cx, peak_val) in enumerate(galaxy_centers, start=1):
                # Calculate radial distances for this galaxy
                radii = np.sqrt((x_indices - cx) ** 2 + (y_indices - cy) ** 2)
                radii = radii.astype(int)

                # Calculate the radial intensity profile
                radial_profile = np.array([
                    image_data[(radii == r) & ~processed_pixels].mean() if np.any((radii == r) & ~processed_pixels) else 0
                    for r in range(max_possible_radius)
                ])

                # Smooth the radial profile
                smoothed_profile = gaussian_filter(radial_profile, sigma=2)

                # Determine the boundary radius
                boundary_detected = False
                boundary_radius = None
                for i in range(len(smoothed_profile) - 1):
                    current_radius = i + 1
                    radius_values = image_data[(radii == current_radius) & ~processed_pixels]

                    # Check blending condition (can be adjusted based on requirements)
                    if np.sum(radius_values < background_threshold) >= 0.7 * len(radius_values) and np.any(
                        radius_values > background_threshold + 60):
                        boundary_detected = True
                        boundary_radius = i
                        print(f"Blending detected for Galaxy {galaxy_count} at radius {boundary_radius}")
                        break
                    elif current_radius >= 70:
                        boundary_detected = False
                        print(f"No blending detected within radius 70 for Galaxy {galaxy_count}. Proceeding with non-blending case.")
                        break

                if boundary_detected and boundary_radius is not None:
                    threshold_radius = boundary_radius
                else:
                    # Determine boundary based on where the smoothed profile drops below the background threshold
                    threshold_radius = None
                    for r in range(len(smoothed_profile)):
                        if smoothed_profile[r] < background_threshold:
                            threshold_radius = r
                            print(f"Galaxy {galaxy_count} boundary detected at radius {threshold_radius}")
                            break
                    if threshold_radius is None:
                        threshold_radius = max_possible_radius

                # Draw the circle for the galaxy
                circle_mask = (radii >= threshold_radius - 1) & (radii <= threshold_radius + 1)
                output_image[circle_mask] = output_image.max()  # Boundary in white
                # Mark the center of the galaxy in black
                output_image[cy, cx] = image_data.min()
                # Mark pixels as processed for the galaxy
                processed_pixels[radii <= threshold_radius] = True

                print(f"Galaxy {galaxy_count} - Radius: {threshold_radius}, Center: ({cx}, {cy})")
    else:
        # Non-blending case
        # Draw the circle and mark pixels for the first galaxy
        circle_mask1 = (radii1 >= threshold_radius1 - 1) & (radii1 <= threshold_radius1 + 1)
        output_image[circle_mask1] = output_image.max()  # Boundary in white
        # Mark the center of the first galaxy in black
        output_image[center_y1, center_x1] = image_data.min()
        # Mark pixels as processed for the first galaxy
        processed_pixels[radii1 <= threshold_radius1] = True


        print(f"Galaxy {galaxy_count} - Radius: {threshold_radius}, Center: ({center_x}, {center_y})")
        centers_list.append((center_y,center_x))
        radii_list.append(threshold_radius)
        galaxy_count += 1
    # Print the total number of galaxies detected
    print(f"Total number of galaxies detected: {galaxy_count}")

    # Step 9: Save the modified data to a new FITS file with circles and centers marked
    output_path = "/Users/yuri/Desktop/Year 3 Lab/Astronomical Image Processing/Git repository/astronomical-imaging/fake_files/fakeimage_results.fits"
    hdu = fits.PrimaryHDU(output_image)
    hdul_with_circles = fits.HDUList([hdu])
    hdul_with_circles.writeto(output_path, overwrite=True)


    # plt.imshow(output_image, cmap='gray')
    # plt.colorbar()
    # plt.title('Detected Galaxies')
    # plt.show()
    
    return centers_list,radii_list