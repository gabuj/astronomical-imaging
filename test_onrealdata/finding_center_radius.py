import numpy as np
from scipy.ndimage import gaussian_filter
import background_estimation
import matplotlib.pyplot as plt
# Loop to detect galaxies until no significant peak remains


def finding_centers_radii(image_data,background_threshold,max_possible_radius,overexposed_threshold):
    #PARAMETERS WE CAN CORRECT
    
    overthresh_persentage=0.6 #was 0.7
    min_intensity_stillgalaxy=background_threshold*1.1 #was 20
    max_intensity_no_galaxy=background_threshold*0.8 #was 5
    # Parameters
    output_image = image_data.copy()  # Image for marking centers and circles
    galaxy_count = 1  # Counter for detected galaxies

    # Loop to detect galaxies until no significant peak remains
    centers_list=[]
    radii_list=[]
    while True:
        # Step 1: Find the highest pixel in the image (galaxy center)
        highest_pixel_value = image_data.max()
        print(f"Highest pixel value: {highest_pixel_value}")
        if highest_pixel_value < background_threshold:  # Stop if no significant peaks remain
            break

        # Find the coordinates of the highest pixel in the image
        highest_pixel_coords = np.unravel_index(np.argmax(image_data), image_data.shape)
        center_y, center_x = highest_pixel_coords

        # Mark the center of the galaxy in black on the output image
        output_image[center_y, center_x] = image_data.min()  # Mark galaxy center in black

        # Step 2: Calculate radial distances from the highest pixel (assumed center)
        y_indices, x_indices = np.indices(image_data.shape)
        radii = np.sqrt((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2)
        radii = radii.astype(int)

        # Step 3: Calculate the radial intensity profile by averaging pixel values at each radius

        radial_profile = np.array([image_data[radii == r].mean() if np.any(radii == r) else 0 for r in range(max_possible_radius)]) #if 

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
            if np.sum(radius_values < max_intensity_no_galaxy) >= overthresh_persentage * len(radius_values) and np.any(radius_values > min_intensity_stillgalaxy):
                blending_detected = True
                boundary_radius = i  # Mark the boundary for blending
                print(f"Blending detected for Galaxy {galaxy_count} at radius {boundary_radius}")
                break
            
            # If we've reached a radius of 70 without detecting blending, exit and proceed with non-blending
            elif current_radius >= 70:
                blending_detected = False
                print(f"No blending detected within radius 70 for Galaxy {galaxy_count}. Proceeding with non-blending case.")
                break

        # If blending is not detected, determine the boundary based on background threshold
        if not blending_detected:
            threshold_radius = None
            for r in range(len(smoothed_profile)):
                # Stop expanding as soon as the intensity falls below background threshold
                if smoothed_profile[r] < background_threshold:
                    threshold_radius = r
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
        image_data[radii <= mask_radius] = background_threshold - 1  # Set to background - 1
        #if radius is <2, we can't detect the galaxy
        if threshold_radius<2:
            continue
            if highest_pixel_value > overexposed_threshold:
                continue
        
        print(f"Galaxy {galaxy_count} - Radius: {threshold_radius}, Center: ({center_x}, {center_y})")
        centers_list.append((center_y,center_x))
        radii_list.append(threshold_radius)
        galaxy_count += 1
    # Print the total number of galaxies detected
    print(f"Total number of galaxies detected: {galaxy_count}")

    plt.imshow(output_image, cmap='gray')
    plt.colorbar()
    plt.title('Detected Galaxies')
    plt.show()
    
    return centers_list,radii_list