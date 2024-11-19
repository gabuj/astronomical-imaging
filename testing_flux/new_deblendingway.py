import numpy as np
from scipy.ndimage import gaussian_filter
import background_estimation
import matplotlib.pyplot as plt
import subprocess
from astropy.io import fits
from scipy.ndimage import label, center_of_mass
from scipy.ndimage import maximum_filter

# Loop to detect galaxies until no significant peak remains


def finding_centers_radii(data, max_possible_radius, overexposed_threshold, background_level, background_std):
    # PARAMETERS WE CAN CORRECT

    # Important tuning percentage parameters
    image_data = np.copy(data)
    # Radius relative to hole radius in respect to the first centre where to start
    min_rad = 0.5
    minimum_radius = 4
    # Relative intensity of second centre
    relative_int = 0.01

    # Initialize lists to store centers and radii of detected galaxies
    centers_list = []  # List to store tuples of (center_x, center_y)
    radii_list = []     # List to store threshold_radius for each galaxy

    # with fits.open(file_path) as hdul:
    #     image_data = hdul[0].data.copy()  # Copy of the 2D array of pixel values

    # Parameters
    std_multiplier = 5
    higher_thanbackground_blended_galaxy=1
    background_threshold = background_level + std_multiplier * background_std # Consider anything less as background
    std_multiplier_highest_pixel = 7
    std_highest_pixel_threshold = background_level + std_multiplier_highest_pixel * background_std

    output_image = image_data.copy()  # Image for marking centers and circles
    galaxy_count = 0  # Counter for detected galaxies

    # Keep track of processed pixels
    processed_pixels = np.zeros_like(image_data, dtype=bool)

    # Loop to detect galaxies until no significant peak remains
    last_radius=0
    last_center=[0,0] #y,x
    while True:
        detected_galaxies = []  # Initialize list for detected galaxies in this iteration
        masked_galaxies = []  # Initialize list for masked galaxies in this iteration
        # Step 1: Find the highest pixel in the image (galaxy center)
        # Ignore already processed pixels
        num_proc=np.sum(processed_pixels)
        print("number of processed pixels: ",num_proc)
        masked_image = np.where(processed_pixels, background_level, image_data) #masked image is the image_data with the background level
        #show masked image
        # plt.imshow(masked_image, cmap='gray')
        # plt.colorbar()
        # plt.title('Masked Image')
        # plt.show()
        
        highest_pixel_value = masked_image.max()        
        if highest_pixel_value > std_highest_pixel_threshold:  # Stop if no significant peaks remain
            #center
            #center_x1, center_y1 = np.unravel_index(masked_image.argmax(), masked_image.shape)
            flat_index = masked_image.argmax()
            center_y1, center_x1 = np.unravel_index(flat_index, masked_image.shape)
            print(f"highest pixel value found at: ({center_x1}, {center_y1}), and the value is {highest_pixel_value}")
            print(f"highest pixel value should be {masked_image[center_y1,center_x1]}")
        elif highest_pixel_value > background_threshold and highest_pixel_value <= std_highest_pixel_threshold:
             binary_mask = (masked_image > background_threshold)
             center_x1, center_y1 = np.unravel_index(masked_image.argmax(), masked_image.shape)
             print(f"cemter found with centroid at: ({center_x1}, {center_y1}), and the value is {highest_pixel_value}")
             labels, num_labels = label(binary_mask)
             #if the center is part of a label, find the center of mass of that label
             galaxy_label = labels[center_y1,center_x1]
             if galaxy_label == 0:
                 print("Center not part of any label")
                 #mask off previous galaxy with mask radius +1
                 mask_centery, mask_centerx = last_center
                 mask_radius = last_radius + 2
                 y_indices_mask, x_indices_mask = np.indices(image_data.shape)
                 radii_mask = np.sqrt((x_indices_mask - mask_centerx) ** 2 + (y_indices_mask - mask_centery) ** 2)
                 radii_mask = radii_mask.astype(int)
                    # Masking the galaxy region
                #  image_data[radii_mask <= mask_radius] = background_level  # Set to background
                 processed_pixels[radii_mask <= mask_radius] = True  # Ensure these pixels are not reprocessed
                 last_radius=mask_radius
                 last_center=[center_y1,center_x1]
                 continue
             center_x1, center_y1 = center_of_mass(binary_mask, labels, galaxy_label)      
             center_x1=center_x1.astype(int)
             center_y1=center_y1.astype(int)

        else:
            print("No significant peaks remain")
            break
        
        # if [center_y1, center_x1] == last_center:
        #     print("Same center as last iteration. masking this part.")
        #     mask_centery, mask_centerx = last_center
        #     mask_radius = last_radius + 20
        #     print(f"masking with mask radius: {mask_radius}")
        #     y_indices_mask, x_indices_mask = np.indices(image_data.shape)
        #     radii_mask = np.sqrt((x_indices_mask - mask_centerx) ** 2 + (y_indices_mask - mask_centery) ** 2)
        #     radii_mask = radii_mask.astype(int)
        #     # Masking the galaxy region
        #     processed_pixels[radii_mask <= mask_radius] = True  # Ensure these pixels are not reprocessed 
        #     #show masked radius
        #     masked_image[radii_mask<=mask_radius]  == 4000
        #     masked_image[mask_centery,mask_centerx] = 4000
        #     plt.imshow(masked_image, cmap='gray')
        #     plt.colorbar()
        #     plt.title('Masked Image shoiwing radius')
        #     plt.show() 
        #     last_radius=mask_radius
        #     continue  
        
        print(f"value at center is {masked_image[center_y1,center_x1]}")
        if [center_y1, center_x1] == last_center:
            print("Same center as last iteration. masking this part.")
            mask_centery, mask_centerx = last_center
            mask_radius = last_radius + 20
            print(f"masking with mask radius: {mask_radius}")
            y_indices_mask, x_indices_mask = np.indices(image_data.shape)
            radii_mask = np.sqrt((x_indices_mask - mask_centerx) ** 2 + (y_indices_mask - mask_centery) ** 2)
            radii_mask = radii_mask.astype(int)
            search_region=masked_image.copy()
            search_region[~(radii_mask >= mask_radius)]=background_level #everything outside the mask radius is set to background level
            #find closest label to center and take it away
            labels, num_labels = label(search_region > background_threshold)
            galaxy_label = labels[mask_centery,mask_centerx]
            print(f"galaxy label: {galaxy_label}")
            processed_pixels[labels == galaxy_label] = True
            last_radius=mask_radius
                     
            continue  
        
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
        #smoothed_profile1 = gaussian_filter(radial_profile1, sigma=2)
        smoothed_profile1 = radial_profile1

        # Blending detection variables
        blending_detected = False
        boundary_radius1 = None

        # Step 4: Blending detection for the first galaxy
        for i in range(len(smoothed_profile1) - 1):
            current_radius = i + 1
            radius_values = image_data[(radii1 == current_radius) & ~processed_pixels]  # Exclude already processed pixels

            # Check blending condition
            #If 70% is less than threshold and at least 1 value is more than threshold
            if np.sum((radius_values < background_threshold))/(len(radius_values)) >= 0.7 and np.any(
                radius_values > background_threshold + higher_thanbackground_blended_galaxy):
                blending_detected = True
                boundary_radius1 = i  # Mark the boundary for blending
                blended_radius=boundary_radius1
                print(f"Blending detected for Galaxy {galaxy_count} at radius {boundary_radius1}")
                break

            elif current_radius >= 70:
                blending_detected = False
                print(f"No blending detected within radius 70 for Galaxy {galaxy_count}. Proceeding with non-blending case.")
                break


        #check radius normally and then compare to the blended radius
         # After detecting a single galaxy
            # Determine the boundary based on the first galaxy's profile
        threshold_radius1 = None
        for r in range(len(smoothed_profile1)):
            if smoothed_profile1[r] < background_threshold:
                threshold_radius1 = r
                not_blended_radius=threshold_radius1
                print(f"Galaxy {galaxy_count} boundary detected at radius {not_blended_radius}")
                break
        if not_blended_radius is None:
            #threshold_radius1 = max_possible_radius
            print("no first galaxy found")
        
        #compare the two radii
        if blending_detected:
            if blended_radius>not_blended_radius:
                print("blended radius is smaller so blended not detected")
                blending_detected=False
        # Step 5: Determine the boundary for the first galaxy
        if blending_detected:
            # Use the boundary_radius1 detected during blending detection
            threshold_radius1 = boundary_radius1
            # Proceed to detect the second galaxy
            masked_galaxies.append((center_y1, center_x1, threshold_radius1))
            #append to detected_galaxies if radius is greater than minimum radius and not overexposed
            if threshold_radius1 >= minimum_radius and image_data[center_y1, center_x1] < overexposed_threshold:
                detected_galaxies.append((center_y1, center_x1, threshold_radius1))
            
            # Append to centers_list and radii_list
            centers_list.append((center_x1, center_y1))
            radii_list.append(threshold_radius1)
            
            # Start from radius 10 from the first galaxy's center
            min_radius = int(threshold_radius1 * min_rad) # perché? sei sicuro non sia biased?
            max_radius = threshold_radius1*4 #search 2 radii apart from the first galaxy
            found_second_center = False
            #take away inner part of first galaxy
            temporary_mask = (radii1 >= min_radius) & (radii1 <= max_radius)
            temporary_mask_image=masked_image.copy()
            temporary_mask_image[~temporary_mask]=background_level
            local_max = (image_data == maximum_filter(temporary_mask_image, size=max_radius))
            # local_max[radii1 <= min_radius] = False
            local_max = local_max & (masked_image > background_threshold)
            # Get the indices of pixels at this radius
            y_indices_shell, x_indices_shell = np.where(local_max)
            for y, x in zip(y_indices_shell, x_indices_shell):
                print(f"local max at ({x},{y})")
                pixel_value = image_data[y, x]           
                if pixel_value >= background_threshold:
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
                        if np.sum((radius_values2 < background_threshold))/(len(radius_values2)) >= 0.3:
                            threshold_radius2 = current_radius2  # Mark the boundary for blending
                            print(f"Blending detected for Galaxy {galaxy_count} at radius {threshold_radius2}")
                            break

                    if threshold_radius2 is None:
                        # threshold_radius2 = max_possible_radius
                        print(f"not found radius of second blended galaxy")
                    masked_galaxies.append((center_y2, center_x2, threshold_radius2))
                    #append to detected_galaxies if radius is greater than minimum radius and not overexposed
                    if threshold_radius2 >= minimum_radius and image_data[center_y2, center_x2] < overexposed_threshold:
                        detected_galaxies.append((center_y2, center_x2, threshold_radius2))
                        
                    
                    # Append to centers_list and radii_list
                    centers_list.append((center_x2, center_y2))
                    radii_list.append(threshold_radius2)
                    
                    found_second_center = True
            if not found_second_center:
                print("Second galaxy not detected in the blending region.")
                # Proceed to process the first galaxy only
                # Draw the circle and mark pixels for the first galaxy only
                # Draw the circle for the first galaxy
                # circle_mask1 = (radii1 >= threshold_radius1 - 1) & (radii1 <= threshold_radius1 + 1)
                # output_image[circle_mask1] = output_image.max()  # Boundary in white
                # # Mark the center of the first galaxy in black
                # output_image[center_y1, center_x1] = background_level
                # Mark pixels as processed for the first galaxy
                processed_pixels[radii1 <= threshold_radius1] = True
        else:
        
            masked_galaxies.append((center_y1, center_x1, not_blended_radius))
            #append to detected_galaxies if radius is greater than minimum radius and not overexposed
            if not_blended_radius >= minimum_radius and image_data[center_y1, center_x1] < overexposed_threshold:
                detected_galaxies.append((center_y1, center_x1, not_blended_radius))
                
            
            # Append to centers_list and radii_list
            centers_list.append((center_x1, center_y1))
            radii_list.append(not_blended_radius)
            # print(f"detected galaxies: {detected_galaxies}")
        # Apply masking for all detected galaxies in this iteration
        for galaxy in masked_galaxies:
            center_y, center_x, threshold_radius = galaxy
            mask_radius = threshold_radius + 2  # Slightly larger than the detected radius
            print(f"new galaxy masking with mask radius: {mask_radius}")
            y_indices_mask, x_indices_mask = np.indices(image_data.shape)
            radii_mask = np.sqrt((x_indices_mask - center_x) ** 2 + (y_indices_mask - center_y) ** 2)
            radii_mask = radii_mask.astype(int)
            # Masking the galaxy region
            # image_data[radii_mask <= mask_radius] = background_threshold - 1  # Set to background - 1
            processed_pixels[radii_mask <= mask_radius] = True  # Ensure these pixels are  reprocessed
            last_radius=threshold_radius
            last_center=[center_y,center_x]
            
            # Additional conditions to continue or skip
            if threshold_radius < minimum_radius:
                print(f"Galaxy at ({center_x}, {center_y}) has a threshold radius less than 4. Skipping adding to galaxies.")
                continue
            if image_data[center_y, center_x] > overexposed_threshold:
                print(f"Galaxy at ({center_x}, {center_y}) is overexposed. Skipping adding to galaxies.")
                continue

            # Mark pixels as processed for this galaxy
            print(f"found galaxy at ({center_x}, {center_y}) with threshold radius {threshold_radius}")
            #draw galaxy circle in output image
            circle_mask = (radii_mask >= threshold_radius - 1) & (radii_mask <= threshold_radius + 1)
            output_image[circle_mask] = output_image.max()
            # Mark the center of the galaxy in black
            output_image[center_y, center_x] = background_level
            galaxy_count+=1

        # Continue with the loop
    # Print the total number of galaxies detected
    print(f"\nTotal number of galaxies detected: {galaxy_count}")

    # Optional: Display the image
    plt.imshow(output_image, cmap='gray')
    plt.colorbar()
    plt.title('Detected Galaxies')
    plt.show()
    
    return centers_list, radii_list


# name6 = "1_extended_diffuses"
# file_path = f"/Users/yuri/Desktop/Year 3 Lab/Astronomical Image Processing/Git repository/astronomical-imaging/fake_files/{name6}.fits"
# with fits.open(file_path) as hdul:
#     image_data = hdul[0].data.copy()  # Copy of the 2D array of pixel values
# background_threshold = 
# finding_centers_radii(image_data, background_threshold, max_possible_radius, overexposed_threshold, file_path)