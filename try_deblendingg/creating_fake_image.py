import numpy as np
import matplotlib.pyplot as plt



def add_galaxy(image_data, center_x, center_y, galaxy_peak, sigma):
    # Generate a Gaussian profile for the galaxy
    image_size=image_data.shape
    for x in range(image_size[0]):
        for y in range(image_size[1]):
            # Calculate distance from the center
            distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            # Apply a Gaussian profile
            image_data[x, y] += galaxy_peak * np.exp(-distance ** 2 / (2 * sigma ** 2))
    return image_data

def add_background_noise(image_data, noise_level, image_size,background_value):
    noise = np.random.normal(background_value, noise_level, image_size)
    image_data += noise
    # Ensure no negative values after adding noise
    image_data = np.clip(image_data, 0, None)
    return image_data

# Add the galaxies to the image
def create_fake_image(image_size, centers, galaxy_peaks, sigmas, background_value,noise_level):
    image_data = np.zeros(image_size)
    #add background noise
    image_data = add_background_noise(image_data, noise_level, image_size,background_value)
    #add galaxies
    for center, galaxy_peak, sigma in zip(centers, galaxy_peaks, sigmas):
        image_data = add_galaxy(image_data, center[0], center[1], galaxy_peak, sigma)
    return image_data

#parameters to create fake image:
image_size = (1028, 1028)  # Size of the image (512x512 pixels)
centers = [(200, 200), (800, 800)]
galaxy_peaks = [2000, 6000]
sigmas = [4, 5]
noise_level = 20
background_value= 100


# image_data = create_fake_image(image_size, centers, galaxy_peaks, sigmas, background_value, noise_level)
# #save image as npy file
# data_path='fake_files/fake_image_2distantbright.npy'
# np.save(data_path, image_data)
 