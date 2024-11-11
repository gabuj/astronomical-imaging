import numpy as np
import matplotlib.pyplot as plt
# Parameters for the fake galaxy image
image_size = (512, 512)  # Size of the image (512x512 pixels)
centers = [(100, 100), (200, 200), (300, 300)]
galaxy_peaks = [100, 150, 200]
sigmas = [20, 30, 40]
noise_level = 10  # Standard deviation of the Gaussian noise



def add_galaxy(image_data, center_x, center_y, galaxy_peak, sigma):
    # Generate a Gaussian profile for the galaxy
    for x in range(image_size[0]):
        for y in range(image_size[1]):
            # Calculate distance from the center
            distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            # Apply a Gaussian profile
            image_data[x, y] += galaxy_peak * np.exp(-distance ** 2 / (2 * sigma ** 2))
    return image_data

def add_background_noise(image_data, noise_level):
    noise = np.random.normal(0, noise_level, image_size)
    image_data += noise
    # Ensure no negative values after adding noise
    image_data = np.clip(image_data, 0, None)
    return image_data

# Add the galaxies to the image
def create_fake_image(image_size, centers, galaxy_peaks, sigmas, noise_level):
    image_data = np.zeros(image_size)
    #add background noise
    image_data = add_background_noise(image_data, noise_level)
    #add galaxies
    for center, galaxy_peak, sigma in zip(centers, galaxy_peaks, sigmas):
        image_data = add_galaxy(image_data, center[0], center[1], galaxy_peak, sigma)
    return image_data

# image_data = create_fake_image(image_size, centers, galaxy_peaks, sigmas, noise_level)
# #show the image
# plt.imshow(image_data, cmap='gray')
# plt.colorbar()
# plt.show()