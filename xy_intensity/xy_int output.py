from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import creating_fake_image
import finding_center_radius

#parameters to create fake image:
image_size = (1028, 1028)  # Size of the image (512x512 pixels)
centers = [(200, 200), (300, 300), (400, 400)]
galaxy_peaks = [2000, 1500, 6000]
sigmas = [4, 5, 6]
noise_level = 20
background_value= 100


data=creating_fake_image.create_fake_image(image_size, centers, galaxy_peaks, sigmas,background_value,noise_level)
#show the image
plt.imshow(data, cmap='gray')
plt.colorbar()
plt.show()


centers_radii=finding_center_radius.finding_centers_radii(data)

# Define the Sérsic profile
def sersic_profile(r, I_e, r_e, n):
    b_n = 1.9992 * n - 0.3271  # Approximate value for b_n
    return I_e * np.exp(-b_n * ((r / r_e)**(1/n) - 1))

# Create radial distances and intensities for the galaxy
def radial_profile(data, x_center, y_center, max_radius, r):
    y, x = np.indices(data.shape)
    radial_distances = np.arange(1, max_radius)
    radial_intensity = [data[(r >= rad - 1) & (r < rad + 1)].mean() for rad in radial_distances]
    return radial_distances, radial_intensity

# Fit the Sérsic profile to the radial data
def fit_sersic(data, x_center, y_center, max_radius, r):
    radii, intensities = radial_profile(data, x_center, y_center, max_radius, r)
    I_e_guess = np.max(intensities)
    r_e_guess = max_radius / 2
    n_guess = 2
    popt, covt = curve_fit(sersic_profile, radii, intensities, p0=[I_e_guess, r_e_guess, n_guess])
    I_e, r_e, n = popt
    I_e_err, r_e_err, n_err = np.sqrt(np.diag(covt))
    print(f"I_e: {I_e} +/- {I_e_err}")
    print(f"r_e: {r_e} +/- {r_e_err}")
    print(f"n: {n} +/- {n_err}")
    return I_e, r_e, n, I_e_err, r_e_err, n_err

def flux_within_radius(I_e, r_e, n, I_e_err, r_e_err, n_err):
    # Calculate the total flux within the threshold radius
    b_n = 1.9992 * n - 0.3271
    total_flux = 2 * np.pi * I_e * r_e**2 * n * np.exp(b_n) / (b_n**(2 * n))  # Sérsic total flux formula
    total_flux_err = total_flux * np.sqrt((I_e_err / I_e)**2 + (2 * r_e_err / r_e)**2 + (n_err / n)**2)
    return total_flux, total_flux_err
#create data with bakcground=0 and star positions with their intensity
data_with_star_positions = np.zeros(data.shape)

total_fluxes = []
total_fluxes_err = []
for y_center, x_center, radius in centers_radii:
    x, y = np.indices(data.shape)
    r = np.sqrt((x - x_center)**2 + (y - y_center)**2)
    #convert to integer for binning
    r = r.astype(int)
    I_e, r_e, n, I_e_err, r_e_err, n_err = fit_sersic(data, x_center, y_center, radius, r)
    total_fluxes.append(flux_within_radius(I_e, r_e, n, I_e_err, r_e_err, n_err)[0])
    total_fluxes_err.append(flux_within_radius(I_e, r_e, n, I_e_err, r_e_err, n_err)[1])
    galaxy_profile = sersic_profile(r, I_e, r_e, n)
    #add the galaxy profile around its center
    data_with_star_positions += galaxy_profile
    
# plot original image and model image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(data, cmap='gray', origin='lower')
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(data_with_star_positions, cmap='gray', origin='lower')
plt.title('Model Image')
plt.show()

for i in range(len(total_fluxes)):
    print(f"Total flux for galaxy {i + 1}: {total_fluxes[i]} +/- {total_fluxes_err[i]}")
    