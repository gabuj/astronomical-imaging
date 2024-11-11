from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import creating_fake_image
import finding_center_radius
import pandas as pd
import  takeout_bleeing
import background_estimation
#parameters to create fake image:
image_size = (1028, 1028)  # Size of the image (512x512 pixels)
centers = [(200, 200),(600, 600)]
galaxy_peaks = [2000, 6000]
sigmas = [4, 6]
noise_level = 20
background_value= 100

#create the data
# data=creating_fake_image.create_fake_image(image_size, centers, galaxy_peaks, sigmas,background_value,noise_level)

#without making the data each time can just import it
distant2_data_path='fake_files/fake_image_2distantbright.npy'
distant3_data_path='try_deblendingg/fake_image_2distantbright.npy'
data=np.load(distant2_data_path)

original_data=np.copy(data)
#parameters for the background estimation
fraction_bin=4 #num bins is data shape/fraction_bin
sigmas_thershold = 3#how many sigmas of std after background is the threshold

#paramters for finding centers and radius
#set max radius being max distance from center to  edge of image    IMRPVOE THIS
max_radius=int(np.sqrt(data.shape[0]**2+data.shape[1]**2)/4)
backgroundfraction_tolerance=0.9



#final files parameters
vot_file = 'try_deblendingg/galaxy_catalog.vot'
vot_highintensity_file = "try_deblendingg/highestintensity_galaxies.vot"
#transform df to cat file
cat_file = 'try_deblendingg/galaxy_catalog.cat'
cat_highintensity_file = "try_deblendingg/highestintensity_galaxies.cat"   




# background_thershold=background_estimation.finding_background(data, fraction_bin, sigmas_thershold)

background_thershold=background_value+5*noise_level

#bleeding centerss
bleeding_centers= [(3217,1427), (2281,905),(2773,974),(3315,776)] #list of (y, x) coordinates of the centers of the bleeding regions

#show the image
plt.imshow(data, cmap='gray')
plt.colorbar()
plt.show()


#take out the bleeding regions
# data=takeout_bleeing.takeou_bleeing(data,bleeding_centers,background_thershold)

#still have to do: take out bad data


centers_list,radii_list=finding_center_radius.finding_centers_radii(data,max_radius,backgroundfraction_tolerance,background_thershold,max_radius)
x, y = np.indices(data.shape)



# Define the Sérsic profile
def sersic_profile(r, I_e, r_e, n):
    b_n = 1.9992 * n - 0.3271  # Approximate value for b_n
    return I_e * np.exp(-b_n * ((r / r_e)**(1/n) - 1))

# Create radial distances and intensities for the galaxy
def radial_profile(data, x_center, y_center, max_radius, r):


        # Step 3: Calculate the radial intensity profile by averaging pixel values at each radius
    radial_distances=np.arange(1,max_radius)
    radial_intensity = np.array([data[r == rad].mean() if np.any(r == r) else 0 for rad in radial_distances])
    return radial_distances, radial_intensity

# Fit the Sérsic profile to the radial data
def fit_sersic(data, x_center, y_center, max_radius, r):
    radii, intensities = radial_profile(data, x_center, y_center, max_radius, r)
    I_e_guess = np.max(intensities)
    r_e_guess = max_radius / 2
    n_guess = 4
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


def otherway_flux_within_radius(data, x_center, y_center, max_radius, r):
    radii, intensities = radial_profile(data, x_center, y_center, max_radius, r)
    total_flux = np.sum(intensities)
    total_flux_err = np.sqrt(np.sum(intensities)) #not correct but don't know how to do it
    return total_flux, total_flux_err

#create data with bakcground=0 and star positions with their intensity
data_with_star_positions = np.zeros(data.shape)

total_fluxes = []
total_fluxes_err = []
data=original_data
for i, center in enumerate(centers_list):
    
    y_center, x_center = center
    radius = radii_list[i]
    #convert to integer for binning
    r = np.sqrt((x - x_center)**2 + (y - y_center)**2)
    r = r.astype(int)
    try:
        I_e, r_e, n, I_e_err, r_e_err, n_err = fit_sersic(data, x_center, y_center, radius, r)
        total_fluxes.append(flux_within_radius(I_e, r_e, n, I_e_err, r_e_err, n_err)[0])
        total_fluxes_err.append(flux_within_radius(I_e, r_e, n, I_e_err, r_e_err, n_err)[1])
    except RuntimeError:
        print(f"Could not fit Sérsic profile for galaxy {i + 1}")
        #fit in another way
        total_fluxes.append(otherway_flux_within_radius(data, x_center, y_center, radius, r)[0])
        total_fluxes_err.append(otherway_flux_within_radius(data, x_center, y_center, radius, r)[1])
        continue
    
    #to check if doinf correct job create own image with centers, raii and intensity found and see if same aas old image
    galaxy_profile = sersic_profile(r, I_e, r_e, n)
    #add the galaxy profile around its center
    data_with_star_positions += galaxy_profile
    
# plot original image and created model image
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
    
    
    
    
    
#CREATE CATALOG FILE
#how many largest galaxy do you want to see?
num_largest = 10


#create dictionary with x and y coordinates and intensity
galaxies = []
for i, (y, x) in enumerate(centers_list):
    galaxy = {"x": x, "y": y, "intensity": total_fluxes[i],"intensity error": total_fluxes_err[i]}
    galaxies.append(galaxy)

#create csv file with geader x, y and intensity and values of the galaxies
df = pd.DataFrame(galaxies)
#only pick out top 10 high intensity galaxies
df_largest = df.nlargest(num_largest, "intensity")

with open(cat_file, "w") as f:
    f.write("# x y intensity\n")
    for i, row in df.iterrows():
        f.write(f"{row['x']} {row['y']} {row['intensity']}\n")
        
with open(cat_highintensity_file, "w") as f:
    f.write("# x y intensity\n")
    for i, row in df_largest.iterrows():
        f.write(f"{row['x']} {row['y']} {row['intensity']}\n")

#transform from df to votable file
from astropy.table import Table
from astropy.io.votable import writeto
vot_file = "galaxies.vot"
table= Table.from_pandas(df)
writeto(table, vot_file)
print("vot File created")
table_highintensity= Table.from_pandas(df_largest)
writeto(table_highintensity, vot_highintensity_file)