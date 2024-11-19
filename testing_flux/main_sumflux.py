from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import creating_fake_image
import finding_center_radius
import pandas as pd
import background_estimation
from astropy.io import fits
import bad_data_clean
import new_deblendingway
#get data
max_localbackground_radius=200
fraction_bin=max_localbackground_radius*2


#creating fake image data
# Parameters for creating images
image_size = (1028, 1028)
centers = [(500, 500),(700,700),(300,300),(200,200),(800,800)]
peaks = [200,150, 100, 50, 75]
sigmas = [25, 8,50, 10, 15]
ns = [0.5, 0.5, 0.5, 0.5, 0.5]

noise_level = 5
background_level = 3415

# data=creating_fake_image.create_fake_image(image_size, centers, peaks, sigmas,background_level,noise_level,ns)
filename="fake_files/nice_lotsofgalaxies.npy"
data=np.load(filename)


#show initial image
plt.imshow(data, cmap='gray', origin='lower')
plt.colorbar()
plt.title('Initial Image')
plt.show()


#parameters for the background estimation
fraction_bin_totalbackground=3 #num bins is data shape/fraction_bin
sigmas_thershold = 5#how many sigmas of std after background is the threshold
#find background
#background_thershold=background_estimation.finding_background(data, fraction_bin_totalbackground, sigmas_thershold)
background_thershold= background_level + 5 * noise_level

background_std=noise_level

#finding radius paramters
overexposed_threshold=65535
background_gain=3 #how many times more than radius is local background radius
#create the data
# data=creating_fake_image.create_fake_image(image_size, centers, galaxy_peaks, sigmas,background_value,noise_level,ns)



# original_data=np.copy(data)

#paramters for finding centers and radius
#set max radius being max distance from center to  edge of image    IMPROVE THIS
max_radius=int(np.sqrt(data.shape[0]**2+data.shape[1]**2)/4)
backgroundfraction_tolerance=0.9



#final files parameters
vot_file = 'LAST_PUTTINGSTUFFTOGETHER/galaxy_catalog.vot'
vot_highintensity_file = "LAST_PUTTINGSTUFFTOGETHER/highestintensity_galaxies.vot"
#transform df to cat file
cat_file = "LAST_PUTTINGSTUFFTOGETHER/galaxy_catalog.cat"
cat_highintensity_file = "LAST_PUTTINGSTUFFTOGETHER/highestintensity_galaxies.cat"   



centers_list,radii_list=new_deblendingway.finding_centers_radii(data, max_radius, overexposed_threshold,background_level,background_std)
print(f"centers are {centers_list}")
x, y = np.indices(data.shape)

# Define the Sérsic profile
def sersic_profile(r, I_e, r_e, n):
    b_n = 1.9992 * n - 0.3271  # Approximate value for b_n
    return I_e * np.exp(-b_n * ((r / r_e)**(1/n) - 1))

# Create radial distances and intensities for the galaxy
def radial_profile(data, max_radius, r):
    # Step 3: Calculate the radial intensity profile by averaging pixel values at each radius
    radial_distances=np.arange(0,max_radius+1)
    radial_intensity = np.array([data[r == rad].mean() for rad in radial_distances])
    # print(f"radial distances are {radial_distances}")
    # print(f"radial intensities are {radial_intensity}")
    return radial_distances, radial_intensity

# Fit the Sérsic profile to the radial data
def fit_sersic(data, max_radius, r):
    print(f"max radius is {max_radius}")
    radii, intensities = radial_profile(data, max_radius, r)
    I_e_guess = np.max(intensities)
    r_e_guess = max_radius / 2
    n_guess = 4
    popt, covt = curve_fit(sersic_profile, radii, intensities, p0=[I_e_guess, r_e_guess, n_guess])
    I_e, r_e, n = popt
    I_e_err, r_e_err, n_err = np.sqrt(np.diag(covt))
    print(f"I_e: {I_e} +/- {I_e_err}")
    print(f"r_e: {r_e} +/- {r_e_err}")
    print(f"n: {n} +/- {n_err}")
    
    #plot data and fit to see if it is correct
    # plt.figure()
    # plt.plot(radii, intensities, label='Data')
    # plt.plot(radii, sersic_profile(radii, I_e, r_e, n), label='Fit')
    # plt.xlabel('Radius')
    # plt.ylabel('Intensity')
    # plt.legend()
    # plt.show()
    
    
    return I_e, r_e, n, I_e_err, r_e_err, n_err

def flux_within_radius(I_e, r_e, n, I_e_err, r_e_err, n_err):
    # Calculate the total flux within the threshold radius
    b_n = 1.9992 * n - 0.3271
    total_flux = 2 * np.pi * I_e * r_e**2 * n * np.exp(b_n) / (b_n**(2 * n))  # Sérsic total flux formula
    total_flux_err = total_flux * np.sqrt((I_e_err / I_e)**2 + (2 * r_e_err / r_e)**2 + (n_err / n)**2)
    return total_flux, total_flux_err


def otherway_flux_within_radius(data, max_radius, r):
    radii, intensities = radial_profile(data, max_radius, r)
    gain=1.8
    R=10
    total_flux = np.sum(intensities)
    var=0
    for intensity in intensities:
        var+=(intensity/gain) + (R/gain)**2
    total_flux_err = np.sqrt(var)
    return total_flux, total_flux_err

def take_away_localbackground(data,radius,r,background_gain):
    #take away local background value from intensity
    max_radius=max_localbackground_radius
    background_data=data[r<=max_radius]
    local_background=background_estimation.finding_local_background(background_data, fraction_bin, sigmas_thershold)
    local_background_err=0
    return local_background,local_background_err
#create data with bakcground=0 and star positions with their intensity
data_with_star_positions = np.zeros(data.shape)
image_size = data.shape
# background= np.random.normal(background_value, background_std, image_size)
# data_with_star_positions += background

total_fluxes = []
total_fluxes_err = []
flux_summed=0
for i, center in enumerate(centers_list):
    
    x_center, y_center = center
    radius = radii_list[i]
    print(f"analysing galaxy {i + 1} at center ({x_center}, {y_center}) with radius {radius}")
    #convert to integer for binning
    r = np.sqrt((x - x_center)**2 + (y - y_center)**2)
    r = r.astype(int)
    
     #take away local background value from intensity
    local_background,local_background_err=take_away_localbackground(data,radius,r,background_gain)
    temporary_data=np.copy(data).astype(float)
    #make loacl background same kind of array as data
    temporary_data-=local_background
    #show temporary data
    # imageshow=np.copy(temporary_data)
    # #show circles around the centers
    # circle_mask=(r<=radius + 1) & (r>=radius - 1)
    # imageshow[circle_mask]=imageshow.max()
    # plt.imshow(imageshow, cmap='gray', origin='lower')
    # plt.colorbar()
    # plt.title('Temporary Image')
    # plt.show()
    
    
   
    # try:
    #     I_e, r_e, n, I_e_err, r_e_err, n_err = fit_sersic(temporary_data, radius, r)
    #     flux, flux_err = flux_within_radius(I_e, r_e, n, I_e_err, r_e_err, n_err)
    #     total_fluxes.append(flux)
    #     total_fluxes_err.append(flux_err)
    # except RuntimeError:
        # print(f"Could not fit Sérsic profile for galaxy {i + 1}")
        #fit in another way
    flux,flux_err=otherway_flux_within_radius(temporary_data, radius, r)
    total_fluxes.append(flux)
    total_fluxes_err.append(flux_err)
    flux_summed+=1
        # continue
    print(f"Total flux for galaxy {i + 1}: {flux} +/- {flux_err}")
    
    
    # to check if doinf correct job create own image with centers, raii and intensity found and see if same aas old image
    # galaxy_profile = sersic_profile(r, I_e, r_e, n)
    # # add the galaxy profile around its center
    # data_with_star_positions += galaxy_profile
    

# plot original image and created model image
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(data, cmap='gray', origin='lower')
# plt.title('Original Image')
# plt.subplot(1, 2, 2)
# plt.imshow(data_with_star_positions, cmap='gray', origin='lower')
# plt.title('Model Image')
# plt.show()

for i in range(len(total_fluxes)):
    print(f"Total flux for galaxy {i + 1}: {total_fluxes[i]} +/- {total_fluxes_err[i]}")
    
print(f"percentage of fluxes that could not be fitted: {flux_summed/len(centers_list)*100}%")
    
    
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
    f.write("# x y intensity intensity error\n")
    for i, row in df.iterrows():
        f.write(f"{row['x']} {row['y']} {row['intensity']} {row['intensity error']}\n")
        
with open(cat_highintensity_file, "w") as f:
    f.write("# x y intensity\n")
    for i, row in df_largest.iterrows():
        f.write(f"{row['x']} {row['y']} {row['intensity']} {row['intensity error']}\n")

#transform from df to votable file
from astropy.table import Table
from astropy.io.votable import writeto
table= Table.from_pandas(df)
writeto(table, vot_file)
print("vot File created")
table_highintensity= Table.from_pandas(df_largest)
writeto(table_highintensity, vot_highintensity_file)