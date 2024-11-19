from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import creating_fake_image
import finding_center_radius
import pandas as pd
import  takeout_bleeding
import background_estimation
from astropy.io import fits
import bad_data_clean
#gete data
max_localbackground_radius=200
fraction_bin=max_localbackground_radius*2
background_gain=3 #how many times more than radius is local background radius



#open file
path='fits_file/mosaic.fits'
hdulist = fits.open(path)

data = hdulist[0].data


#show initial image
plt.imshow(data, cmap='gray', origin='lower')
plt.colorbar()
plt.title('Initial Image')
plt.show()


#parameters for the background estimation
fraction_bin_totalbackground=1.1 #num bins is data shape/fraction_bin
sigmas_thershold = 5#how many sigmas of std after background is the threshold
#find background
background_value,background_std=background_estimation.finding_background(data, fraction_bin_totalbackground, sigmas_thershold) #problem!!
background_thershold=background_value+sigmas_thershold*background_std

#close file
hdulist.close()


#bleeding centerss
bleeding_centers= [(3217,1427), (2281,905),(2773,974),(3315,776),(5,1430)] #list of (y, x) coordinates of the centers of the bleeding regions
#take away bleeing
data=takeout_bleeding.takeou_bleeing(data,bleeding_centers,background_thershold,background_value)

#still have to do: take out bad data
maxx=data.shape[1]
maxy=data.shape[0]
baddata_coords=[[0,0,33,430],[0,0,124,119],[0,0,105,408],[0,2462,126,maxx],[0,0,408,99],[0,0,430,26],[0,0,4518,4],[4516,0,maxy,120],[4504,2161,maxy,maxx],[0,2467,maxy,maxx]] #top left and top right corner of region (y1,x1,y2,x2)
data=bad_data_clean.takeout_baddata(data,baddata_coords,background_value)


#show cleaned image
#show image in zscale
plt.imshow(data, cmap='gray', origin='lower')
plt.colorbar()
plt.title('Cleaned Image')
plt.show()


#use only part of the data
size=300
data=data[0:size,0:size]


#finding radius paramters
overexposed_threshold=65535
#create the data
# data=creating_fake_image.create_fake_image(image_size, centers, galaxy_peaks, sigmas,background_value,noise_level,ns)



original_data=np.copy(data)

#paramters for finding centers and radius
#set max radius being max distance from center to  edge of image    IMRPVOE THIS
max_radius=int(np.sqrt(data.shape[0]**2+data.shape[1]**2)/4)
backgroundfraction_tolerance=0.9



#final files parameters
vot_file = 'test_onrealdata/galaxy_catalog.vot'
vot_highintensity_file = "test_onrealdata/highestintensity_galaxies.vot"
#transform df to cat file
cat_file = 'test_onrealdata/galaxy_catalog.cat'
cat_highintensity_file = "test_onrealdata/highestintensity_galaxies.cat"   




centers_list,radii_list=finding_center_radius.finding_centers_radii(data,overexposed_threshold,background_value,background_std,background_gain)
x, y = np.indices(data.shape)



# Define the Sérsic profile
def sersic_profile(r, I_e, r_e, n):
    b_n = 1.9992 * n - 0.3271  # Approximate value for b_n
    return I_e * np.exp(-b_n * ((r / r_e)**(1/n) - 1))

# Create radial distances and intensities for the galaxy
def radial_profile(data, max_radius, r):
    # Step 3: Calculate the radial intensity profile by averaging pixel values at each radius
    radial_distances=np.arange(1,max_radius+1)
    radial_intensity = np.array([data[r == rad].mean() if np.any(r == r) else 0 for rad in radial_distances])
    print(f"radial distances are {radial_distances}")
    print(f"radial intensities are {radial_intensity}")
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
    return I_e, r_e, n, I_e_err, r_e_err, n_err

def flux_within_radius(I_e, r_e, n, I_e_err, r_e_err, n_err):
    # Calculate the total flux within the threshold radius
    b_n = 1.9992 * n - 0.3271
    total_flux = 2 * np.pi * I_e * r_e**2 * n * np.exp(b_n) / (b_n**(2 * n))  # Sérsic total flux formula
    total_flux_err = total_flux * np.sqrt((I_e_err / I_e)**2 + (2 * r_e_err / r_e)**2 + (n_err / n)**2)
    return total_flux, total_flux_err


def otherway_flux_within_radius(data, max_radius, r):
    radii, intensities = radial_profile(data, max_radius, r)
    total_flux = np.sum(intensities)
    total_flux_err = np.sqrt(np.sum(intensities)) #not correct but don't know how to do it
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
data_with_star_positions += background_value

total_fluxes = []
total_fluxes_err = []
data=original_data
flux_summed=0
for i, center in enumerate(centers_list):
    
    y_center, x_center = center
    radius = radii_list[i]
    #convert to integer for binning
    r = np.sqrt((x - x_center)**2 + (y - y_center)**2)
    r = r.astype(int)
    
     #take away local background value from intensity
    local_background,local_background_err=take_away_localbackground(data,radius,r,background_gain)
    temporary_data=np.copy(data)
    #make loacl background same kind of array as data
    local_background=np.full(data.shape,local_background)
    temporary_data-=local_background
   
    try:
        I_e, r_e, n, I_e_err, r_e_err, n_err = fit_sersic(temporary_data, radius, r)
        flux, flux_err = flux_within_radius(I_e, r_e, n, I_e_err, r_e_err, n_err)
        flux=flux-local_background
        
        total_fluxes.append(flux)
        total_fluxes_err.append(flux_err)
    except RuntimeError:
        print(f"Could not fit Sérsic profile for galaxy {i + 1}")
        #fit in another way
        flux,flux_err=otherway_flux_within_radius(temporary_data, radius, r)
        total_fluxes.append(flux)
        total_fluxes_err.append(flux_err)
        flux_summed+=1
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