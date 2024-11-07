from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label, center_of_mass
import pandas as pd

# from photutils import aperture_photometry, CircularAperture
path="fits_file/mosaic.fits"
#get threshold from other program!!!!!!!!
threshold = 3481

#open file and get data
hdulist = fits.open(path)
data = hdulist[0].data


#for now only work with part of the image with side length 1/10 of the image
side=data.shape[0]//10
data = data[:side, :side]

#create mask
binary_mask= data > threshold

# Use label to identify connected regions in the binary mask
labeled_image, num_galaxies = label(binary_mask)

print(f"Number of detected objects (galaxies): {num_galaxies}")

print(labeled_image)

#Find the center of mass for each labeled region
centroids = []  # List to store (y, x) coordinates of each centroid
for i in range(1, num_galaxies + 1):
    # Calculate the center of mass for the current object
    y_center, x_center = center_of_mass(binary_mask, labeled_image, i)
    centroids.append((y_center, x_center))
    print(f"Centroid of object {i} found")

# Display the results
# print("Detected centroids (y, x):", centroids)

#create aperature for each galaxy without using photutils
aperture_radius = 5
aperture_positions = centroids
aperatures= []
for position in aperture_positions:
    #don't use photutils to create aperatures inclluding not using CircularAperture
    y, x = position
    y, x = int(y), int(x)
    # Create a circular aperture around the centroid
    aperture = np.zeros_like(data, dtype=bool)
    y_indices, x_indices = np.indices(data.shape) # Create 2D arrays of x and y indices
    r = np.sqrt((x_indices - x)**2 + (y_indices - y)**2) # Calculate distance from the centroid
    aperture[r < aperture_radius] = True
    aperatures.append(aperture)
    print(f"aperature number {len(aperatures)} created")
    
#calculate the intensity of each galaxy as mean of the pixels in the aperature and add it to dictionary with x and y coordinates by labelling the aperatures

# Calculate the intensity of each object
intensities = []
for i, aperture in enumerate(aperatures, start=1):
    # Calculate the intensity of the current object
    intensity = np.mean(data[aperture])
    intensities.append(intensity)
    print(f"Intensity of object {i} appended")

#create dictionary with x and y coordinates and intensity
galaxies = []
for i, (y, x) in enumerate(centroids, start=1):
    galaxy = {"x": x, "y": y, "intensity": intensities[i-1]}
    galaxies.append(galaxy)

#create csv file with geader x, y and intensity and values of the galaxies
df = pd.DataFrame(galaxies)

#only pick out top 10 high intensity galaxies
df = df.nlargest(10, "intensity")

vot_file = "galaxies.vot"
#transform df to cat file
cat_file = "galaxies.cat"
with open(cat_file, "w") as f:
    f.write("# x y intensity\n")
    for galaxy in galaxies:
        df.to_csv(f, header=False, index=False, sep=" ")


#transform from df to votable file
from astropy.table import Table
from astropy.io.votable import writeto

table= Table.from_pandas(df)
writeto(table, vot_file)
print("vot File created")



#save file

#close fits file
hdulist.close()