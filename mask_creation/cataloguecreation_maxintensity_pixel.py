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
# side=data.shape[0]//10
# data = data[:side, :side]


# Visualize the result
plt.figure(figsize=(10, 5))
#plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(data, cmap='gray')
plt.colorbar()
plt.show()

#get rid of bleeding regions
#define bleeding region
bleeding_centers= [(3217,1427), (2281,905),(2773,974),(3315,776)] #list of (y, x) coordinates of the centers of the bleeding regions

# Create a mask for the bleeding regions
mask = np.zeros_like(data, dtype=bool)
#create binary mask to see when over the threshold
binary_mask= data > threshold
# Create a mask for the bleeding regions
bleeding_region = np.zeros_like(data, dtype=bool)

#create labels and take aaway all pixels that are true in the binary mask that are part of the same label
labels, num_labels = label(binary_mask)
for centers in bleeding_centers:
    #find label they belong to
    y, x = centers
    y, x = int(y), int(x)
    star_label = labels[y, x]
    print(f"Star label: {star_label}")
    #take away all pixels that are part of the same label
    bleeding_region[labels==star_label]=True


galaxy_mask=np.copy(data)
galaxy_mask[bleeding_region==True]=0

#galaxy_mask is data without the bleeding regions
#show bleeding regions
plt.figure(figsize=(10, 5))
plt.title("Bleeding Region")
plt.imshow(bleeding_region, cmap='gray')
plt.colorbar()
plt.show()








#Find the center of mass for each labeled region
centroids = []  # List to store (y, x) coordinates of each centroid

    

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
df_largest = df.nlargest(10, "intensity")

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