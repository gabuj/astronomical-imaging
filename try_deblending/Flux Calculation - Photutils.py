from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import detect_threshold, detect_sources, deblend_sources, SourceCatalog
from photutils.aperture import CircularAperture
import numpy as np
import os
import subprocess
import matplotlib.pyplot as plt

# Existing name variables (retain as needed)
name1 = "1"  # Example placeholder
name2 = "2_minimal_blending"
name3 = "2_image_edge"
name4 = "2_close_noblend"
name5 = "2_similar"
name6 = "1_extended_diffues"
name7 = "3_cluster_60"
name8 = "4_cluster_60"  # Newly added

# File path using name1 (adjust as needed)
file_path = f'/Users/yuri/Desktop/Year 3 Lab/Astronomical Image Processing/Git repository/astronomical-imaging/fake_files/{name1}.fits'

# Load FITS file
with fits.open(file_path) as hdul:
    image_data = hdul[0].data.copy()

# Background estimation using sigma clipping
mean, median, std = sigma_clipped_stats(image_data, sigma=3.0, maxiters=5)

# Source detection
threshold = detect_threshold(image_data, nsigma=5.0)  # Updated parameter from snr to nsigma
segm = detect_sources(image_data, threshold, npixels=5)
if segm is None:
    print("No sources detected.")
    exit()

# Deblending sources
segm_deblend = deblend_sources(image_data, segm, npixels=5, nlevels=32, contrast=0.001)

# Create SourceCatalog
catalog = SourceCatalog(image_data, segm_deblend)

# Annotate image
output_image = image_data.copy()
for source in catalog:
    y, x = source.centroid
    aperture = CircularAperture((x, y), r=10)
    aperture.plot(color='white', lw=1.5, alpha=0.5)  # White circles
    output_image[int(y), int(x)] = image_data.min()    # Black centers

# Save annotated FITS file
output_directory = '/Users/yuri/Desktop/Year 3 Lab/Astronomical Image Processing/Git repository/astronomical-imaging/fake_files'
output_path = os.path.join(output_directory, 'output.fits')
hdu = fits.PrimaryHDU(output_image)
hdu.writeto(output_path, overwrite=True)

# Open the output FITS file
# For macOS: 'open', Windows: 'start', Linux: 'xdg-open'
subprocess.run(['open', output_path])  # Change 'open' to 'start' if on Windows