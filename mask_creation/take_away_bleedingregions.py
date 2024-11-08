from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label, center_of_mass
import pandas as pd
def bleed(data,centers):
    