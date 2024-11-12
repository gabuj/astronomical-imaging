import numpy as np

def takeout_baddata(data,baddata_coords):
    #get rid of bad data
    #define bad data
    bad_data = np.zeros_like(data, dtype=bool)
    #take out region of data within the 4 corners given in baddata_coords
    for coords in baddata_coords:
        y1, x1, y2, x2 = coords
        y1, x1, y2, x2 = int(y1), int(x1), int(y2), int(x2)
        bad_data[y1:y2, x1:x2] = True
    #create mask for bad data
    good_data=np.copy(data)
    good_data[bad_data==True]=0
    return good_data