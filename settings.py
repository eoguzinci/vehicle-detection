# HYPER-PARAMETERS
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
spatial_feat = False # Spatial features on or off
spatial_size = (32, 32) # Spatial binning dimensions
hist_feat = False # Histogram features on or off
hist_bins = 32    # Number of histogram bins
hog_feat = True # HOG features on or off
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
x_start_stop = [0,1280] # Vehicles in the active range
y_start_stop = [[400, 650],
								[400,560],
								[400,530],
								[400,480]] # Min and max in y to search in slide_window()
scales = [3.0,2.0, 1.5, 1.0] # Scales for each y window
heat_threshold = 4 # The heat map threshold to eliminate false positives (detections)