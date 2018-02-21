import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
import pickle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from lesson_functions import *
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# from sklearn.cross_validation import train_test_split
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
from settings import *

# dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )
# svc = dist_pickle["svc"]
# X_scaler = dist_pickle["scaler"]
# orient = dist_pickle["orient"]
# pix_per_cell = dist_pickle["pix_per_cell"]
# cell_per_block = dist_pickle["cell_per_block"]
# spatial_size = dist_pickle["spatial_size"]
# hist_bins = dist_pickle["hist_bins"]

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
scales = [3.0,2.0, 1.5, 1.0]
heat_threshold = 4
    
# Read in cars and notcars
images = glob.glob('./dataset/*.jpeg')
cars = []
notcars = []
for image in images:
    if 'image' in image or 'extra' in image:
        notcars.append(image)
    else:
        cars.append(image)

print(len(cars),len(notcars))

# Reduce the sample size because
# The quiz evaluator times out after 13s of CPU time
sample_size = 500
cars = cars[0:sample_size]
notcars = notcars[0:sample_size]

car_img = mpimg.imread(cars[7])
car_img2 = mpimg.imread(cars[9])
_, car_hog = get_hog_features(car_img[:,:,2], orient, pix_per_cell, cell_per_block, vis=True, feature_vec=True)
notcar_img = mpimg.imread(notcars[7])
_, notcar_hog = get_hog_features(notcar_img[:,:,2], orient, pix_per_cell, cell_per_block, vis=True, feature_vec=True)

# Visualize Car and NonCar
f, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(7,7))
f.subplots_adjust(hspace = .4, wspace=.2)
ax1.imshow(car_img)
ax1.set_title('Car Image', fontsize=16)
ax2.imshow(notcar_img)
ax2.set_title('Non-Car Image', fontsize=16)

# # Visualize HOG
# f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7,7))
# f.subplots_adjust(hspace = .4, wspace=.2)
# ax1.imshow(car_img)
# ax1.set_title('Car Image', fontsize=16)
# ax2.imshow(car_hog, cmap='gray')
# ax2.set_title('Car HOG', fontsize=16)
# ax3.imshow(notcar_img)
# ax3.set_title('Non-Car Image', fontsize=16)
# ax4.imshow(notcar_hog, cmap='gray')
# ax4.set_title('Non-Car HOG', fontsize=16)

# # Visualize color histogram
# f, ((ax1, ax2), (ax3, ax4), (ax5, ax6),(ax7, ax8)) = plt.subplots(4, 2, figsize=(7,7))
# f.subplots_adjust(hspace = .4, wspace=.2)
# ax1.imshow(car_img)
# ax1.set_title('Car Image 1', fontsize=16)
# ax2.imshow(car_img2)
# ax2.set_title('Car Image 2', fontsize=16)
# ax3.hist(car_img[:,:,0], bins=hist_bins, range=(0, 256), facecolor='r')
# ax3.set_title('Car1 Color RED', fontsize=16)
# ax4.hist(car_img[:,:,0], bins=hist_bins, range=(0, 256), facecolor='r')
# ax4.set_title('Car2 Color RED', fontsize=16)
# ax5.hist(car_img[:,:,1], bins=hist_bins, range=(0, 256), facecolor='g')
# ax5.set_title('Car1 Color GREEN', fontsize=16)
# ax6.hist(car_img2[:,:,1], bins=hist_bins, range=(0, 256), facecolor='g')
# ax6.set_title('Car2 Color GREEN', fontsize=16)
# ax7.hist(car_img[:,:,2], bins=hist_bins, range=(0, 256), facecolor='b')
# ax7.set_title('Car1 Color BLUE', fontsize=16)
# ax8.hist(car_img2[:,:,2], bins=hist_bins, range=(0, 256), facecolor='b')
# ax8.set_title('Car2 Color BLUE', fontsize=16)

# plt.figure()
# car_features = extract_features(cars, color_space=color_space, 
#                         spatial_size=spatial_size, hist_bins=hist_bins, 
#                         orient=orient, pix_per_cell=pix_per_cell, 
#                         cell_per_block=cell_per_block, 
#                         hog_channel=hog_channel, spatial_feat=spatial_feat, 
#                         hist_feat=hist_feat, hog_feat=hog_feat)
# notcar_features = extract_features(notcars, color_space=color_space, 
#                         spatial_size=spatial_size, hist_bins=hist_bins, 
#                         orient=orient, pix_per_cell=pix_per_cell, 
#                         cell_per_block=cell_per_block, 
#                         hog_channel=hog_channel, spatial_feat=spatial_feat, 
#                         hist_feat=hist_feat, hog_feat=hog_feat)

# # Create an array stack of feature vectors
# X = np.vstack((car_features, notcar_features)).astype(np.float64)

# # Define the labels vector
# y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# # Split up data into randomized training and test sets
# rand_state = np.random.randint(0, 100)
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=rand_state)
    
# # Fit a per-column scaler
# X_scaler = StandardScaler().fit(X_train)
# # Apply the scaler to X
# X_train = X_scaler.transform(X_train)
# X_test = X_scaler.transform(X_test)

# print('Using:',orient,'orientations',pix_per_cell,
#     'pixels per cell and', cell_per_block,'cells per block')
# print('Feature vector length:', len(X_train[0]))
# # Use a linear SVC 
# svc = LinearSVC()
# # Check the training time for the SVC
# t=time.time()
# svc.fit(X_train, y_train)
# t2 = time.time()
# print(round(t2-t, 2), 'Seconds to train SVC...')

# # Check the score of the SVC
# print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# # Check the prediction time for a single sample
# t=time.time()

# with open('model.p', 'rb') as f:
#     save_dict = pickle.load(f)
# svc = save_dict['svc']
# X_scaler = save_dict['X_scaler']
# print('The model is loaded.')

heat_map = []
hot_map = []
boxed = []
final_box = []

for i in range(6):
    image = mpimg.imread('./test_images/test'+str(i+1)+'.jpg')
    draw_img = np.copy(image)
    draw_img2 = np.copy(image)
    plt.figure()
    plt.imshow(image)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    # plt.figure()
    # plt.imshow(image[y_start_stop[0][0]:y_start_stop[0][1],:,:])

    colors = [(0,0,255), (0,255,255),(0,255,0),(255,0,0)] 
    box_list = []
    window = 64
    for j in range(len(scales)):
        # cv2.rectangle(draw_img, (x_start_stop[0],y_start_stop[j][0]), (x_start_stop[1],y_start_stop[j][1]), colors[j], 4)
        # cv2.rectangle(draw_img, (x_start_stop[0],y_start_stop[j][0]), (x_start_stop[0]+np.int(window*scales[j]),y_start_stop[j][0]+np.int(window*scales[j])), colors[j], 4)
        boxes= find_cars(image, x_start_stop, y_start_stop[j], scales[j], svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        box_list.extend(boxes)
    # plt.figure()
    # plt.imshow(draw_img)

    for j in range(len(box_list)):
        cv2.rectangle(draw_img2,box_list[j][0],box_list[j][1],(0,0,255),6)
    plt.figure()
    plt.imshow(draw_img2)

    # Heat Map
    heat = np.zeros_like(image[:,:,0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat,box_list)

    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(draw_img2)
    plt.title('Car Positions')
    plt.subplot(122)
    plt.imshow(heat, cmap='hot')
    plt.title('No Thres. Heat Map')
    fig.tight_layout()

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,heat_threshold)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)

    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(draw_img)
    plt.title('Car Positions')
    plt.subplot(122)
    plt.imshow(heatmap, cmap='hot')
    plt.title('Heat Map')
    fig.tight_layout()

plt.show()