# COMMON LIBRARIES
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
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# from sklearn.cross_validation import train_test_split
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
# USER LIBRARIES
from feature_extraction import *
from settings import *
from detect_cars import find_cars
from image_process import hot_img

global svc, X_scaler

with open('model.p', 'rb') as f:
	save_dict = pickle.load(f)
svc = save_dict['svc']
X_scaler = save_dict['X_scaler']
print('The model is loaded.')


def process_image(image):

	box_list = find_cars(image, x_start_stop, y_start_stop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
	out_img = hot_img(image, box_list)
	return out_img


def process_video(input_file, output_file):
	""" Given input_file video, save annotated video to output_file """

	video = VideoFileClip(input_file)
	annotated_video = video.fl_image(process_image)
	annotated_video.write_videofile(output_file, audio=False)

process_video('test_video.mp4', 'test_video_out.mp4')