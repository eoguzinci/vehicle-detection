# import matplotlib.image as mpimg
# import matplotlib.pyplot as plt
# import numpy as np

# # Read in the image
# image = mpimg.imread('./test_images/cutout1.jpg')

# # Take histograms in R, G, and B
# rhist = np.histogram(image[:,:,0], bins=32, range=(0, 256))
# ghist = np.histogram(image[:,:,1], bins=32, range=(0, 256))
# bhist = np.histogram(image[:,:,2], bins=32, range=(0, 256))

# # plt.figure()
# # plt.imshow(rhist)
# # plt.show()

# # Generating bin centers
# bin_edges = rhist[1]
# bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2

# # Plot a figure with all three bar charts
# fig = plt.figure(figsize=(12,3))
# plt.subplot(131)
# plt.bar(bin_centers, rhist[0])
# plt.xlim(0, 256)
# plt.title('R Histogram')
# plt.subplot(132)
# plt.bar(bin_centers, ghist[0])
# plt.xlim(0, 256)
# plt.title('G Histogram')
# plt.subplot(133)
# plt.bar(bin_centers, bhist[0])
# plt.xlim(0, 256)
# plt.title('B Histogram')

# hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))


import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('./test_images/cutout1.jpg')

print(image.shape[1])

# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
	# Compute the histogram of the RGB channels separately
	rhist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
	ghist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
	bhist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
	# Generating bin centers
	bin_edges = rhist[1]
	bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
	# Concatenate the histograms into a single feature vector
	hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
	# Return the individual histograms, bin_centers and feature vector
	return rhist, ghist, bhist, bin_centers, hist_features
    


# def color_hist(img, nbins=32, bins_range=(0, 256)):
# 	# Compute the histogram of the RGB channels separately
# 	rhist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
# 	ghist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
# 	bhist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
# 	# Generating bin centers
# 	bin_edges = rhist[1]
# 	bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
# 	print(bin_centers)
# 	# Concatenate the histograms into a single feature vector
# 	hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
# 	# Return the individual histograms, bin_centers and feature vector
# 	return rhist, ghist, bhist, bin_centers, hist_features
    
rh, gh, bh, bincen, feature_vec = color_hist(image, nbins=32, bins_range=(0, 256))

# Plot a figure with all three bar charts
if rh is not None:
	fig = plt.figure(figsize=(12,3))
	plt.subplot(131)
	plt.bar(bincen, rh[0])
	plt.xlim(0, 256)
	plt.title('R Histogram')
	plt.subplot(132)
	plt.bar(bincen, gh[0])
	plt.xlim(0, 256)
	plt.title('G Histogram')
	plt.subplot(133)
	plt.bar(bincen, bh[0])
	plt.xlim(0, 256)
	plt.title('B Histogram')
	fig.tight_layout()
else:
  print('Your function is returning None for at least one variable...')


plt.show()