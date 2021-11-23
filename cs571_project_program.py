import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM

img = cv2.imread("flower_pic.jpg")
plt.imshow(img)
# Convert MxNx3 image into Kx3 where K=MxN
img2 = img.reshape((-1, 3))  #-1 reshape means, in this case MxN


#for K Mean cluster
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=42)
model = kmeans.fit(img2)
predicted_values = kmeans.predict(img2)

#res = center[label.flatten()]
segm_image = predicted_values.reshape((img.shape[0], img.shape[1]))
segm_image = np.expand_dims(segm_image, axis=-1)
plt.imshow(segm_image)


#for GMM cluster
#covariance choices, full, tied, diag, spherical
gmm_model = GMM(n_components=2, covariance_type='full').fit(img2)  #tied works better than full
gmm_labels = gmm_model.predict(img2)

#Put numbers back to original shape so we can reconstruct segmented image
original_shape = img.shape
segmented = gmm_labels.reshape(original_shape[0], original_shape[1])
segmented = np.expand_dims(segmented, axis=-1)
plt.imshow(segmented)

#BIC is used to calculate the no.of componenets. 
# =============================================================================
# n_components = np.arange(1,10)
# gmm_models = [GMM(n, covariance_type='tied').fit(img2) for n in n_components]
# plt.plot(n_components, [m.bic(img2) for m in gmm_models], label='BIC')
# 
# =============================================================================

#for foreground and background segmentation using K-mean
foreground = np.multiply(segm_image, img)
background = img - foreground
plt.imshow(foreground) 
plt.imshow(background)

#for foreground and background segmentation using GMM
foreground1 = np.multiply(segmented, img)
background1 = img - foreground1
plt.imshow(foreground1) 
plt.imshow(background1)