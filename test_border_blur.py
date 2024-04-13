import cv2
import numpy as np

# Load your image
image = cv2.imread('D:\dataset\hair_style\hairshop_sample_from_gisu/520.jpg')

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform edge detection
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
cv2.imshow('Edges', edges)
cv2.waitKey(0)

# Perform dilation to make edges thicker
kernel = np.ones((5,5), np.uint8)
dilated_edges = cv2.dilate(edges, kernel, iterations=1)

# Apply Gaussian blur to the dilated edges
blurred_edges = cv2.GaussianBlur(dilated_edges, (15, 15), 0)

# Create a mask by thresholding the blurred edges
_, mask = cv2.threshold(blurred_edges, 10, 255, cv2.THRESH_BINARY)

# Invert the mask
mask_inv = cv2.bitwise_not(mask)

# Apply the mask to the original image
blurred_image = cv2.bitwise_and(image, image, mask=mask_inv)

# Display the result
cv2.imshow('Blurred Border', blurred_image)
cv2.waitKey(0)
cv2.destroyAllWindows()