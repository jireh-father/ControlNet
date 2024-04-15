import cv2
import numpy as np

# read image with alpha channel
# ori_image = cv2.imread('clipspace-mask-4773943.399999976.png')
ori_image = cv2.imread("ComfyUI_06694_.png")
# face_image = cv2.imread('/Users/ilseo/Downloads/hint/ComfyUI_06575_.png', cv2.IMREAD_UNCHANGED)

# mask_image = cv2.imread("ComfyUI_temp_pvouq_00005_.png")
mask_image = cv2.imread("ComfyUI_temp_pvouq_00011_.png")
print(mask_image.shape)
print(ori_image.shape)
face_image = ori_image.copy()
print((mask_image == 0).shape)
# face_image[mask_image == 0] = 0

# add alpha channel to image
face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2BGRA)
blurred_img = cv2.GaussianBlur(mask_image, (21, 21), 0)
mask = np.zeros(mask_image.shape, np.uint8)

gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)[1]
contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(mask, contours, -1, (255,255,255),5)
output = np.where(mask == np.array([255, 255, 255]), blurred_img, mask_image)

# invert output
output = cv2.bitwise_not(output)
output_gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
print(output_gray.shape)
# cv2.imshow('image', image)
# cv2.imshow('output', output)
# cv2.waitKey(0)

# convert output image to grayscale and 1 channel
# alpha = cv2.cvtColor(output, cv2.COLOR_BGRA2GRAY)
# print(output.shape)
# from PIL import Image
# face_image_pil = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGRA2RGBA))
# alpha_pil = Image.fromarray(alpha)
#
# face_image_pil.putalpha(alpha_pil)
face_image[:,:,3] = output_gray


# cv2.imwrite('output_image.png', output)
cv2.imwrite('output_image_invert.png', face_image)
# face_image_pil.save('output_image_with_alpha.png')