import cv2
import numpy as np

# 이미지를 읽어온다.
image = cv2.imread('/Users/ilseo/Downloads/hint/ComfyUI_06575_.png')

# 가장자리를 검출한다.
edges = cv2.Canny(image, 100, 200)

# 가장자리를 feather 처리한다.
blurred = cv2.GaussianBlur(edges, (9, 9), 0)

cv2.imwrite('edges.jpg', edges)

cv2.imwrite('blurred.jpg', blurred)

print(image.shape)
print(blurred.shape)

#expand blurred image to 3 channels
blurred = cv2.merge([blurred, blurred, blurred])

# 원본 이미지에 feather 처리된 가장자리를 추가한다.
result = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

# 결과 이미지를 저장한다.
cv2.imwrite('output_image.jpg', result)