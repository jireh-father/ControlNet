import cv2

# 이미지 불러오기
image = cv2.imread('/Users/ilseo/Downloads/hint/ComfyUI_06575_.png')

# Gaussian 블러 적용
blurred = cv2.GaussianBlur(image, (9, 9), 0)

# 이미지 페더링 (가장자리 부드럽게 만들기)
feathered = cv2.addWeighted(image, 0.6, blurred, 0.4, 0)

# 결과 이미지 출력
cv2.imwrite('output_image.jpg', feathered)
