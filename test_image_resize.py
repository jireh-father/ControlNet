import cv2
image_path = "D:\dataset\hair_style\hairshop_sample_from_gisu/KakaoTalk_20240316_213657048.jpg"
source = cv2.imread("path")

# resize to 512x512 keeping the aspect ratio with padding aligned to the center
h, w, _ = source.shape
if h > w:
    pad = (h - w) // 2
    source = cv2.copyMakeBorder(source, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
else:
    pad = (w - h) // 2
    source = cv2.copyMakeBorder(source, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

print(source.shape)
source = cv2.resize(source, (512, 512))
print(source.shape)