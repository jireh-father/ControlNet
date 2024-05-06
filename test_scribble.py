
import cv2
import torch

from annotator.util import resize_image, HWC3
from annotator.hed import HEDdetector, nms
from annotator.canny import CannyDetector
from annotator.mlsd import MLSDdetector
from annotator.hed import HEDdetector
apply_hed = HEDdetector()
apply_canny = CannyDetector()
apply_mlsd = MLSDdetector()

# model = create_model('./models/cldm_v15.yaml').cpu()
# model.load_state_dict(load_state_dict('./models/control_sd15_scribble.pth', location='cuda'))
# model = model.cuda()
# ddim_sampler = DDIMSampler(model)

input_path = "D:\dataset\hair_style\hairshop_sample_from_gisu/random_280_detailer_00001_.jpg"
input_image = cv2.imread(input_path)#, cv2.IMREAD_COLOR)
# cv2.imshow("input_image", input_image)
# cv2.waitKey(0)
image_resolution = 512
detect_resolution = 512
low_threshold = 100
high_threshold = 200
pre_type = 'scribble'
value_threshold = 0.5
distance_threshold = 0.5
increase_contrast = False
contrast_alpha = 1.7

with torch.no_grad():
    input_image = HWC3(input_image)
    resize_image(input_image, detect_resolution)

    if increase_contrast:
        input_image = cv2.convertScaleAbs(input_image, alpha=contrast_alpha, beta=0)
        # cv2.imshow("input_image", input_image)
        # cv2.waitKey(0)

    if pre_type == 'scribble':
        detected_map = apply_hed(input_image)
        detected_map = HWC3(detected_map)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        detected_map = nms(detected_map, 127, 3.0)
        detected_map = cv2.GaussianBlur(detected_map, (0, 0), 3.0)
        detected_map[detected_map > 4] = 255
        detected_map[detected_map < 255] = 0
    elif pre_type == 'canny':
        img = input_image
        #control contrast
        H, W, C = img.shape

        detected_map = apply_canny(img, low_threshold, high_threshold)
        detected_map = HWC3(detected_map)
    elif pre_type == 'mlsd':
        detected_map = apply_mlsd(input_image, value_threshold, distance_threshold)
        detected_map = HWC3(detected_map)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
    elif pre_type == 'hed':
        detected_map = apply_hed(input_image)
        detected_map = HWC3(detected_map)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
    print(detected_map.shape)
    #show_image(detected_map)
    cv2.imshow("detected_map", detected_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()