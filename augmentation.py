#Transform augmentations
import albumentations as A
from albumentations import functional as F
from albumentations.augmentations.crops.functional import random_crop, bbox_random_crop
from albumentations import resize
from albumentations.augmentations.bbox_utils import union_of_bboxes
import random
import numpy as np
import cv2

## general style
#def get_transform(augment):
    #"""Albumentations transformation of bounding boxs"""
    #if augment:
        #transform = A.Compose([
            #A.HorizontalFlip(p=0.5),
            #ToTensorV2()
        #], bbox_params=A.BboxParams(format='pascal_voc',label_fields=["category_ids"]))
        
    #else:
        #transform = A.Compose([ToTensorV2()])
        
    #return transform

def get_transform(augment):
    """Custom albumentations transformation of bounding boxs"""
    if augment:
        transform = A.Compose([
            A.PadIfNeeded(1000,1000, border_mode=cv2.BORDER_CONSTANT, value=0),
            RandomSizedBBoxSafeCrop(height=1000,width=1000,erosion_rate=1,p=0.75),
            A.Flip(p=0.5),
            A.RandomBrightnessContrast(),
            A.pytorch.ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc',label_fields=["category_ids"]))
        
    else:
        transform = A.Compose([
            A.pytorch.ToTensorV2()
        ])
        
    return transform

class MeanSubtract(A.ImageOnlyTransform):
    def __init__(self, always_apply=True, p=0.5):
        super().__init__(always_apply, p)
    
    def apply(self, image, **params):
        return image - image.mean()
    

class RandomSizedBBoxSafeCrop(A.DualTransform):
    """Crop a random part of the input and rescale it to some size without loss of bboxes.
    Args:
        height (int): height after crop and resize.
        width (int): width after crop and resize.
        erosion_rate (float): erosion rate applied on input image height before crop.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.
    Targets:
        image, mask, bboxes
    Image types:
        uint8, float32
    """

    def __init__(self, height, width, erosion_rate=0.0, interpolation=cv2.INTER_LINEAR, always_apply=False, p=1.0):
        super(RandomSizedBBoxSafeCrop, self).__init__(always_apply, p)
        self.height = height
        self.width = width
        self.interpolation = interpolation
        self.erosion_rate = erosion_rate

    def apply(self, img, crop_height=0, crop_width=0, h_start=0, w_start=0, interpolation=cv2.INTER_LINEAR, **params):
        crop = random_crop(img, crop_height, crop_width, h_start, w_start)
        return resize(crop, self.height, self.width, interpolation)

    def get_params_dependent_on_targets(self, params):
        img_h, img_w = params["image"].shape[:2]
        if len(params["bboxes"]) == 0:  # less likely, this class is for use with bboxes.
            erosive_h = int(img_h * (1.0 - self.erosion_rate))
            crop_height = img_h if erosive_h >= img_h else random.randint(erosive_h, img_h)
            return {
                "h_start": random.random(),
                "w_start": random.random(),
                "crop_height": crop_height,
                "crop_width": int(crop_height * img_w / img_h),
            }
        # get union of all bboxes
        #random sample of proportion of rows
        if len(params["bboxes"]) > 1:
            n_boxes = len(params["bboxes"])
            selected_index = np.random.choice(range(n_boxes), size=1)
            params["bboxes"]  = [params["bboxes"][x] for x in selected_index]
        
        x, y, x2, y2 = union_of_bboxes(
            width=img_w, height=img_h, bboxes=params["bboxes"], erosion_rate=self.erosion_rate
        )
        # find bigger region
        left_expand = random.random()
        right_expand = random.random()
        
        bx, by = x * left_expand,  y * left_expand
        bx2, by2 = x2 + (1 - x2) * right_expand, y2 + (1 - y2) * right_expand
        bw, bh = bx2 - bx, by2 - by
        crop_height = img_h if bh >= 1.0 else int(img_h * bh)
        crop_width = img_w if bw >= 1.0 else int(img_w * bw)
        h_start = np.clip(0.0 if bh >= 1.0 else by / (1.0 - bh), 0.0, 1.0)
        w_start = np.clip(0.0 if bw >= 1.0 else bx / (1.0 - bw), 0.0, 1.0)
        return {"h_start": h_start, "w_start": w_start, "crop_height": crop_height, "crop_width": crop_width}

    def apply_to_bbox(self, bbox, crop_height=0, crop_width=0, h_start=0, w_start=0, rows=0, cols=0, **params):
        return bbox_random_crop(bbox, crop_height, crop_width, h_start, w_start, rows, cols)

    @property
    def targets_as_params(self):
        return ["image", "bboxes"]

    def get_transform_init_args_names(self):
        return ("height", "width", "erosion_rate", "interpolation")
