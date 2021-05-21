#Transform augmentations
import albumentations as A
from albumentations import functional as F
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

#TODO Make the crop size probabilistic. 

def get_transform(augment):
    """Albumentations transformation of bounding boxs"""
    if augment:
        transform = A.Compose([
            A.PadIfNeeded(min_height=600,min_width=600, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.OneOf([
            ZoomSafe(height=200,width=200),                
            ZoomSafe(height=300,width=300),
            ZoomSafe(height=400,width=400),
            ZoomSafe(height=500,width=500),
            ], p=0.75),
            A.Flip(p=0.5),
            A.pytorch.ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc',label_fields=["category_ids"]))
        
    else:
        transform = A.Compose([A.pytorch.ToTensorV2()])
        
    return transform

class ZoomSafe(A.DualTransform):
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

    def __init__(self, height, width, interpolation=cv2.INTER_LINEAR, always_apply=False, p=1.0):
        super(ZoomSafe, self).__init__(always_apply, p)
        self.height = height
        self.width = width
        self.interpolation = interpolation
        
        #Sanity check
        if self.height > height:
            raise ValueError("Requested crop height {} is greater than image height {}".format(self.height, height))

        if self.width > width:
            raise ValueError("Requested crop width {} is greater than image width {}".format(self.width, width))

    def apply(self, img, crop_height=0, crop_width=0, h_start=0, w_start=0, interpolation=cv2.INTER_LINEAR, **params):
        crop = F.random_crop(img, crop_height, crop_width, h_start, w_start)
        return F.resize(crop, self.height, self.width, interpolation)

    def get_params_dependent_on_targets(self, params):
        img_h, img_w = params["image"].shape[:2]
        
        crop_width = self.width/img_w
        crop_height = self.height/img_h
        
        if len(params["bboxes"]) == 0:
            raise ValueError("No bounding boxes found for image")

        # get union of selected bboxes (single box)
        index = np.random.choice(len(params["bboxes"]), 1, replace=False)[0]
        if type(params["bboxes"]) == np.ndarray:
            selected_box = params["bboxes"][index,:]
        else:
            selected_box = params["bboxes"][index]            
        
        x, y, x2, y2 = selected_box[:4]
        
        # Create a box around the x, y
        x_box_width = x2-x
        side_width = 1 - x2
        if x_box_width > crop_width:
            raise ValueError("Box width {} is larger than crop width {}".format(x_box_width, crop_width))
        
        w_lower = np.max([x - ( crop_width- x_box_width), 0])
        w_upper = np.max([1-crop_width-side_width,x])
        
        #Edge case, if box touches the edge of the image, the w_start has to be exactly at img_w - self.width
        if w_lower >= w_upper:
            w_start = w_lower
        else:
            w_start = np.random.random() * (w_upper-w_lower) + w_lower
        
        y_box_height = y2-y
        
        if y_box_height > crop_height:
            raise ValueError("Box height {} is larger than crop height {}".format(y_box_height, crop_height))
        
        side_height = 1 - y2
        h_lower = np.max([y - (crop_height - y_box_height), 0])
        h_upper = np.max([1-crop_height-side_height,y])
        
        #Same edge case as above
        if h_lower >= h_upper:
            h_start = h_lower
        else:
            h_start = np.random.random() * (h_upper-h_lower) + h_lower
    
        return {"h_start": h_start, "w_start": w_start, "crop_height": self.height, "crop_width": self.width}

    def apply_to_bbox(self, bbox, crop_height=0, crop_width=0, h_start=0, w_start=0, rows=0, cols=0, **params):
        augmented_boxes = F.bbox_random_crop(bbox, crop_height, crop_width, h_start, w_start, rows, cols)
        if len(augmented_boxes) == 0:
            raise ValueError("Blank annotations created from input boxes {}, with crop height {}, crop width {}, h_start {}, w_start, rows {}, cols {}".format(
            bbox, crop_height, crop_width, h_start, w_start, rows, cols))
        else:
            return augmented_boxes

    @property
    def targets_as_params(self):
        return ["image", "bboxes"]

    def get_transform_init_args_names(self):
        return ("height", "width", "erosion_rate", "interpolation")