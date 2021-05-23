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
            ], p=1),
            #A.Flip(p=0.5),
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
        return crop

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
        
        w_lower = np.clip(x - ( crop_width- x_box_width), 0,1)
        w_upper = np.max([1-crop_width-side_width,x])
        
        #Edge case, if box touches the edge of the image, the w_start has to be exactly at img_w - self.width
        if w_lower == w_upper:
            w_start = w_lower
        else:
            w_start = np.random.random() * (w_upper-w_lower) + w_lower
        
        y_box_height = y2-y
        
        if y_box_height > crop_height:
            raise ValueError("Box height {} is larger than crop height {}".format(y_box_height, crop_height))
        
        side_height = 1 - y2
        h_lower = np.clip(y - (crop_height - y_box_height),0,1)
        h_upper = np.max([1-crop_height-side_height,y])
        
        #Same edge case as above
        if h_lower == h_upper:
            h_start = h_lower
        else:
            h_start = np.random.random() * (h_upper-h_lower) + h_lower
    
        if h_start > y:
            raise ValueError("Bad crop, h_start is {}, but y is {}".format(h_start, y))
        
        if w_start > x:
            raise ValueError("Bad crop, w_start is {}, but x is {}".format(w_start, x))
        
        if h_start + crop_height < y2:
            raise ValueError("Bad crop, crop height end is {}, but y2 is {}".format(h_start + crop_height, y2))
        
        if w_start + crop_width < x2:
            raise ValueError("Bad crop, crop width end is {}, but x2 is {}".format(w_start + crop_width, x2))
        
        return {"h_start": h_start, "w_start": w_start, "crop_height": self.height, "crop_width": self.width}

    def apply_to_bbox(self, bbox, crop_height=0, crop_width=0, h_start=0, w_start=0, rows=0, cols=0, **params):
        return F.bbox_random_crop(bbox, crop_height, crop_width, h_start, w_start, rows, cols)
        
    @property
    def targets_as_params(self):
        return ["image", "bboxes"]

    def get_transform_init_args_names(self):
        return ("height", "width", "interpolation")