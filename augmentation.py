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
            A.RandomSizedBBoxSafeCrop(height=600, width=600, erosion_rate=0.5,p=0.25),
            A.Flip(p=0.5),
            A.pytorch.ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc',label_fields=["category_ids"]))
        
    else:
        transform = A.Compose([
            A.PadIfNeeded(min_height=600,min_width=600, border_mode=cv2.BORDER_CONSTANT, value=0),            
            A.pytorch.ToTensorV2()
        ], A.BboxParams(format='pascal_voc'))
        
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
        
        #side_height = 1 - y2
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
        
        ## h_start 
        return {"h_start": h_start, "w_start": w_start, "crop_height": self.height, "crop_width": self.width}

    def apply_to_bbox(self, bbox, crop_height=0, crop_width=0, h_start=0, w_start=0, rows=0, cols=0, **params):
        return F.bbox_random_crop(bbox, crop_height, crop_width, h_start, w_start, rows, cols)
        
    @property
    def targets_as_params(self):
        return ["image", "bboxes"]

    def get_transform_init_args_names(self):
        return ("height", "width", "interpolation")
    

class RandomBBoxSafeCrop(A.DualTransform):
    """Crop a random part of the input and rescale it to some size without loss of bboxes.
    Args:
        height (int): height after crop and resize.
        width (int): width after crop and resize.
        erosion_rate (float): erosion rate applied on input image height before crop.
        n (int): number of annotations to select, if greater than available annotations, all annotations used
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.
    Targets:
        image, mask, bboxes
    Image types:
        uint8, float32
    """

    def __init__(self, erosion_rate=0.0, n=3, interpolation=cv2.INTER_LINEAR, always_apply=False, p=1.0):
        super(RandomBBoxSafeCrop, self).__init__(always_apply, p)
        self.interpolation = interpolation
        self.erosion_rate = erosion_rate
        self.n = n

    def apply(self, img, crop_height=0, crop_width=0, h_start=0, w_start=0, interpolation=cv2.INTER_LINEAR, **params):
        crop = F.random_crop(img, crop_height, crop_width, h_start, w_start)
        return crop

    def get_params_dependent_on_targets(self, params):
        img_h, img_w = params["image"].shape[:2]
        if len(params["bboxes"]) == 0:  # less likely, this class is for use with bboxes.
            raise ValueError("No Bounding boxes")
        
        # get union of all bboxes
        if self.n > len(params["bboxes"]):
            self.n = len(params["bboxes"])
            
        index = np.random.choice(len(params["bboxes"]), self.n, replace=False)    
        
        if type(params["bboxes"]) == np.ndarray:
            selected_boxes = params["bboxes"][index,:]
        else:
            selected_boxes = [params["bboxes"][i] for i in index ]
            
        if self.n == 1:
            selected_boxes = [selected_boxes]
            
        x, y, x2, y2 = union_of_bboxes(
            width=img_w, height=img_h, bboxes=selected_boxes, erosion_rate=self.erosion_rate
        )
        # find bigger region while mantaining x,y ratio 
        min_expand = random.random()
        max_expand = random.random()
        
        bx, by = x * min_expand, y * min_expand
        bx2, by2 = x2 + (1 - x2) * max_expand , y2 + (1 - y2) * max_expand
        bw, bh = bx2 - bx, by2 - by
        crop_height = img_h if bh >= 1.0 else int(img_h * bh)
        crop_width = img_w if bw >= 1.0 else int(img_w * bw)
        h_start = np.clip(0.0 if bh >= 1.0 else by / (1.0 - bh), 0.0, 1.0)
        w_start = np.clip(0.0 if bw >= 1.0 else bx / (1.0 - bw), 0.0, 1.0)
        return {"h_start": h_start, "w_start": w_start, "crop_height": crop_height, "crop_width": crop_width}

    def apply_to_bbox(self, bbox, crop_height=0, crop_width=0, h_start=0, w_start=0, rows=0, cols=0, **params):
        return F.bbox_random_crop(bbox, crop_height, crop_width, h_start, w_start, rows, cols)

    @property
    def targets_as_params(self):
        return ["image", "bboxes"]

    def get_transform_init_args_names(self):
        return ("height", "width", "erosion_rate", "interpolation")