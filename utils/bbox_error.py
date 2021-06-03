from albumentations import functional as F
bbox = (0.129, 0.7846, 0.1626, 0.818)
cropped_bbox = F.bbox_random_crop(bbox=bbox, crop_height=100, crop_width=100, h_start=0.7846, w_start=0.12933, rows=1500, cols=1500)
cropped_bbox

assert bbox[3] < (100/1500) + 0.7846
assert all([(y >= 0) & (y<=1) for y in list(cropped_bbox)])

####
from albumentations import functional as F
bbox = (0.129, 0.7846, 0.1626, 0.818)
cropped_bbox = F.bbox_random_crop(bbox=bbox, crop_height=100, crop_width=100, h_start=0.7846 - (100/1500), w_start=0.12933 - (100/1500), rows=1500, cols=1500)
cropped_bbox

assert bbox[3] < (100/1500) + 0.7846
assert all([(y >= 0) & (y<=1) for y in list(cropped_bbox)])


###


bbox = (0, 0, 0.1,0.1)
crop_height = 200
rows = 1000
h_start = 0
w_start = 0
cropped_bbox = F.bbox_random_crop(bbox=bbox, crop_height=crop_height, crop_width=crop_height, h_start=h_start, w_start=w_start, rows=rows, cols=rows)
cropped_bbox

assert cropped_bbox[3] == bbox[3] / ((crop_height/rows))
assert all([(y >= 0) & (y<=1) for y in list(cropped_bbox)])

bbox = (0.1, 0.1, 0.2,0.2)
crop_height = 200
rows = 1000
h_start = 0
w_start = 0
cropped_bbox = F.bbox_random_crop(bbox=bbox, crop_height=crop_height, crop_width=crop_height, h_start=h_start, w_start=w_start, rows=rows, cols=rows)
cropped_bbox

assert cropped_bbox[3] == bbox[3] / ((crop_height/rows))
assert all([(y >= 0) & (y<=1) for y in list(cropped_bbox)])

bbox = (0.1, 0.1, 0.2,0.2)
crop_height = 200
rows = 1000
h_start = 0.1
w_start = 0.1
cropped_bbox = F.bbox_random_crop(bbox=bbox, crop_height=crop_height, crop_width=crop_height, h_start=h_start, w_start=w_start, rows=rows, cols=rows)
cropped_bbox


F.get_random_crop_coords(height=rows, width=rows, crop_height=crop_height, crop_width=crop_height, h_start=h_start, w_start=w_start)

