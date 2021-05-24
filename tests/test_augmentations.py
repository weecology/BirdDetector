#Test Augmentations
import augmentation
import os
import model
from deepforest import main
from deepforest import get_data
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
    
def test_get_transform():
    ROOT = os.path.dirname(model.__file__)    
    csv_file = "{}/data/46544951.csv".format(ROOT)    
    m = model.BirdDetector(transforms=augmentation.get_transform)
    m.config["workers"] = 0
    ds = m.load_dataset(csv_file=csv_file, root_dir=os.path.dirname(csv_file), augment=True)
    
    #Probabilistic transforms
    for x in ds:
        path, image, boxes = x
        
    step = next(iter(ds))
    assert len(step) == 3
    
    m.config["train"]["fast_dev_run"] = True
    m.config["train"]["csv_file"] = csv_file
    m.config["train"]["root_dir"] = os.path.dirname(csv_file)
    m.create_trainer()
    
    m.trainer.fit(m)

def test_RandomBBoxSafeCrop():
    z = augmentation.RandomBBoxSafeCrop()
    image = np.array(Image.open("/Users/benweinstein/Downloads/46544951.png"))
    df = pd.read_csv("/Users/benweinstein/Downloads/everglades_train.csv")
    df = df[df.image_path == '46544951.png']
    bboxes = df[["xmin", "ymin", "xmax","ymax"]].values.astype(float)    
    fig = plt.figure()
    for x in range(25):
        augmented = z(image=image, bboxes=bboxes/image.shape[0])
        augmented_image = augmented["image"]
        augmented_image.shape[0] == augmented_image.shape[1] 
        passes = False
        ax = fig.add_subplot(5,5, x+1)
        ax.imshow(augmented_image)
        #Atleast one box passes
        for x in augmented["bboxes"]:
            if all([(y >= 0) & (y<=1) for y in list(x)]):
                passes = True
            
        assert passes
    
    #plt.show()
    print("end")
        
def test_ZoomSafe():
    z = augmentation.ZoomSafe(height=100, width=100)
    image = np.array(Image.open("/Users/benweinstein/Downloads/46544951.png"))
    df = pd.read_csv("/Users/benweinstein/Downloads/everglades_train.csv")
    df = df[df.image_path == '46544951.png']
    bboxes = df[["xmin", "ymin", "xmax","ymax"]].values.astype(float)    
    for x in range(100):
        print(x)
        augmented = z(image=image, bboxes=bboxes/image.shape[0])
        augmented_image = augmented["image"]
        assert augmented_image.shape[0] == 100
        passes = False
        
        #Atleast one box passes
        for x in augmented["bboxes"]:
            if all([(y >= 0) & (y<=1) for y in list(x)]):
                passes = True
            
        assert passes
        
        #pyplot.figure()
        #pyplot.imshow(augmented_image)
        
def test_ZoomSafe_dataset(tmpdir):
    m = main.deepforest(num_classes=1, label_dict={"Bird":0},transforms=augmentation.get_transform)
    
    df = pd.read_csv("/Users/benweinstein/Downloads/everglades_train.csv")
    df = df[df.image_path == '46544951.png']
    df.to_csv("{}/example.csv".format(tmpdir))
    
    m.config["workers"] = 0
    ds = m.load_dataset(csv_file="{}/example.csv".format(tmpdir), root_dir="/Users/benweinstein/Downloads/", augment=True)
    
    for x in range(10):    
        print(x)
        for x in ds:
            path, image, targets = x
            print(targets[0]["boxes"])
            assert not len(targets[0]["boxes"]) == 0