#Test Augmentations
import augmentation
import os
from deepforest import main
from deepforest import get_data
from PIL import Image
import numpy as np
from matplotlib import pyplot
import pandas as pd
    
def test_get_transform():
    csv_file = get_data("OSBS_029.csv")    
    m = main.deepforest(num_classes=1, label_dict={"Tree":0},transforms=augmentation.get_transform)
    m.config["workers"] = 0
    ds = m.load_dataset(csv_file=csv_file, root_dir=os.path.dirname(csv_file), augment=True)
    
    #Probabilistic transforms
    for x in range(10):
        step = next(iter(ds))
        
    step = next(iter(ds))
    assert len(step) == 3
    
    m.config["train"]["fast_dev_run"] = True
    m.config["train"]["csv_file"] = csv_file
    m.config["train"]["root_dir"] = os.path.dirname(csv_file)
    m.create_trainer()
    
    m.trainer.fit(m)
    
def test_ZoomSafe():
    z = augmentation.ZoomSafe(height=300, width=300)
    image = np.array(Image.open(get_data("OSBS_029.png")))
    df = pd.read_csv(get_data("OSBS_029.csv"))
    bboxes = df[["xmin", "ymin", "xmax","ymax"]].iloc[0].values.astype(float)    
    for x in range(100):
        augmented_image = z(image=image, bboxes=[bboxes])["image"]
        #pyplot.figure()
        #pyplot.imshow(augmented_image)