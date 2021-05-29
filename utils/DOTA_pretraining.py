#Pretrain on DOTA benchmark
#https://captain-whu.github.io/DOTA/dataset.html
import comet_ml
import cv2
from pytorch_lightning.loggers import CometLogger
import torch
import os
import pandas as pd
import glob
from deepforest import main
from deepforest import visualize
from deepforest import preprocess
import tempfile
import traceback
from datetime import datetime
import albumentations as A
import numpy as np

def prepare():
    files = glob.glob("/orange/ewhite/b.weinstein/DOTA/train/labelTxt-v1.0/*.txt")
    train_data = []
    for x in files:
        df = pd.read_csv(x,skiprows=[0,1],names=["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4", "category", "difficult"],sep=" ")
        df["image_path"] = "{}.png".format(os.path.splitext(os.path.basename(x))[0])
        df["xmin"] = df[["x1","x2","x3","x4"]].apply(lambda x: x.min(),axis=1)
        df["xmax"] = df[["x1","x2","x3","x4"]].apply(lambda x: x.max()-2,axis=1)
        df["ymin"] = df[["y1","y2","y3","y4"]].apply(lambda y: y.min(),axis=1)
        df["ymax"] = df[["y1","y2","y3","y4"]].apply(lambda y: y.max()-2,axis=1)
        df = df[["image_path","xmin","ymin","xmax","ymax","category"]].rename(columns={"category":"label"})
        fname = "/orange/ewhite/b.weinstein/DOTA/val/labels/{}".format(os.path.basename(x))
        df.to_csv(fname)
        
        split_labels = preprocess.split_raster(
            annotations_file=fname,
            patch_size=1024,
            allow_empty=False,
         base_dir="/orange/ewhite/b.weinstein/DOTA/val/images/images/")
        
        train_data.append(split_labels)
    
    train_df = pd.concat(train_data)
    train_df.to_csv("/orange/ewhite/b.weinstein/DOTA/train/train.csv")
    
    files = glob.glob("/orange/ewhite/b.weinstein/DOTA/test/labelTxt-v1.0/*.txt")
    test_data = []
    for x in files:
        df = pd.read_csv(x,skiprows=[0,1],names=["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4", "category", "difficult"],sep=" ")
        df["image_path"] = "{}.png".format(os.path.splitext(os.path.basename(x))[0])
        df["xmin"] = df[["x1","x2","x3","x4"]].apply(lambda x: x.min(),axis=1)
        df["xmax"] = df[["x1","x2","x3","x4"]].apply(lambda x: x.max()-2,axis=1)
        df["ymin"] = df[["y1","y2","y3","y4"]].apply(lambda y: y.min(),axis=1)
        df["ymax"] = df[["y1","y2","y3","y4"]].apply(lambda y: y.max()-2,axis=1)
        df = df[["image_path","xmin","ymin","xmax","ymax","category"]].rename(columns={"category":"label"})
        fname = "/orange/ewhite/b.weinstein/DOTA/val/labels/{}".format(os.path.basename(x))
        df.to_csv(fname)
        split_labels = preprocess.split_raster(
            annotations_file=fname,
            patch_size=1024,
            allow_empty=False,
            base_dir="/orange/ewhite/b.weinstein/DOTA/val/images/images/")
        
        test_data.append(split_labels)
    
    test_df = pd.concat(test_data)
    test_df.to_csv("/orange/ewhite/b.weinstein/DOTA/val/test.csv")

    train_df, test_df
    
def get_transform(augment):
    """Albumentations transformation of bounding boxs"""
    if augment:
        transform = A.Compose([
            A.Flip(p=0.5),
            A.RandomBrightnessContrast(),
            A.pytorch.ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc',label_fields=["category_ids"]))
        
    else:
        transform = A.Compose([
            A.pytorch.ToTensorV2()
        ])
        
    return transform

comet_logger = CometLogger(api_key="ypQZhYfs3nSyKzOfz13iuJpj2",
                              project_name="everglades", workspace="bw4sz")
comet_logger.experiment.add_tag("DOTA")
save_dir = "/orange/ewhite/b.weinstein/DOTA/snapshots"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_savedir = "{}/{}".format(save_dir,timestamp)  

train_df, test_df = prepare()

label_dict = {x: index for index, x in enumerate(train_df.label.unique())}
n_classes = len(train_df.label.unique())
m  = main.deepforest(num_classes= n_classes, label_dict=label_dict)

m.config["train"]["csv_file"] = "/orange/ewhite/b.weinstein/DOTA/train/train.csv"
m.config["train"]["root_dir"] = "/orange/ewhite/b.weinstein/DOTA/train/images/images/"
m.config["validation"]["csv_file"] = "/orange/ewhite/b.weinstein/DOTA/val/test.csv"
m.config["validation"]["root_dir"] = "/orange/ewhite/b.weinstein/DOTA/validation/images/images/"

#view traning
ds = m.load_dataset(csv_file=m.config["train"]["csv_file"], root_dir=m.config["train"]["root_dir"], shuffle=True, augment=True)
for i in np.arange(10):
    batch = next(iter(ds))
    image_path, image, targets = batch
    df = visualize.format_boxes(targets[0], scores=False)
    image = np.moveaxis(image[0].numpy(),0,2)[:,:,::-1] * 255
    image = visualize.plot_predictions(image, df)
    with tempfile.TemporaryDirectory() as tmpdirname:
        cv2.imwrite("{}/{}".format(tmpdirname, image_path[0]),image )
        comet_logger.experiment.log_image("{}/{}".format(tmpdirname, image_path[0]),image_scale=0.25)   
        
m.create_trainer()
m.trainer.fit(m)

results = m.evaluate(csv_file=m.config["validation"]["csv_file"], root_dir=m.config["validation"]["root_dir"], savedir=model_savedir)

if comet_logger is not None:
    try:
        results["results"].to_csv("{}/iou_dataframe.csv".format(model_savedir))
        comet_logger.experiment.log_asset("{}/iou_dataframe.csv".format(model_savedir))
        
        results["class_recall"].to_csv("{}/class_recall.csv".format(model_savedir))
        comet_logger.experiment.log_asset("{}/class_recall.csv".format(model_savedir))
        
        for index, row in results["class_recall"].iterrows():
            comet_logger.experiment.log_metric("{}_Recall".format(row["label"]),row["recall"])
            comet_logger.experiment.log_metric("{}_Precision".format(row["label"]),row["precision"])
        
        comet_logger.experiment.log_metric("Average Class Recall",results["class_recall"].recall.mean())
        comet_logger.experiment.log_metric("Box Recall",results["box_recall"])
        comet_logger.experiment.log_metric("Box Precision",results["box_precision"])
        
        comet_logger.experiment.log_parameter("saved_checkpoint","{}/species_model.pl".format(model_savedir))
        
        ypred = results["results"].predicted_label.astype('category').cat.codes.to_numpy()            
        ypred = torch.from_numpy(ypred)
        ypred = torch.nn.functional.one_hot(ypred, num_classes = m.num_classes).numpy()
        
        ytrue = results["results"].true_label.astype('category').cat.codes.to_numpy()
        ytrue = torch.from_numpy(ytrue)
        ytrue = torch.nn.functional.one_hot(ytrue, num_classes = m.num_classes).numpy()
        comet_logger.experiment.log_confusion_matrix(y_true=ytrue, y_predicted=ypred, labels = list(m.label_dict.keys()))
    except Exception as e:
        print("logger exception: {} with traceback \n {}".format(e, traceback.print_exc()))
        

    #log images
    with comet_logger.experiment.context_manager("validation"):
        images = glob.glob("{}/*.png".format(model_savedir))
        for img in images:
            comet_logger.experiment.log_image(img, image_scale=0.25)    