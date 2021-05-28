#Pretrain on DOTA benchmark
#https://captain-whu.github.io/DOTA/dataset.html
import comet_ml
from pytorch_lightning.loggers import CometLogger
import torch
import os
import pandas as pd
import glob

from deepforest import main
from datetime import datetime

comet_logger = CometLogger(api_key="ypQZhYfs3nSyKzOfz13iuJpj2",
                              project_name="everglades", workspace="bw4sz")

comet_logger.experiment.add_tag("DOTA")

save_dir = "/orange/ewhite/b.weinstein/DOTA/snapshots"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_savedir = "{}/{}".format(save_dir,timestamp)  

files = glob.glob("/orange/ewhite/b.weinstein/DOTA/train/labelTxt-v1.5/*.txt")

train_data = []
for x in files:
    df = pd.read_csv(x,skiprows=[0,1],names=["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4", "category", "difficult"],sep=" ")
    df["image_path"] = "{}.png".format(os.path.splitext(os.path.basename(x))[0])
    train_data.append(df)

train_df = pd.concat(train_data)
train_df.to_csv("/orange/ewhite/b.weinstein/DOTA/train/train.csv")

files = glob.glob("/orange/ewhite/b.weinstein/DOTA/validation/labelTxt-v1.5/*.txt")
test_data = []
for x in files:
    df = pd.read_csv(x,skiprows=[0,1],names=["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4", "category", "difficult"],sep=" ")
    df["image_path"] = "{}.png".format(os.path.splitext(os.path.basename(x))[0])
    test_data.append(df)

test_df = pd.concat(test_data)
test_df.to_csv("/orange/ewhite/b.weinstein/DOTA/validation/validation.csv")

label_dict = {x: index for x, index in enumerate(train_df.category.unique())}
n_classes = len(train_df.category.unique())
m  = main.deepforest(num_classes= n_classes, label_dict=label_dict)

m.config["train"]["csv_file"] = "/orange/ewhite/b.weinstein/DOTA/train/train.csv"
m.config["train"]["root_dir"] = "/orange/ewhite/b.weinstein/DOTA/train/images/images/"
m.config["validation"]["csv_file"] = "/orange/ewhite/b.weinstein/DOTA/validation/validation.csv"
m.config["validation"]["root_dir"] = "/orange/ewhite/b.weinstein/DOTA/validation/images/images/"

m.config
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
        images = glob.glob("{}/*.png".format(savedir))
        for img in images:
            comet_logger.experiment.log_image(img, image_scale=0.25)    