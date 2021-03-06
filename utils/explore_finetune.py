#Explore finetuning hyperparameters
import comet_ml
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.plugins import DDPPlugin
from deepforest import main
import pandas as pd
import random
import glob
import tempfile
import torch
from deepforest.model import create_model

tempdir = tempfile.gettempdir()

def select(df, n):
    selected_annotations = []
    count = 0
    available = list(df.image_path.unique())
    random.shuffle(available)
    while count < n:
        try:
            selected_image = available.pop()
        except:
            break
        new_annotations = df[df.image_path==selected_image]
        selected_annotations.append(new_annotations)
        count += new_annotations.shape[0]
    train_annotations = pd.concat(selected_annotations)
    
    return train_annotations

comet_logger = CometLogger(project_name="everglades", workspace="bw4sz",auto_output_logging = "simple")
comet_logger.experiment.add_tag("fine tune")

train_df = pd.read_csv("/blue/ewhite/b.weinstein/AerialDetection/data/trainval1024/train.csv")
label_dict = {x: index for index, x in enumerate(train_df.label.unique())}    
pretrained_DOTA = main.deepforest(num_classes=15, label_dict=label_dict)
pretrained_DOTA.load_state_dict(torch.load("/orange/ewhite/b.weinstein/AerialDetection/snapshots/20210530_233702/DOTA.pl")["state_dict"])
        
#update backbone weights with new Retinanet head
model = main.deepforest(label_dict={"Bird":0})          
model.model = create_model(num_classes=1, nms_thresh=model.config["nms_thresh"], score_thresh=model.config["score_thresh"], backbone=pretrained_DOTA.model.backbone)

df = pd.read_csv("/blue/ewhite/b.weinstein/generalization/crops/hayes_train.csv")
selected_df = select(df, 1000)
selected_df.to_csv("/blue/ewhite/b.weinstein/generalization/crops/finetune_example.csv")

model.config["train"]["csv_file"] = "/blue/ewhite/b.weinstein/generalization/crops/finetune_example.csv"
model.config["validation"]["csv_file"] = "/blue/ewhite/b.weinstein/generalization/crops/hayes_test.csv"
model.config["validation"]["root_dir"] = "/blue/ewhite/b.weinstein/generalization/crops/"
model.config["train"]["root_dir"] = "/blue/ewhite/b.weinstein/generalization/crops/"

model.create_trainer(logger=comet_logger, plugins=DDPPlugin(find_unused_parameters=False))
test_results = model.evaluate(csv_file="/blue/ewhite/b.weinstein/generalization/crops/hayes_test.csv", root_dir="/blue/ewhite/b.weinstein/generalization/crops/", iou_threshold=0.25, savedir=tempdir)

print("Original Recall is {}".format(test_results["box_recall"]))
print("Original Precision is {}".format(test_results["box_precision"]))


model.trainer.fit(model)

test_results = model.evaluate(csv_file="/blue/ewhite/b.weinstein/generalization/crops/hayes_test.csv",
                              root_dir="/blue/ewhite/b.weinstein/generalization/crops/", iou_threshold=0.25, savedir=tempdir)

print("Fine tune Recall is {}".format(test_results["box_recall"]))
print("Fine tune Precision is {}".format(test_results["box_precision"]))

for x in glob.glob("{}/*.png".format(tempdir)):
    comet_logger.experiment.log_image(x, image_scale=0.25)
