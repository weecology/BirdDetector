#Everglades 1000 bird test, entirely new code to check results from other parts of the analysis
import comet_ml
from deepforest import main
from pytorch_lightning import loggers
import pandas as pd
import random
import tempfile
import glob
import cv2

comet_logger = loggers.CometLogger(project_name="everglades", workspace="bw4sz")
train = pd.read_csv("/blue/ewhite/b.weinstein/generalization/crops/everglades_train.csv")
comet_logger.experiment.add_tag("Everglades")

train_images = list(train.image_path.unique())
random.shuffle(train_images)
tmpdir = tempfile.gettempdir()

sampled_annotations = []
counter = 0

while counter < 5000:
    img_name = train_images.pop()
    img_annotations = train[train.image_path == img_name]
    sampled_annotations.append(img_annotations)
    counter = counter + img_annotations.shape[0]

sampled_annotations = pd.concat(sampled_annotations)

comet_logger.experiment.log_parameter("training images",len(sampled_annotations.image_path.unique()))
comet_logger.experiment.log_parameter("training annotations",sampled_annotations.shape[0])

sampled_annotations.to_csv("{}/annotations.csv".format(tmpdir))
m = main.deepforest(label_dict={"Bird":0})
#m.use_release()
m.config["train"]["csv_file"] = "{}/annotations.csv".format(tmpdir)
m.config["train"]["root_dir"] = "/blue/ewhite/b.weinstein/generalization/crops/"
    
#m.config["validation"]["csv_file"] = "/blue/ewhite/b.weinstein/generalization/crops/everglades_test.csv"
#m.config["validation"]["root_dir"] = "/blue/ewhite/b.weinstein/generalization/crops/"
m.config["train"]["epochs"] = 30
m.create_trainer(logger=comet_logger)
m.trainer.fit(m)

results = m.evaluate(csv_file="/blue/ewhite/b.weinstein/generalization/crops/everglades_test.csv",
                     root_dir="/blue/ewhite/b.weinstein/generalization/crops/", iou_threshold=0.2)

recall = results["box_recall"]
precision = results["box_precision"]

comet_logger.experiment.log_metric("Recall",recall)
comet_logger.experiment.log_metric("Precision",precision)

results = m.evaluate(csv_file=m.config["train"]["csv_file"], root_dir=m.config["train"]["root_dir"], iou_threshold=0.2)
recall = results["box_recall"]
precision = results["box_precision"]

comet_logger.experiment.log_metric("Train Recall",recall)
comet_logger.experiment.log_metric("Train Precision",precision)

for x in train_images[:10]:
    img = m.predict_image(path = "{}/{}".format(m.config["train"]["root_dir"],x), return_plot=True)
    cv2.imwrite("{}/{}".format(tmpdir,x), img)
    comet_logger.experiment.log_image("{}/{}".format(tmpdir,x), image_scale=0.25)
