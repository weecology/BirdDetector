#neill 1000 bird test, entirely new code to check results from other parts of the analysis
import comet_ml
from deepforest import main
from pytorch_lightning import loggers
import pandas as pd
import random
import tempfile

comet_logger = loggers.CometLogger(project_name="neill", workspace="bw4sz",auto_output_logging = "simple")
train = pd.read_csv("/blue/ewhite/b.weinstein/generalization/crops/neill_train.csv")
comet_logger.experiment.add_tag("neill")

train_images = train.image_path.unique()
random.shuffle(train_images)
tmpdir = tempfile.gettempdir()

sampled_annotations = []
counter = 0
while counter < 1000:
    img_name = list(train_images).pop()
    img_annotations = train[train.image_path == img_name]
    sampled_annotations.append(img_annotations)
    counter = counter + img_annotations.shape[0]
    
sampled_annotations = pd.concat(sampled_annotations)
sampled_annotations.to_csv("{}/annotations.csv".format(tmpdir))
m = main.deepforest(label_dict={"Bird":0})

m.config["train"]["csv_file"] = "{}/annotations.csv".format(tmpdir)
m.config["train"]["root_dir"] = "/blue/ewhite/b.weinstein/generalization/crops/"
    
m.config["validation"]["csv_file"] = "/blue/ewhite/b.weinstein/generalization/crops/neill_test.csv"
m.config["validation"]["root_dir"] = "/blue/ewhite/b.weinstein/generalization/crops/"
m.config["epochs"] = 10
m.create_trainer(logger=comet_logger)
m.trainer.fit(m)

results = m.evaluate(csv_file=m.config["validation"]["csv_file"], root_dir=m.config["validation"]["root_dir"], iou_threshold=0.2)
recall = results["box_recall"]
precision = results["box_precision"]

comet_logger.experiment.log_metric("Recall",recall)
comet_logger.experiment.log_metric("Precision",precision)


