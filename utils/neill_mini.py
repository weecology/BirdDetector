#terns 1000 bird test, entirely new code to check results from other parts of the analysis
import comet_ml
from deepforest import main
from pytorch_lightning import loggers
import pandas as pd
import random
import tempfile
from pytorch_lightning.plugins import DDPPlugin
from deepforest.dataset import get_transform as deepforest_transform
from ..model import BirdDetector

comet_logger = loggers.CometLogger(project_name="everglades", workspace="bw4sz",auto_output_logging = "simple")
train = pd.read_csv("/blue/ewhite/b.weinstein/generalization/crops/terns_train.csv")
comet_logger.experiment.add_tag("terns")

train_images = train.image_path.unique()
random.shuffle(train_images)
tmpdir = tempfile.gettempdir()

sampled_annotations = []
counter = 0
while counter < 20000:
    print(counter)
    img_name = list(train_images).pop()
    img_annotations = train[train.image_path == img_name]
    sampled_annotations.append(img_annotations)
    counter = counter + img_annotations.shape[0]
    
sampled_annotations = pd.concat(sampled_annotations)
sampled_annotations.to_csv("{}/annotations.csv".format(tmpdir))
model = BirdDetector(transforms = deepforest_transform)                   


m.config["train"]["csv_file"] = "{}/annotations.csv".format(tmpdir)
m.config["train"]["root_dir"] = "/blue/ewhite/b.weinstein/generalization/crops/"
    
m.config["validation"]["csv_file"] = "/blue/ewhite/b.weinstein/generalization/crops/terns_test.csv"
m.config["validation"]["root_dir"] = "/blue/ewhite/b.weinstein/generalization/crops/"
m.config["epochs"] = 20
m.config["train"]["lr"] = 0.002 

m.create_trainer(logger=comet_logger, plugins=DDPPlugin(find_unused_parameters=True))
m.trainer.fit(m)

results = m.evaluate(csv_file=m.config["validation"]["csv_file"], root_dir=m.config["validation"]["root_dir"], iou_threshold=0.2)
recall = results["box_recall"]
precision = results["box_precision"]

comet_logger.experiment.log_metric("Recall",recall)
comet_logger.experiment.log_metric("Precision",precision)


