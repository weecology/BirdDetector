#terns 1000 bird test, entirely new code to check results from other parts of the analysis
import comet_ml
from pytorch_lightning import loggers
import pandas as pd
from generalization import *
from model import BirdDetector
import tempfile

dataset = "monash"
tmpdir = tempfile.gettempdir()

comet_logger = loggers.CometLogger(project_name="everglades", workspace="bw4sz",auto_output_logging = "simple")
train = pd.read_csv("/blue/ewhite/b.weinstein/generalization/crops/{}_train.csv".format(dataset))
comet_logger.experiment.add_tag(dataset)

df = pd.read_csv("/blue/ewhite/b.weinstein/generalization/crops/{}_train.csv".format(dataset))  
n = df.shape[0]
n=5000
train_annotations = select(df, n=n)
model = BirdDetector(transforms = deepforest_transform)   
#model.config["validation"]["csv_file"] = "/blue/ewhite/b.weinstein/generalization/crops/{}_test.csv".format(dataset)
#model.config["validation"]["root_dir"] = "/blue/ewhite/b.weinstein/generalization/crops/"
model.config["train"]["epochs"] = 100
model.config["lr"] = 0.2
model = fit(model, df, comet_logger,"{}_random_{}".format(dataset, n))
finetune_results = model.evaluate(csv_file="/blue/ewhite/b.weinstein/generalization/crops/{}_test.csv".format(dataset), root_dir="/blue/ewhite/b.weinstein/generalization/crops/", iou_threshold=0.2)
comet_logger.experiment.log_metric("Box Recall {} {}".format(n,dataset),finetune_results["box_recall"])
comet_logger.experiment.log_metric("Box Precision {} {}".format(n,dataset),finetune_results["box_precision"])



