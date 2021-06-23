#Explore finetuning hyperparameters
import comet_ml
from pytorch_lightning.loggers import CometLogger
from deepforest import main
import pandas as pd
import random

def select(df):
    selected_annotations = []
    count = 0
    available = list(df.image_path.unique())
    random.shuffle(available)
    while count < 1000:
        try:
            selected_image = available.pop()
        except:
            break
        new_annotations = df[df.image_path==selected_image]
        selected_annotations.append(new_annotations)
        count += new_annotations.shape[0]
    train_annotations = pd.concat(selected_annotations)
    
    return train_annotations

comet_logger = CometLogger(api_key="ypQZhYfs3nSyKzOfz13iuJpj2",
                            project_name="everglades", workspace="bw4sz",auto_output_logging = "simple")

comet_logger.experiment.add_tag("fine tune")
model = main.deepforest.load_from_checkpoint("/orange/ewhite/b.weinstein/generalization/20210616_101934/mckellar_USGS_hayes_terns_penguins_pfeifer_palmyra_everglades.pl")
model.label_dict = {"Bird": 0}
model.create_trainer(logger=comet_logger)

df = pd.read_csv("/orange/ewhite/b.weinstein/generalization/crops/seabirdwatch_train.csv")
selected_df = select(df)
selected_df.to_csv("/orange/ewhite/b.weinstein/generalization/crops/finetune_example.csv")

model.config["train"]["csv_file"] = "/orange/ewhite/b.weinstein/generalization/crops/finetune_example.csv"
model.config["validation"]["csv_file"] = "/orange/ewhite/b.weinstein/generalization/crops/seabirdwatch_test.csv"
model.config["validation"]["root_dir"] = "/orange/ewhite/b.weinstein/generalization/crops/"
model.config["train"]["root_dir"] = "/orange/ewhite/b.weinstein/generalization/crops/"

test_results = model.evaluate(csv_file="/orange/ewhite/b.weinstein/generalization/crops/seabirdwatch_test.csv", root_dir="/orange/ewhite/b.weinstein/generalization/crops/", iou_threshold=0.25)

print("Zero shot Recall is {}".format(test_results["box_recall"]))
print("Zero shot Precision is {}".format(test_results["box_precision"]))


model.trainer.fit(model)

test_results = model.evaluate(csv_file="/orange/ewhite/b.weinstein/generalization/crops/seabirdwatch_test.csv",
                              root_dir="/orange/ewhite/b.weinstein/generalization/crops/", iou_threshold=0.25)

print("Fine tune Recall is {}".format(test_results["box_recall"]))
print("Fine tune Precision is {}".format(test_results["box_precision"]))
