#Explore finetuning hyperparameters
from deepforest import main
from utils import prepare
model = main.deepforest.load_from_checkpoint("/orange/ewhite/b.weinstein/generalization/20210616_101934/mckellar_USGS_hayes_terns_penguins_pfeifer_palmyra_everglades.pl")
path_dict = prepare.prepare()
model.config["train"]["csv_file"] = path_dict["monash"]["train"]
model.config["validation"]["csv_file"] = path_dict["monash"]["test"]
model.config["validation"]["root_dir"] = "/orange/ewhite/b.weinstein/generalization/crops/"
model.config["train"]["root_dir"] = "/orange/ewhite/b.weinstein/generalization/crops/"

test_results = model.evaluate(csv_file=path_dict["monash"]["test"], root_dir="/orange/ewhite/b.weinstein/generalization/crops/", iou_threshold=0.25)

print("Zero shot Recall is {}".format(test_results["box_recall"]))
print("Zero shot Precision is {}".format(test_results["box_precision"]))


model.trainer.fit(model)

test_results = model.evaluate(csv_file=path_dict["monash"]["test"], root_dir="/orange/ewhite/b.weinstein/generalization/crops/", iou_threshold=0.25)

print("Fine tune Recall is {}".format(test_results["box_recall"]))
print("Fine tune Precision is {}".format(test_results["box_precision"]))
