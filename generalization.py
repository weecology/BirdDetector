#Prepare all training sets
import comet_ml
from model import BirdDetector
from augmentation import get_transform
import glob
from PIL import ImageFile
from pytorch_lightning.loggers import CometLogger
from deepforest.model import create_model
from deepforest import main
from deepforest.dataset import get_transform as deepforest_transform
from datetime import datetime
from deepforest import visualize

from utils.PR import precision_recall_curve
from utils.preprocess import *
from utils.prepare import *

import os
import numpy as np
import cv2
import random
from model import BirdDetector
import pandas as pd
import torch
import gc
from pytorch_lightning.plugins import DDPPlugin
import subprocess

def view_training(paths,comet_logger, n=10):
    """For each site, grab three images and view annotations
    Args:
        n: number of images to load
    """
    m = BirdDetector(transforms=get_transform)
    
    with comet_logger.experiment.context_manager("view_training"):
        for site in paths:
            for split in ["train","test"]:
                if split == "train":
                    augment = True
                else:
                    augment = False
                try:
                    x = paths[site][split]
                    ds = m.load_dataset(csv_file=x, root_dir=os.path.dirname(x), shuffle=True, augment=augment)
                    for i in np.arange(10):
                        batch = next(iter(ds))
                        image_path, image, targets = batch
                        df = visualize.format_boxes(targets[0], scores=False)
                        image = np.moveaxis(image[0].numpy(),0,2)[:,:,::-1] * 255
                        image = visualize.plot_predictions(image, df)
                        with tempfile.TemporaryDirectory() as tmpdirname:
                            cv2.imwrite("{}/{}".format(tmpdirname, image_path[0]),image )
                            comet_logger.experiment.log_image("{}/{}".format(tmpdirname, image_path[0]),image_scale=0.25)                
                except Exception as e:
                    print(e)
                    continue
                
def train(path_dict, config, train_sets = ["penguins","terns","everglades","palmyra"],test_sets=["everglades"], comet_logger=None, save_dir=None):
        
    comet_logger.experiment.log_parameter("timestamp",timestamp)
    comet_logger.experiment.log_parameter("train_set",train_sets)
    comet_logger.experiment.log_parameter("test_set",test_sets)
    comet_logger.experiment.add_tag("Generalization")
    
    all_sets = []
    print("Train sets: {}".format(train_sets))
    for x in train_sets:
        
        try:
            df = pd.read_csv(path_dict[x]["train"])
            df_test = pd.read_csv(path_dict[x]["test"])
            
        except:
            raise ValueError("No training path supplied for {}".format(x))
        all_sets.append(df)
        all_sets.append(df_test)
    
    train_annotations = pd.concat(all_sets)
    
    #A couple illegal boxes, make slightly smaller
    train_annotations["xmin"] = train_annotations["xmin"].astype(float) 
    train_annotations["xmax"] = train_annotations["xmax"].astype(float) - 3
    train_annotations["ymin"] = train_annotations["ymin"].astype(float)
    train_annotations["ymax"] = train_annotations["ymax"].astype(float) - 3
    
    train_annotations = train_annotations[~(train_annotations.xmin >= train_annotations.xmax)]
    train_annotations = train_annotations[~(train_annotations.ymin >= train_annotations.ymax)]
    
    train_annotations.to_csv("/orange/ewhite/b.weinstein/generalization/crops/training_annotations.csv")

    all_val_sets = []
    for x in test_sets:
        df = pd.read_csv(path_dict[x]["test"])
        all_val_sets.append(df)
    
    test_annotations = pd.concat(all_val_sets)
    
    test_annotations.to_csv("/orange/ewhite/b.weinstein/generalization/crops/test_annotations.csv")

    comet_logger.experiment.log_parameter("training_images",len(train_annotations.image_path.unique()))
    comet_logger.experiment.log_parameter("training_annotations",train_annotations.shape[0])
    
    train_df =  pd.read_csv("/orange/ewhite/b.weinstein/AerialDetection/data/trainval1024/train.csv")
    label_dict = {x: index for index, x in enumerate(train_df.label.unique())}    
    pretrained_DOTA = main.deepforest(num_classes=15, label_dict=label_dict)
    
    model = BirdDetector(transforms = get_transform)
    
    #update backbone weights with new Retinanet head
    model.model = create_model(num_classes=1, nms_thresh=model.config["nms_thresh"], score_thresh=model.config["score_thresh"], backbone=pretrained_DOTA.model.backbone)
    
    model.config = config

    model.config["train"]["csv_file"] = "/orange/ewhite/b.weinstein/generalization/crops/training_annotations.csv"
    model.config["train"]["root_dir"] = "/orange/ewhite/b.weinstein/generalization/crops"    
    model.config["validation"]["csv_file"] = "/orange/ewhite/b.weinstein/generalization/crops/test_annotations.csv"
    model.config["validation"]["root_dir"] = "/orange/ewhite/b.weinstein/generalization/crops"
        
    model.create_trainer(logger=comet_logger, plugins=DDPPlugin(find_unused_parameters=False), num_sanity_val_steps=0)
    comet_logger.experiment.log_parameters(model.config)
    
    model.trainer.fit(model)
    
    for x in test_sets:
        test_results = model.evaluate(csv_file=path_dict[x]["test"], root_dir="/orange/ewhite/b.weinstein/generalization/crops/", iou_threshold=0.25, savedir=savedir)
        if comet_logger is not None:
            try:
                test_results["results"].to_csv("{}/{}_iou_dataframe.csv".format(savedir, x))
                comet_logger.experiment.log_asset("{}/{}_iou_dataframe.csv".format(savedir, x))
                comet_logger.experiment.log_metric("{} Box Recall".format(x),test_results["box_recall"])
                comet_logger.experiment.log_metric("{} Box Precision".format(x),test_results["box_precision"])
            except Exception as e:
                print(e)    

    if save_dir:
        model_path = "{}/{}.pt".format(save_dir,"_".join(train_sets))
        torch.save(model.model.state_dict(),model_path)
    
    #Fine tuning, up to 100 birds from train
    #fine_tune = pd.read_csv("/orange/ewhite/b.weinstein/generalization/crops/{}_train.csv".format(test_sets[0]))
    #selected_annotations = []
    #count = 0
    #available = fine_tune.image_path.unique()
    #random.shuffle(available)
    #while count < 1000:
        #try:
            #selected_image = available.pop()
        #except:
            #break
        #new_annotations = fine_tune[fine_tune.image_path==selected_image]
        #selected_annotations.append(new_annotations)
        #count += new_annotations.shape[0]
    #selected_annotations = pd.concat(selected_annotations)
    #selected_annotations.to_csv("/orange/ewhite/b.weinstein/generalization/crops/{}_finetune.csv".format(test_sets[0]))
    
    #A couple illegal boxes, make slightly smaller
    train_annotations = pd.read_csv("/orange/ewhite/b.weinstein/generalization/crops/{}_train.csv".format(test_sets[0]))
    train_annotations["xmin"] = train_annotations["xmin"].astype(float) 
    train_annotations["xmax"] = train_annotations["xmax"].astype(float)
    train_annotations["ymin"] = train_annotations["ymin"].astype(float)
    train_annotations["ymax"] = train_annotations["ymax"].astype(float)
    
    train_annotations = train_annotations[~(train_annotations.xmin >= train_annotations.xmax)]
    train_annotations = train_annotations[~(train_annotations.ymin >= train_annotations.ymax)]
    
    train_annotations.to_csv("/orange/ewhite/b.weinstein/generalization/crops/training_annotations.csv")    

    model.transforms = deepforest_transform
    model.config["train"]["csv_file"] = "/orange/ewhite/b.weinstein/generalization/crops/training_annotations.csv"
    model.config["validation"]["csv_file"] = "/orange/ewhite/b.weinstein/generalization/crops/test_annotations.csv"
    model.config["train"]["root_dir"] = "/orange/ewhite/b.weinstein/generalization/crops/"
    model.config["validation"]["root_dir"] = "/orange/ewhite/b.weinstein/generalization/crops/"
    model.create_trainer(logger=comet_logger, num_sanity_val_steps=0)
    model.trainer.fit(model)
    
    finetune_results = model.evaluate(csv_file="/orange/ewhite/b.weinstein/generalization/crops/{}_test.csv".format(test_sets[0]), root_dir="/orange/ewhite/b.weinstein/generalization/crops/", iou_threshold=0.25)
    if comet_logger is not None:
        comet_logger.experiment.log_metric("Fine Tuned {} Box Recall".format(x),finetune_results["box_recall"])
        comet_logger.experiment.log_metric("Fine Tuned {} Box Precision".format(x),finetune_results["box_precision"])
        
    
    #The last position in the loop is the LOO score
    return test_results["box_recall"], test_results["box_precision"]

if __name__ =="__main__":
    #save original config during loop
    #comet_logger=None
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir="/orange/ewhite/b.weinstein/generalization"
    savedir = "{}/{}".format(save_dir,timestamp)  
    
    try:
        os.mkdir(savedir)
    except Exception as e:
        print(e)
        
    model = BirdDetector(transforms=get_transform)
    config = model.config
    
    path_dict = prepare()
    comet_logger = CometLogger(api_key="ypQZhYfs3nSyKzOfz13iuJpj2",
                                project_name="everglades", workspace="bw4sz",auto_output_logging = "simple")
    
    #Log commit
    comet_logger.experiment.log_parameter("commit hash",subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip())
    view_training(path_dict, comet_logger=comet_logger)

    ###leave one out
    train_list = ["neill","USGS","hayes","terns","penguins","pfeifer","palmyra","mckellar","monash"]
    results = []
    for x in train_list:
        train_sets = [y for y in train_list if not y==x]
        train_sets.append("everglades")
        #train_sets.append("murres")
        test_sets = [x]
        recall, precision = train(path_dict=path_dict, config=config, train_sets=train_sets, test_sets=test_sets, comet_logger=comet_logger, save_dir=savedir)
        result = pd.DataFrame({"test_sets":[x],"recall":[recall],"precision":[precision]})
        results.append(result)
        torch.cuda.empty_cache()
        gc.collect()        
    
    results = pd.concat(results)
    results.to_csv("Figures/generalization.csv")
    comet_logger.experiment.log_asset(file_data="Figures/generalization.csv", file_name="results.csv")
    comet_logger.experiment.log_metric(name="Mean LOO Recall", value=results.recall.mean())
    comet_logger.experiment.log_metric(name="Mean LOO Precision", value=results.precision.mean())
    
    #Joint model for fine-tuning
    train_sets = ["neill","monash","terns","penguins","pfeifer","hayes","everglades","USGS","mckellar","palmyra"]
    test_sets = ["palmyra"]
    recall, precision = train(path_dict=path_dict, config=config, train_sets=train_sets, test_sets=test_sets, comet_logger=comet_logger, save_dir=savedir)

    #log images
    with comet_logger.experiment.context_manager("validation"):
        images = glob.glob("{}/*.png".format(savedir))
        for img in images:
            comet_logger.experiment.log_image(img, image_scale=0.25)    
