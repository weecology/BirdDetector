#Prepare all training sets
import comet_ml
from model import BirdDetector
import glob
from PIL import ImageFile
from pytorch_lightning.loggers import CometLogger
from deepforest.model import create_model
from deepforest import main
from datetime import datetime

from utils.PR import precision_recall_curve
from utils.preprocess import *
from utils.prepare import *

import os
import pandas as pd
import torch
import gc
from pytorch_lightning.plugins import DDPPlugin
import subprocess

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
    #model.config["validation"]["csv_file"] = "/orange/ewhite/b.weinstein/generalization/crops/test_annotations.csv"
    #model.config["validation"]["root_dir"] = "/orange/ewhite/b.weinstein/generalization/crops"
        
    model.create_trainer(logger=comet_logger, plugins=DDPPlugin(find_unused_parameters=False))
    comet_logger.experiment.log_parameters(model.config)
    
    with comet_logger.experiment.context_manager(test_sets[0]):
        model.trainer.fit(model)
    
    for x in test_sets:
        test_results = model.evaluate(csv_file=path_dict[x]["test"], root_dir="/orange/ewhite/b.weinstein/generalization/crops/", iou_threshold=0.25, savedir=savedir)
        pr_data, plot = precision_recall_curve(model, csv_file=path_dict[x]["test"], root_dir="/orange/ewhite/b.weinstein/generalization/crops/", iou_threshold=0.25)
        comet_logger.experiment.log_figure()
        pr_data.to_csv("{}/precision_recall_curve.csv".format(savedir))
        comet_logger.experiment.log_asset("{}/precision_recall_curve.csv".format(savedir))
        if comet_logger is not None:
            try:
                test_results["results"].to_csv("{}/iou_dataframe.csv".format(savedir))
                comet_logger.experiment.log_asset("{}/iou_dataframe.csv".format(savedir))
                
                test_results["class_recall"].to_csv("{}/class_recall.csv".format(savedir))
                comet_logger.experiment.log_asset("{}/class_recall.csv".format(savedir))
                
                for index, row in test_results["class_recall"].iterrows():
                    comet_logger.experiment.log_metric("{}_Recall".format(row["label"]),row["recall"])
                    comet_logger.experiment.log_metric("{}_Precision".format(row["label"]),row["precision"])
                
                comet_logger.experiment.log_metric("Average Class Recall",test_results["class_recall"].recall.mean())
                comet_logger.experiment.log_metric("{} Box Recall".format(x),test_results["box_recall"])
                comet_logger.experiment.log_metric("{} Box Precision".format(x),test_results["box_precision"])
            except Exception as e:
                print(e)    
    if save_dir:
        try:
            model.trainer.save_checkpoint("{}/{}.pl".format(save_dir,"_".join(train_sets)))
        except Exception as e:
            print(e)        
            
    #delete model and free up memory
    del model
    torch.cuda.empty_cache()
    
    #The last position in the loop is the LOO score
    return test_results["box_recall"], test_results["box_precision"]

if __name__ =="__main__":
    #save original config during loop
    #comet_logger=None
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir="/orange/ewhite/b.weinstein/generalization/"
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
    train_list = ["monash","USGS","hayes","palmyra","terns","penguins","pfeifer"]
    results = []
    for x in train_list:
        train_sets = [y for y in train_list if not y==x]
        train_sets.append("everglades")
        test_sets = [x]
        recall, precision = train(path_dict=path_dict, config=config, train_sets=train_sets, test_sets=test_sets, comet_logger=comet_logger, save_dir=savedir)
        torch.cuda.empty_cache()
        gc.collect()
        result = pd.DataFrame({"test_sets":[x],"recall":[recall],"precision":[precision]})
        results.append(result)
    
    results = pd.concat(results)
    results.to_csv("Figures/generalization.csv")
    comet_logger.experiment.log_asset(file_data="Figures/generalization.csv", file_name="results.csv")
    comet_logger.experiment.log_metric(name="Mean LOO Recall", value=results.recall.mean())
    comet_logger.experiment.log_metric(name="Mean LOO Precision", value=results.precision.mean())
    
    #Joint model
    train_sets = ["monash","terns","palmyra","penguins","pfeifer","hayes","everglades","USGS"]
    test_sets = ["murres","pelicans","schedl"]
    recall, precision = train(path_dict=path_dict, config=config, train_sets=train_sets, test_sets=test_sets, comet_logger=comet_logger, save_dir=savedir)
    #Don't log validation scores till the end of project
    
    #log images
    with comet_logger.experiment.context_manager("validation"):
        images = glob.glob("{}/*.png".format(savedir))
        for img in images:
            comet_logger.experiment.log_image(img, image_scale=0.25)    
