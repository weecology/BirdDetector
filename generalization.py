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
import subprocess
from time import sleep

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

def fit(model, train_annotations, comet_logger):
    train_annotations["xmin"] = train_annotations["xmin"].astype(float) 
    train_annotations["xmax"] = train_annotations["xmax"].astype(float)
    train_annotations["ymin"] = train_annotations["ymin"].astype(float)
    train_annotations["ymax"] = train_annotations["ymax"].astype(float)
    
    train_annotations = train_annotations[~(train_annotations.xmin >= train_annotations.xmax)]
    train_annotations = train_annotations[~(train_annotations.ymin >= train_annotations.ymax)]
    
    train_annotations.to_csv("/orange/ewhite/b.weinstein/generalization/crops/training_annotations.csv")    

    model.config["train"]["csv_file"] = "/orange/ewhite/b.weinstein/generalization/crops/training_annotations.csv"
    model.config["validation"]["csv_file"] = "/orange/ewhite/b.weinstein/generalization/crops/test_annotations.csv"
    model.config["train"]["root_dir"] = "/orange/ewhite/b.weinstein/generalization/crops/"
    model.config["validation"]["root_dir"] = "/orange/ewhite/b.weinstein/generalization/crops/"
    model.create_trainer(logger=comet_logger, num_sanity_val_steps=0)
    model.trainer.fit(model)
    
    return model
    
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

def zero_shot(train_sets, test_sets, comet_logger, savedir, config):
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
    
    train_df = pd.read_csv("/orange/ewhite/b.weinstein/AerialDetection/data/trainval1024/train.csv")
    label_dict = {x: index for index, x in enumerate(train_df.label.unique())}    
    pretrained_DOTA = main.deepforest(num_classes=15, label_dict=label_dict)
    model = BirdDetector(transforms = get_transform)
    
    #update backbone weights with new Retinanet head
    model.model = create_model(num_classes=1, nms_thresh=model.config["nms_thresh"], score_thresh=model.config["score_thresh"], backbone=pretrained_DOTA.model.backbone)
    model.config = config
    model_path = "{}/{}_zeroshot.pt".format(savedir,test_sets[0])
    
    if os.path.exists(model_path):
        print("loading {}".format(model_path))
        model.model.load_state_dict(torch.load(model_path))
    else:
        model = fit(model, train_annotations, comet_logger)
        if savedir:
            torch.save(model.model.state_dict(),model_path)                
        
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
        result_frame = pd.DataFrame({"test_set":[test_sets[0]],"Recall":[test_results["box_recall"]], "Precision":[test_results["box_recall"]],"Model":["FZero Shot"]})
    
    del model
    torch.cuda.empty_cache()
    gc.collect()  
    
    return result_frame

def fine_tune(dataset, comet_logger, savedir, config):
    train_annotations = pd.read_csv("/orange/ewhite/b.weinstein/generalization/crops/{}_train.csv".format(dataset))
    model_path = "{}/{}_finetune.pt".format(savedir, dataset)
    model = BirdDetector(transforms = deepforest_transform)   
    model.config = config
    weights = "{}/{}_zeroshot.pt".format(savedir,dataset)
    model.model.load_state_dict(torch.load(weights))
    
    if os.path.exists(model_path):
        model.model.load_state_dict(torch.load(model_path))
    else:
        model = fit(model, train_annotations, comet_logger)
        if savedir:
            torch.save(model.model.state_dict(),model_path)           
    finetune_results = model.evaluate(csv_file="/orange/ewhite/b.weinstein/generalization/crops/{}_test.csv".format(dataset), root_dir="/orange/ewhite/b.weinstein/generalization/crops/", iou_threshold=0.25)
    if comet_logger is not None:
        comet_logger.experiment.log_metric("Fine Tuned {} Box Recall".format(dataset),finetune_results["box_recall"])
        comet_logger.experiment.log_metric("Fine Tuned {} Box Precision".format(dataset),finetune_results["box_precision"])
        result_frame = pd.DataFrame({"test_set":[dataset],"Recall":[finetune_results["box_recall"]], "Precision":[finetune_results["box_recall"]],"Model":["Fine Tune"]})
    
    del model
    torch.cuda.empty_cache()
    gc.collect()
        
    return result_frame

def mini_fine_tune(dataset, comet_logger, config, savedir):
    #Fine tuning, up to 1000 birds from train
    min_annotation_results = []
    for i in range(5):
        model_path = "{}/{}_mini_{}.pt".format(savedir, dataset,i)
        model = BirdDetector(transforms = deepforest_transform)   
        model.config = config
        weights = "{}/{}_zeroshot.pt".format(savedir,dataset)
        model.model.load_state_dict(torch.load(weights))
        
        if os.path.exists(model_path):
            model.model.load_state_dict(torch.load(model_path))
        else: 
            df = pd.read_csv("/orange/ewhite/b.weinstein/generalization/crops/{}_train.csv".format(dataset))            
            train_annotations = select(df)
            model = fit(model, train_annotations, comet_logger)
            if savedir:
                torch.save(model.model.state_dict(),model_path)               
        finetune_results = model.evaluate(csv_file="/orange/ewhite/b.weinstein/generalization/crops/{}_test.csv".format(dataset), root_dir="/orange/ewhite/b.weinstein/generalization/crops/", iou_threshold=0.25)
        if comet_logger is not None:
            comet_logger.experiment.log_metric("Fine Tuned 1000 {} Box Recall - Iteration {}".format(dataset, i),finetune_results["box_recall"])
            comet_logger.experiment.log_metric("Fine Tuned 1000 {} Box Precision".format(dataset, i),finetune_results["box_precision"])
        min_annotation_results.append(pd.DataFrame({"Recall":finetune_results["box_recall"], "Precision":finetune_results["box_precision"],"test_set":test_sets[0],"Iteration":[i],"Model":["Min Annotation"]}))
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
    min_annotation_results = pd.concat(min_annotation_results)
    
    return min_annotation_results

def run(path_dict, config, train_sets = ["penguins","terns","everglades","palmyra"],test_sets=["everglades"], comet_logger=None, savedir=None, run_fine_tune=True, run_mini=True):
    #Log experiment
    comet_logger.experiment.log_parameter("train_set",train_sets)
    comet_logger.experiment.log_parameter("test_set",test_sets)
    comet_logger.experiment.add_tag("Generalization")
    
    results = []
    zero_shot_results = zero_shot(train_sets=train_sets, test_sets=test_sets, config=config, comet_logger=comet_logger, savedir=savedir)
    gc.collect()      
    results.append(zero_shot_results)
    if run_fine_tune:
        finetune_results = fine_tune(dataset=test_sets[0], comet_logger=comet_logger, config=config, savedir=savedir)
        gc.collect()          
        results.append(finetune_results)
    if run_mini:
        mini_results = mini_fine_tune(dataset=test_sets[0], config=config, savedir=savedir, comet_logger=comet_logger)
        gc.collect()          
        results.append(mini_results)

    result_frame = pd.concat(results)
    
    return result_frame

if __name__ =="__main__":
    #save original config during loop
    #comet_logger=None
    comet_logger = CometLogger(api_key="ypQZhYfs3nSyKzOfz13iuJpj2",
                                project_name="everglades", workspace="bw4sz",auto_output_logging = "simple")
    
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    existing_dir = '20210622_185244'
    savedir="/orange/ewhite/b.weinstein/generalization"    
    if existing_dir is None:   
        sleep(random.randint(0,20))        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.mkdir(savedir)
        comet_logger.experiment.log_parameter("timestamp",timestamp)        
    else:
        savedir = "{}/{}".format(savedir,existing_dir)                      
        comet_logger.experiment.log_parameter("timestamp",existing_dir)        
        
    model = BirdDetector(transforms=get_transform)
    config = model.config
    
    path_dict = prepare()

    #Log commit
    comet_logger.experiment.log_parameter("commit hash",subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip())
    comet_logger.experiment.log_parameters(model.config)
    #view_training(path_dict, comet_logger=comet_logger)

    #Train Models
    train_list = ["seabirdwatch","neill","USGS","hayes","terns","penguins","pfeifer","palmyra","mckellar","monash"]
    results = []
    for x in train_list:
        train_sets = [y for y in train_list if not y==x]
        train_sets.append("everglades")
        test_sets = [x]
        try:
            result = run(path_dict=path_dict,
                         config=config,
                         train_sets=train_sets,
                         test_sets=test_sets,
                         comet_logger=comet_logger,
                         savedir=savedir)
            results.append(result)
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print("{} failed with {}".format(train_sets, e))
            torch.cuda.empty_cache()
            gc.collect()              
            continue
        
    results = pd.concat(results)
    results.to_csv("Figures/generalization.csv")
    
    mean_zero_shot_recall = results[results.Model == "Zero Shot"].Recall.mean()
    mean_zero_shot_precision = results[results.Model == "Zero Shot"].Precision.mean()
    
    mean_fine_tune_recall = results[results.Model == "Fine Tune"].Recall.mean()
    mean_fine_tune_precision = results[results.Model == "Fine Tune"].Precision.mean()
    
    comet_logger.experiment.log_asset(file_data="Figures/generalization.csv", file_name="results.csv")
    comet_logger.experiment.log_metric(name="Mean LOO Recall", value=mean_zero_shot_recall)
    comet_logger.experiment.log_metric(name="Mean LOO Precision", value=mean_zero_shot_precision)
    
    comet_logger.experiment.log_metric(name="Mean Fine Tune Recall", value=mean_fine_tune_recall)
    comet_logger.experiment.log_metric(name="Mean Fine Tune Precision", value=mean_fine_tune_precision)
    
    #Joint model for fine-tuning
    train_sets = ["seabirdwatch","neill","monash","terns","penguins","pfeifer","hayes","everglades","USGS","mckellar","palmyra"]
    test_sets = ["palmyra"]
    result = run(path_dict=path_dict,
                            config=config,
                            train_sets=train_sets,
                            test_sets=test_sets,
                            comet_logger=comet_logger,
                            savedir=savedir)

    #log images
    with comet_logger.experiment.context_manager("validation"):
        images = glob.glob("{}/*.png".format(savedir))
        for img in images:
            comet_logger.experiment.log_image(img, image_scale=0.25)    
