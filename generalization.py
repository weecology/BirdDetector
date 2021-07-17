#Prepare all training sets
import comet_ml
from model import BirdDetector
from augmentation import get_transform
import glob
from PIL import ImageFile
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.plugins import DDPPlugin
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

def fit(model, train_annotations, comet_logger, name):
    #Give it a unique timestamp
    train_annotations["xmin"] = train_annotations["xmin"].astype(float) 
    train_annotations["xmax"] = train_annotations["xmax"].astype(float)
    train_annotations["ymin"] = train_annotations["ymin"].astype(float)
    train_annotations["ymax"] = train_annotations["ymax"].astype(float)
    
    train_annotations = train_annotations[~(train_annotations.xmin >= train_annotations.xmax)]
    train_annotations = train_annotations[~(train_annotations.ymin >= train_annotations.ymax)]
    train_annotations = train_annotations[["xmin","ymin","xmax","ymax","label","image_path"]]
    comet_logger.experiment.log_parameter("Training_data","/blue/ewhite/b.weinstein/generalization/crops/training_annotations_{}.csv".format(name))
    train_annotations.to_csv("/blue/ewhite/b.weinstein/generalization/crops/training_annotations_{}.csv".format(name))    

    model.config["train"]["csv_file"] = "/blue/ewhite/b.weinstein/generalization/crops/training_annotations_{}.csv".format(name)
    model.config["train"]["root_dir"] = "/blue/ewhite/b.weinstein/generalization/crops/"
    
    #model.config["validation"]["csv_file"] = "/blue/ewhite/b.weinstein/generalization/crops/{}_test.csv".format(name.split("_")[0])
    #model.config["validation"]["root_dir"] = "/blue/ewhite/b.weinstein/generalization/crops/"
    
    model.create_trainer(logger=comet_logger, plugins=DDPPlugin(find_unused_parameters=False))        
    model.trainer.fit(model)
    
    return model
    
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

def zero_shot(path_dict, train_sets, test_sets, comet_logger, savedir, config):
    try:
        image_save_dir = "{}/{}_zeroshot".format(savedir, test_sets[0])
        os.mkdir(image_save_dir)
    except Exception as e:
        print(e)
        
    all_sets = []
    print("Train sets: {}".format(train_sets))
    for x in train_sets:
        try:
            df = pd.read_csv(path_dict[x]["train"])
            all_sets.append(df)            
        except:
            raise ValueError("No training path supplied for {}".format(x))
        try:
            df_test = pd.read_csv(path_dict[x]["test"])
            all_sets.append(df_test)
        except Exception as e:
            print("No test set for {}".format(x))
    
    train_annotations = pd.concat(all_sets)
    
    #A couple illegal boxes, make slightly smaller
    train_annotations["xmin"] = train_annotations["xmin"].astype(float) 
    train_annotations["xmax"] = train_annotations["xmax"].astype(float) - 3
    train_annotations["ymin"] = train_annotations["ymin"].astype(float)
    train_annotations["ymax"] = train_annotations["ymax"].astype(float) - 3
    
    train_annotations = train_annotations[~(train_annotations.xmin >= train_annotations.xmax)]
    train_annotations = train_annotations[~(train_annotations.ymin >= train_annotations.ymax)]
    train_annotations.to_csv("/blue/ewhite/b.weinstein/generalization/crops/training_annotations.csv")

    all_val_sets = []
    for x in test_sets:
        df = pd.read_csv(path_dict[x]["test"])
        all_val_sets.append(df)
    test_annotations = pd.concat(all_val_sets)
    test_annotations.to_csv("/blue/ewhite/b.weinstein/generalization/crops/test_annotations.csv")

    comet_logger.experiment.log_parameter("training_images",len(train_annotations.image_path.unique()))
    comet_logger.experiment.log_parameter("training_annotations",train_annotations.shape[0])
    
    #train_df = pd.read_csv("/blue/ewhite/b.weinstein/AerialDetection/data/trainval1024/train.csv")
    #label_dict = {x: index for index, x in enumerate(train_df.label.unique())}    
    #pretrained_DOTA = main.deepforest(num_classes=15, label_dict=label_dict)
    #pretrained_DOTA.load_state_dict(torch.load("/orange/ewhite/b.weinstein/AerialDetection/snapshots/20210530_233702/DOTA.pl")["state_dict"])
    
    #update backbone weights with new Retinanet head
    model = BirdDetector(transforms = get_transform)    
    #model.model = create_model(num_classes=1, nms_thresh=model.config["nms_thresh"], score_thresh=model.config["score_thresh"], backbone=pretrained_DOTA.model.backbone)
    model.config = config
    model_path = "{}/{}_zeroshot.pt".format(savedir,test_sets[0])
    
    if os.path.exists(model_path):
        print("loading {}".format(model_path))
        model.model.load_state_dict(torch.load(model_path))
    else:
        model = fit(model, train_annotations, comet_logger, name = "{}_zeroshot".format(test_sets[0]))
        if savedir:
            if not model.config["train"]["fast_dev_run"]:
                torch.save(model.model.state_dict(),model_path)            
        
    for x in test_sets:
        test_results = model.evaluate(csv_file=path_dict[x]["test"], root_dir="/blue/ewhite/b.weinstein/generalization/crops/", iou_threshold=0.2, savedir=image_save_dir)
        if comet_logger is not None:
            try:
                test_results["results"].to_csv("{}/{}_iou_dataframe.csv".format(savedir, x))
                comet_logger.experiment.log_asset("{}/{}_iou_dataframe.csv".format(savedir, x))
                comet_logger.experiment.log_metric("{} Box Recall".format(x),test_results["box_recall"])
                comet_logger.experiment.log_metric("{} Box Precision".format(x),test_results["box_precision"])
            except Exception as e:
                print(e)    
        result_frame = pd.DataFrame({"test_set":[test_sets[0]],"Recall":[test_results["box_recall"]], "Precision":[test_results["box_precision"]],"Model":["Zero Shot"]})
    
    del model
    torch.cuda.empty_cache()
    gc.collect()  
    
    return result_frame

def fine_tune(dataset, comet_logger, savedir, config):
    try:
        image_save_dir = "{}/{}_finetune".format(savedir, dataset)
        os.mkdir(image_save_dir)
    except Exception as e:
        print(e)
        
    train_annotations = pd.read_csv("/blue/ewhite/b.weinstein/generalization/crops/{}_train.csv".format(dataset))
    model_path = "{}/{}_finetune.pt".format(savedir, dataset)
    model = BirdDetector(transforms = deepforest_transform)   
    model.config = config
    weights = "{}/{}_zeroshot.pt".format(savedir,dataset)
    model.model.load_state_dict(torch.load(weights))
    
    if os.path.exists(model_path):
        model.model.load_state_dict(torch.load(model_path))
    else:
        model = fit(model, train_annotations, comet_logger, "{}_finetune".format(dataset))
        if savedir:
            if not model.config["train"]["fast_dev_run"]:
                torch.save(model.model.state_dict(),model_path)            
    finetune_results = model.evaluate(csv_file="/blue/ewhite/b.weinstein/generalization/crops/{}_test.csv".format(dataset), root_dir="/blue/ewhite/b.weinstein/generalization/crops/", iou_threshold=0.2, savedir=image_save_dir)
    if comet_logger is not None:
        comet_logger.experiment.log_metric("Fine Tuned {} Box Recall".format(dataset),finetune_results["box_recall"])
        comet_logger.experiment.log_metric("Fine Tuned {} Box Precision".format(dataset),finetune_results["box_precision"])
        result_frame = pd.DataFrame({"test_set":[dataset],"Recall":[finetune_results["box_recall"]], "Precision":[finetune_results["box_precision"]],"Model":["Fine Tune"]})
    
    del model
    torch.cuda.empty_cache()
    gc.collect()
        
    return result_frame

def mini_fine_tune(dataset, comet_logger, config, savedir):
    #Fine tuning, up to 1000 birds from train
    min_annotation_results = []
    for i in range(5):
        try:
            image_save_dir = "{}/{}_mini_{}".format(savedir, dataset, i)
            os.mkdir(image_save_dir)
        except Exception as e:
            print(e)
            
        model_path = "{}/{}_mini_{}.pt".format(savedir, dataset,i)
        model = BirdDetector(transforms = deepforest_transform)   
        model.config = config
        weights = "{}/{}_zeroshot.pt".format(savedir,dataset)
        model.model.load_state_dict(torch.load(weights))
        
        if os.path.exists(model_path):
            model.model.load_state_dict(torch.load(model_path))
        else: 
            df = pd.read_csv("/blue/ewhite/b.weinstein/generalization/crops/{}_train.csv".format(dataset))            
            train_annotations = select(df, 1000)
            model = fit(model, train_annotations, comet_logger, "{}_mini_fine_tune".format(dataset))
            if savedir:
                if not model.config["train"]["fast_dev_run"]:
                    torch.save(model.model.state_dict(),model_path)
        finetune_results = model.evaluate(csv_file="/blue/ewhite/b.weinstein/generalization/crops/{}_test.csv".format(dataset), root_dir="/blue/ewhite/b.weinstein/generalization/crops/", iou_threshold=0.2, savedir=image_save_dir)
        if comet_logger is not None:
            comet_logger.experiment.log_metric("Fine Tuned 1000 {} Box Recall - Iteration {}".format(dataset, i),finetune_results["box_recall"])
            comet_logger.experiment.log_metric("Fine Tuned 1000 {} Box Precision - Iteration {}".format(dataset, i),finetune_results["box_precision"])
        min_annotation_results.append(pd.DataFrame({"Recall":finetune_results["box_recall"], "Precision":finetune_results["box_precision"],"test_set":dataset,"Iteration":[i],"Model":["Min Annotation"]}))
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
    min_annotation_results = pd.concat(min_annotation_results)
    
    return min_annotation_results

def mini_random_weights(dataset, comet_logger, config, savedir):
    #Fine tuning, up to 1000 birds from train
    min_annotation_results = []
    for i in range(5):
        try:
            image_save_dir = "{}/{}_random_{}".format(savedir, dataset, i)
            os.mkdir(image_save_dir)
        except Exception as e:
            print(e)
        
        model = BirdDetector(transforms = deepforest_transform)                   
        model_path = "{}/{}_random_{}.pt".format(savedir, dataset,i)        
        if os.path.exists(model_path):
            model.model.load_state_dict(torch.load(model_path))
        else: 
            #train_df = pd.read_csv("/blue/ewhite/b.weinstein/AerialDetection/data/trainval1024/train.csv")
            #label_dict = {x: index for index, x in enumerate(train_df.label.unique())}    
            #pretrained_DOTA = main.deepforest(num_classes=15, label_dict=label_dict)
            #pretrained_DOTA.load_state_dict(torch.load("/orange/ewhite/b.weinstein/AerialDetection/snapshots/20210530_233702/DOTA.pl")["state_dict"])
                    
            #update backbone weights with new Retinanet head
            #model.model = create_model(num_classes=1, nms_thresh=model.config["nms_thresh"], score_thresh=model.config["score_thresh"], backbone=pretrained_DOTA.model.backbone)
            model.config = config
            model.config["train"]["epochs"] = 30
            model.config["train"]["lr"] = 0.005
            
            df = pd.read_csv("/blue/ewhite/b.weinstein/generalization/crops/{}_train.csv".format(dataset))            
            train_annotations = select(df, 1000)
            model = fit(model, train_annotations, comet_logger,"{}_random".format(dataset))
            if savedir:
                if not model.config["train"]["fast_dev_run"]:
                    torch.save(model.model.state_dict(),model_path)
        finetune_results = model.evaluate(csv_file="/blue/ewhite/b.weinstein/generalization/crops/{}_test.csv".format(dataset), root_dir="/blue/ewhite/b.weinstein/generalization/crops/", iou_threshold=0.2, savedir=image_save_dir)
        if comet_logger is not None:
            comet_logger.experiment.log_metric("Random Weight 1000 {} Box Recall - Iteration {}".format(dataset, i),finetune_results["box_recall"])
            comet_logger.experiment.log_metric("Random Weight 1000 {} Box Precision - Iteration {}".format(dataset, i),finetune_results["box_precision"])
        min_annotation_results.append(pd.DataFrame({"Recall":finetune_results["box_recall"], "Precision":finetune_results["box_precision"],"test_set":dataset,"Iteration":[i],"Model":["RandomWeight"]}))
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
    min_annotation_results = pd.concat(min_annotation_results)
    
    return min_annotation_results

def run(path_dict, config, train_sets = ["penguins","terns","everglades","palmyra"],test_sets=["everglades"], comet_logger=None, savedir=None, run_fine_tune=True, run_mini=True, run_random=True):
    #Log experiment
    comet_logger.experiment.log_parameter("train_set",train_sets)
    comet_logger.experiment.log_parameter("test_set",test_sets)
    comet_logger.experiment.add_tag("Generalization")
    
    results = []
    zero_shot_results = zero_shot(path_dict=path_dict, train_sets=train_sets, test_sets=test_sets, config=config, comet_logger=comet_logger, savedir=savedir)
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
    if run_random:
        random_results = mini_random_weights(dataset=test_sets[0], config=config, savedir=savedir, comet_logger=comet_logger)
        results.append(random_results)
        
    result_frame = pd.concat(results)
    
    return result_frame